"""Heat-pump performance and cost models for H2Integrate.

The performance model wraps two selectable modes:

- ``"fixed_cop"``: constant, user-supplied Coefficient of Performance (COP).
- ``"carnot"``: temperature-dependent COP computed from a Carnot efficiency
  factor and the source / sink temperatures.

Both modes deliver heat at a fixed target ``delivery_temp_C`` supply
temperature, drawing low-grade heat from an upstream source (e.g., data-center
waste heat) and consuming electricity. Delivery is capped at the rated
``system_capacity_mw_th`` and by the available source-heat and electricity
inputs.
"""

import numpy as np
from attrs import field, define, validators

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import (
    CostModelBaseClass,
    CostModelBaseConfig,
    PerformanceModelBaseClass,
)


# Absolute lower/upper clamps on COP used to keep the compute step numerically
# well-behaved for infeasible or degenerate temperature pairs.
_MIN_COP = 1.0
_MAX_COP = 20.0


@define(kw_only=True)
class HeatPumpPerformanceConfig(BaseConfig):
    """Configuration class for :class:`HeatPumpPerformanceModel`.

    Attributes:
        hp_mode (str): Either ``"fixed_cop"`` or ``"carnot"``.
        system_capacity_mw_th (float): Rated thermal output capacity in MW.
        delivery_temp_C (float): Target supply (sink) temperature in degC.
        cop (float, optional): Constant COP, required when
            ``hp_mode == "fixed_cop"``.
        carnot_efficiency (float, optional): Second-law efficiency of the heat
            pump used in Carnot mode. Typical range 0.3 - 0.6. Defaults to 0.5.
        min_source_temp_C (float, optional): If provided in Carnot mode,
            timesteps whose source temperature falls below this value are
            treated as infeasible (no heat delivered).
    """

    hp_mode: str = field(validator=validators.in_(("fixed_cop", "carnot")))
    system_capacity_mw_th: float = field(validator=validators.gt(0))
    delivery_temp_C: float = field(validator=validators.gt(0))
    cop: float = field(default=None)
    carnot_efficiency: float = field(default=0.5)
    min_source_temp_C: float = field(default=None)

    def __attrs_post_init__(self):
        if self.hp_mode == "fixed_cop":
            if self.cop is None:
                raise ValueError("'cop' must be provided when hp_mode == 'fixed_cop'.")
            if self.cop < 1.0:
                raise ValueError("'cop' must be >= 1.0 for a heat pump.")
        elif self.hp_mode == "carnot":
            if self.carnot_efficiency <= 0 or self.carnot_efficiency > 1.0:
                raise ValueError("'carnot_efficiency' must be in the interval (0, 1].")


class HeatPumpPerformanceModel(PerformanceModelBaseClass):
    """Heat-pump performance model.

    Delivers heat at ``delivery_temp_C`` by drawing low-grade heat from
    ``heat_in`` and consuming electricity from ``electricity_in``. The
    per-timestep COP is either the fixed configured value or the
    Carnot-limited value ``COP = eta_carnot * T_sink / (T_sink - T_source)``.
    The delivered heat is capped by (a) the rated capacity, (b) the available
    source heat plus the electricity that can be drawn given the COP, and
    (c) the available electricity.

    Notes:
        Uses the standard heat commodity so its ``heat_out`` output
        auto-connects to a downstream tech's ``heat_in`` input via the
        commodity naming convention.
    """

    _time_step_bounds = (3600, 3600)

    def initialize(self):
        super().initialize()
        self.commodity = "heat"
        self.commodity_rate_units = "MW"
        self.commodity_amount_units = "MW*h"

    def setup(self):
        self.config = HeatPumpPerformanceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        n_timesteps = self.n_timesteps

        # Inputs
        self.add_input(
            "heat_in",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc="Low-grade heat available from the source",
        )
        self.add_input(
            "heat_supply_temp_C_in",
            val=0.0,
            units="degC",
            desc="Source-side supply temperature",
        )
        self.add_input(
            "electricity_in",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc="Available electricity input",
        )
        self.add_input(
            "system_capacity",
            val=self.config.system_capacity_mw_th,
            units="MW",
            desc="Rated thermal output capacity",
        )
        self.add_input(
            "delivery_temp_C",
            val=self.config.delivery_temp_C,
            units="degC",
            desc="Target supply (sink) temperature",
        )
        if self.config.hp_mode == "fixed_cop":
            self.add_input(
                "cop",
                val=self.config.cop,
                units="unitless",
                desc="Fixed Coefficient of Performance",
            )
        else:
            self.add_input(
                "carnot_efficiency",
                val=self.config.carnot_efficiency,
                units="unitless",
                desc="Second-law (Carnot) efficiency of the heat pump",
            )

        # Outputs
        self.add_output(
            "heat_supply_temp_C",
            val=self.config.delivery_temp_C,
            units="degC",
            desc="Delivered heat supply temperature",
        )
        self.add_output(
            "electricity_used",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc="Electricity consumed by the heat pump",
        )
        self.add_output(
            "cop_actual",
            val=0.0,
            shape=n_timesteps,
            units="unitless",
            desc="Actual per-timestep COP",
        )
        self.add_output(
            "unmet_heat_demand",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc="Heat delivery shortfall vs. requested source heat",
        )
        self.add_output(
            "unmet_electricity_demand",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc="Electricity shortfall relative to what would be needed for full delivery",
        )

    def _compute_cop(self, source_temp_C, delivery_temp_C, inputs):
        """Return a scalar or vector COP appropriate for the configured mode."""
        n = self.n_timesteps
        if self.config.hp_mode == "fixed_cop":
            return np.full(n, float(np.asarray(inputs["cop"]).item()))

        eta = float(np.asarray(inputs["carnot_efficiency"]).item())
        t_sink = float(delivery_temp_C) + 273.15
        t_src = float(source_temp_C) + 273.15
        # Infeasible or degenerate operating point: force minimum COP; caller
        # additionally zeros deliveries at infeasible timesteps.
        if t_sink <= t_src:
            return np.full(n, _MIN_COP)
        cop = eta * t_sink / (t_sink - t_src)
        cop = float(np.clip(cop, _MIN_COP, _MAX_COP))
        return np.full(n, cop)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        heat_in = np.asarray(inputs["heat_in"], dtype=float)
        elec_in = np.asarray(inputs["electricity_in"], dtype=float)
        capacity = float(np.asarray(inputs["system_capacity"]).item())
        source_temp_C = float(np.asarray(inputs["heat_supply_temp_C_in"]).item())
        delivery_temp_C = float(np.asarray(inputs["delivery_temp_C"]).item())

        cop = self._compute_cop(source_temp_C, delivery_temp_C, inputs)

        # Feasibility mask: for Carnot mode we cannot lift heat when the source
        # is at or above the target sink temperature and the user has not
        # relaxed the constraint.
        feasible = np.ones(self.n_timesteps, dtype=bool)
        if self.config.hp_mode == "carnot":
            if source_temp_C + 273.15 >= delivery_temp_C + 273.15:
                feasible[:] = False
            if self.config.min_source_temp_C is not None:
                if source_temp_C < self.config.min_source_temp_C:
                    feasible[:] = False

        # Energy balance: Q_delivered = Q_source + W_elec ; COP = Q_delivered / W_elec
        # => Q_source = Q_delivered * (1 - 1/COP), W_elec = Q_delivered / COP
        one_over_cop = np.where(cop > 0, 1.0 / cop, 0.0)
        source_frac = 1.0 - one_over_cop  # fraction of delivered heat from the source

        # Requested delivery is the rated capacity per timestep.
        requested = np.full(self.n_timesteps, capacity)

        # Maximum delivery bound by available source heat.
        # avoid divide-by-zero when source_frac is 0 (would only happen for COP=1 exactly)
        with np.errstate(divide="ignore", invalid="ignore"):
            max_from_source = np.where(source_frac > 0, heat_in / source_frac, np.inf)
        # Maximum delivery bound by available electricity.
        max_from_elec = elec_in * cop

        delivered = np.minimum.reduce([requested, max_from_source, max_from_elec])
        delivered = np.where(feasible, delivered, 0.0)
        delivered = np.clip(delivered, 0.0, capacity)

        elec_used = delivered * one_over_cop
        source_used = delivered * source_frac

        outputs["heat_out"] = delivered
        outputs["heat_supply_temp_C"] = delivery_temp_C
        outputs["electricity_used"] = elec_used
        outputs["cop_actual"] = np.where(feasible, cop, 0.0)
        outputs["unmet_heat_demand"] = np.maximum(0.0, heat_in - source_used)
        outputs["unmet_electricity_demand"] = np.maximum(
            0.0, (requested * one_over_cop) - elec_used
        )

        # Standard base-class outputs
        total_heat = float(np.sum(delivered) * (self.dt / 3600))
        outputs["total_heat_produced"] = total_heat
        outputs["annual_heat_produced"] = total_heat / self.fraction_of_year_simulated
        outputs["rated_heat_production"] = capacity
        max_production = capacity * self.n_timesteps * (self.dt / 3600)
        outputs["capacity_factor"] = total_heat / max_production if max_production > 0 else 0.0


@define(kw_only=True)
class HeatPumpCostConfig(CostModelBaseConfig):
    """Configuration class for :class:`HeatPumpCostModel`.

    Attributes:
        system_capacity_mw_th (float): Rated thermal capacity in MW.
        capex_per_mw_th (float): Capital cost per MW of thermal capacity in USD/MW.
        fixed_opex_per_mw_th_per_year (float): Fixed annual O&M in USD/(MW*year).
        variable_opex_per_mwh_th (float): Variable O&M per delivered MWh_th.
    """

    system_capacity_mw_th: float = field(validator=validators.gt(0))
    capex_per_mw_th: float = field(validator=validators.ge(0))
    fixed_opex_per_mw_th_per_year: float = field(validator=validators.ge(0))
    variable_opex_per_mwh_th: float = field(default=0.0, validator=validators.ge(0))


class HeatPumpCostModel(CostModelBaseClass):
    """Simple CapEx/OpEx cost model for a heat pump."""

    _time_step_bounds = (3600, 3600)

    def setup(self):
        self.config = HeatPumpCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        self.add_input(
            "system_capacity",
            val=self.config.system_capacity_mw_th,
            units="MW",
            desc="Rated thermal capacity",
        )
        self.add_input(
            "heat_out",
            val=0.0,
            shape=self.n_timesteps,
            units="MW",
            desc="Delivered heat profile from performance model",
        )
        self.add_input(
            "capex_per_mw_th",
            val=self.config.capex_per_mw_th,
            units="USD/MW",
            desc="Capital cost per MW of thermal capacity",
        )
        self.add_input(
            "fixed_opex_per_mw_th_per_year",
            val=self.config.fixed_opex_per_mw_th_per_year,
            units="USD/(MW*year)",
            desc="Fixed O&M per MW of thermal capacity",
        )
        self.add_input(
            "variable_opex_per_mwh_th",
            val=self.config.variable_opex_per_mwh_th,
            units="USD/(MW*h)",
            desc="Variable O&M per delivered MWh_th",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        capacity = float(np.asarray(inputs["system_capacity"]).item())
        delivered_MWh = float(inputs["heat_out"].sum()) * (self.dt / 3600)

        capex = float(np.asarray(inputs["capex_per_mw_th"]).item()) * capacity
        fixed_om = float(np.asarray(inputs["fixed_opex_per_mw_th_per_year"]).item()) * capacity
        variable_om = float(np.asarray(inputs["variable_opex_per_mwh_th"]).item()) * delivered_MWh

        outputs["CapEx"] = capex
        outputs["OpEx"] = fixed_om + variable_om

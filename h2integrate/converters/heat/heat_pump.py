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
from h2integrate.core.model_baseclass import (
    CostModelBaseClass,
    CostModelBaseConfig,
    PerformanceModelBaseClass,
)


# Absolute lower/upper clamps on COP used to keep the compute step numerically
# well-behaved for infeasible or degenerate temperature pairs.
_MIN_COP = 1.0
_MAX_COP = 20.0


@define(kw_only=True)
class HeatPumpPerformanceModelConfig(BaseConfig):
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
        heat_demand_profile (list | np.ndarray | float | None): Hourly ceiling
            (MW) on the heat that the off-taker can absorb. Interpretation
            depends on ``throttle_to_demand``. Defaults to ``None``
            (unbounded; the HP always runs to its supply-limited potential).
        throttle_to_demand (bool): Dispatch policy against
            ``heat_demand_profile``:

            * ``True`` (default): the HP throttles down to the off-taker
              demand ceiling every timestep, so ``heat_out ==
              heat_delivered`` and no electricity is consumed for
              unutilized heat. ``heat_curtailed`` reports production
              potential foregone due to insufficient demand.
            * ``False``: the HP always runs to its supply-limited
              production potential; ``heat_out`` is that potential and
              ``heat_delivered = min(heat_out, heat_demand_profile)`` is
              the fraction the off-taker absorbs. Excess heat above
              demand is dumped as ``heat_curtailed`` and the electricity
              to produce it is still consumed and billed.
    """

    hp_mode: str = field(validator=validators.in_(("fixed_cop", "carnot")))
    system_capacity_mw_th: float = field(validator=validators.gt(0))
    delivery_temp_C: float = field(validator=validators.gt(0))
    cop: float = field(default=None)
    carnot_efficiency: float = field(default=0.5)
    min_source_temp_C: float = field(default=None)
    heat_demand_profile: list | np.ndarray | float | None = field(default=None)
    throttle_to_demand: bool = field(default=True)

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
    The supply-limited production potential is capped by (a) the rated
    capacity, (b) the available source heat plus the electricity that can
    be drawn given the COP, and (c) the available electricity. How the
    off-taker demand ``heat_demand_profile`` interacts with that potential
    is set by ``throttle_to_demand`` (see :class:`HeatPumpPerformanceModelConfig`).

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
        self.config = HeatPumpPerformanceModelConfig.from_dict(
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
        # Downstream off-taker demand ceiling. Interpretation depends on
        # ``throttle_to_demand`` (see class-level config docs). Defaults to
        # a very large constant so callers that do not wire this input see
        # unbounded-demand behavior (identical output in both modes).
        demand_val = (
            self.config.heat_demand_profile
            if self.config.heat_demand_profile is not None
            else 1.0e12
        )
        self.add_input(
            "heat_demand_profile",
            val=demand_val,
            shape=n_timesteps,
            units="MW",
            desc="Hourly ceiling on delivered heat (dispatch policy set by throttle_to_demand)",
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
        self.add_output(
            "electricity_demand",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc=(
                "Electricity required to produce the capacity-, source-, and demand-limited "
                "heat output; independent of electricity_in, so suitable as a grid set point"
            ),
        )
        # Off-taker delivery outputs. Under the default
        # ``throttle_to_demand=True`` policy, ``heat_delivered`` is
        # identical to ``heat_out`` (the HP only produces what the
        # off-taker can take) and ``heat_curtailed`` reports production
        # potential foregone due to insufficient demand. Under
        # ``throttle_to_demand=False``, ``heat_out`` is the supply-limited
        # production, ``heat_delivered`` is the portion the off-taker
        # absorbs, and ``heat_curtailed`` is heat produced-and-dumped
        # above demand. In both modes, ``heat_delivered`` is the canonical
        # revenue-billing stream for downstream cost accounting.
        self.add_output(
            "heat_delivered",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc="Heat delivered to the off-taker (dispatch policy set by throttle_to_demand)",
        )
        self.add_output(
            "heat_curtailed",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc=(
                "Under throttle_to_demand=True: production potential foregone due to "
                "insufficient off-taker demand. Under throttle_to_demand=False: produced "
                "heat dumped above off-taker demand."
            ),
        )
        self.add_output(
            "annual_heat_delivered",
            val=0.0,
            units="MW*h/yr",
            desc="Annualized delivered heat (scaled by fraction_of_year_simulated)",
        )
        self.add_output(
            "annual_heat_curtailed",
            val=0.0,
            units="MW*h/yr",
            desc="Annualized production potential foregone due to insufficient off-taker demand",
        )
        self.add_output(
            "dh_utilization_fraction",
            val=0.0,
            units="unitless",
            desc="Fraction of supply-limited production potential delivered to the off-taker",
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
        heat_demand = np.asarray(inputs["heat_demand_profile"], dtype=float)

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

        # ``potential`` = the heat the HP could deliver this timestep given
        # nameplate capacity, source-heat, and electricity supply.
        potential = np.minimum.reduce([requested, max_from_source, max_from_elec])
        potential = np.where(feasible, potential, 0.0)
        potential = np.clip(potential, 0.0, capacity)

        # Dispatch policy against the demand ceiling. Under ``throttle_to_demand``
        # (default) the HP scales its production down so the compressor only
        # consumes electricity for heat that is actually utilized. Under the
        # legacy policy, the HP always runs to ``potential`` and any output
        # above demand is dumped as ``heat_curtailed`` (electricity for it
        # is still consumed).
        heat_demand_clipped = np.maximum(heat_demand, 0.0)
        if self.config.throttle_to_demand:
            heat_out = np.minimum(potential, heat_demand_clipped)
            heat_delivered = heat_out
            heat_curtailed = np.maximum(potential - heat_out, 0.0)
        else:
            heat_out = potential
            heat_delivered = np.minimum(heat_out, heat_demand_clipped)
            heat_curtailed = np.maximum(heat_out - heat_delivered, 0.0)

        # Electricity and source consumption are always billed on ``heat_out``
        # (the physically produced heat). In throttled mode heat_out equals
        # heat_delivered, so no compressor power is billed for unutilized
        # heat; in the legacy mode it includes the curtailed portion.
        elec_used = heat_out * one_over_cop
        source_used = heat_out * source_frac

        outputs["heat_out"] = heat_out
        outputs["heat_supply_temp_C"] = delivery_temp_C
        outputs["electricity_used"] = elec_used
        outputs["cop_actual"] = np.where(feasible, cop, 0.0)
        outputs["unmet_heat_demand"] = np.maximum(0.0, heat_in - source_used)
        # Electricity shortfall reflects only supply-side limits (capacity vs.
        # source/electricity availability), independent of any demand
        # throttling.
        outputs["unmet_electricity_demand"] = np.maximum(
            0.0, (requested * one_over_cop) - potential * one_over_cop
        )

        # Electricity required for the heat output that is achievable without
        # any electricity limit (capacity, source heat, feasibility, and, when
        # throttling, the demand ceiling). This does not depend on
        # ``electricity_in``, so it can drive an upstream grid set point
        # without creating an unstable feedback loop.
        source_limited = np.minimum(requested, max_from_source)
        source_limited = np.clip(np.where(feasible, source_limited, 0.0), 0.0, capacity)
        if self.config.throttle_to_demand:
            source_limited = np.minimum(source_limited, np.maximum(heat_demand, 0.0))
        outputs["electricity_demand"] = source_limited * one_over_cop

        outputs["heat_delivered"] = heat_delivered
        outputs["heat_curtailed"] = heat_curtailed
        delivered_MWh_sim = float(heat_delivered.sum()) * (self.dt / 3600)
        curtailed_MWh_sim = float(heat_curtailed.sum()) * (self.dt / 3600)
        outputs["annual_heat_delivered"] = delivered_MWh_sim / self.fraction_of_year_simulated
        outputs["annual_heat_curtailed"] = curtailed_MWh_sim / self.fraction_of_year_simulated
        # Utilization = delivered / (production potential). Meaningful and
        # identical under both dispatch policies (they differ only in whether
        # the un-utilized portion is produced-and-dumped or never made).
        total_potential = float(potential.sum())
        outputs["dh_utilization_fraction"] = (
            float(heat_delivered.sum()) / total_potential if total_potential > 0 else 0.0
        )

        # Standard base-class outputs (reflect physically produced heat).
        total_heat = float(np.sum(heat_out) * (self.dt / 3600))
        outputs["total_heat_produced"] = total_heat
        outputs["annual_heat_produced"] = total_heat / self.fraction_of_year_simulated
        outputs["rated_heat_production"] = capacity
        max_production = capacity * self.n_timesteps * (self.dt / 3600)
        outputs["capacity_factor"] = total_heat / max_production if max_production > 0 else 0.0


@define(kw_only=True)
class HeatPumpCostModelConfig(CostModelBaseConfig):
    """Configuration class for :class:`HeatPumpCostModel`.

    Attributes:
        system_capacity_mw_th (float): Rated thermal capacity in MW.
        capex_per_mw_th (float): Capital cost per MW of thermal capacity in USD/MW.
        fixed_opex_per_mw_th_per_year (float): Fixed annual O&M in USD/(MW*year).
        variable_opex_per_mwh_th (float): Variable O&M per **delivered** MWh_th.
        dh_connection_distance_m (float): Length of buried supply/return
            interconnection pipe from the HP to the off-taker (m). Zero
            disables trench cost accounting.
        dh_pipe_fixed_cost_usd_per_m (float): Diameter-independent installed
            cost of the interconnection pipe (USD/m). Includes trenching,
            labor, fittings.
        dh_pipe_cost_per_mm_diameter_usd_per_m (float): Diameter-linear
            installed cost of the interconnection pipe
            (USD per m per mm of nominal diameter).
        dh_trench_bulk_cost_usd_per_ft (float | None): Optional bulk
            trench-and-pipe installed cost expressed in USD per trench
            foot. When not ``None``, this value overrides the
            diameter-based cost model above (``dh_pipe_fixed_cost_usd_per_m``
            and ``dh_pipe_cost_per_mm_diameter_usd_per_m`` are ignored)
            and the CapEx is computed as
            ``dh_trench_bulk_cost_usd_per_ft * dh_connection_distance_m
            * 3.28084 ft/m``. The pipe diameter is still reported for
            reference. Useful when the trenching contractor has already
            quoted a lump ``$/ft`` number.
        dh_pipe_annual_om_fraction (float): Annual O&M for the buried
            interconnection pipe, as a fraction of its installed capex.
        dh_pipe_diameter_coeff_mm_per_sqrt_mw (float): Coefficient in the
            DH pipe-sizing rule ``D_mm = coeff * sqrt(Q_peak_MW)``. Default
            of 71 corresponds to a 30 K supply/return delta at ~2 m/s pipe
            velocity (standard DH design point).
        heat_sell_price_usd_per_mwh_th (float | list | None): Sell price for
            delivered heat (USD/MWh_th). ``None`` disables revenue accounting.
            When ``sell_price_mode == "per_year"``, must be a scalar or a list
            of length ``plant_life`` (annual escalation series).
        sell_price_mode (str): ``"per_year"`` (default; annual escalation
            series) or ``"constant"`` (single value applied every year).
    """

    system_capacity_mw_th: float = field(validator=validators.gt(0))
    capex_per_mw_th: float = field(validator=validators.ge(0))
    fixed_opex_per_mw_th_per_year: float = field(validator=validators.ge(0))
    variable_opex_per_mwh_th: float = field(default=0.0, validator=validators.ge(0))
    dh_connection_distance_m: float = field(default=0.0, validator=validators.ge(0))
    dh_pipe_fixed_cost_usd_per_m: float = field(default=0.0, validator=validators.ge(0))
    dh_pipe_cost_per_mm_diameter_usd_per_m: float = field(default=0.0, validator=validators.ge(0))
    dh_trench_bulk_cost_usd_per_ft: float | None = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    dh_pipe_annual_om_fraction: float = field(default=0.0, validator=validators.ge(0))
    dh_pipe_diameter_coeff_mm_per_sqrt_mw: float = field(default=71.0, validator=validators.ge(0))
    heat_sell_price_usd_per_mwh_th: float | list[float] | np.ndarray | None = field(default=None)
    sell_price_mode: str = field(
        default="per_year", validator=validators.in_(["per_year", "constant"])
    )


class HeatPumpCostModel(CostModelBaseClass):
    """Cost model for a heat pump plus its off-taker interconnection.

    CapEx and fixed OpEx cover both the heat-pump unit itself and, optionally,
    a buried supply/return pipe to the off-taker (e.g., a district-heat
    network). Variable OpEx is billed on the *delivered* heat (``heat_delivered``
    from the performance model, capped by the off-taker's demand ceiling), not
    the produced heat, so curtailed heat does not incur water-treatment or
    other volumetric charges.

    When ``heat_sell_price_usd_per_mwh_th`` is provided, revenue from delivered
    heat is subtracted from ``VarOpEx`` so the finance models (e.g.
    ``ProFastLCO``, ``NumpyFinancialNPV``) pick it up as a negative cash flow
    without any additional wiring.
    """

    _time_step_bounds = (3600, 3600)

    def setup(self):
        self.config = HeatPumpCostModelConfig.from_dict(
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
            "heat_delivered",
            val=0.0,
            shape=self.n_timesteps,
            units="MW",
            desc="Delivered heat profile from the performance model (off-taker cap applied)",
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

        # Heat-sell-price input, added only when configured. Scalar for
        # ``constant`` mode, plant_life-length series for ``per_year``.
        if self.config.heat_sell_price_usd_per_mwh_th is not None:
            self._sell_price_mode = self.config.sell_price_mode
            if self._sell_price_mode == "per_year":
                price_shape = self.plant_life
            else:  # "constant"
                price_shape = 1
            self.add_input(
                "heat_sell_price",
                val=self.config.heat_sell_price_usd_per_mwh_th,
                shape=price_shape,
                units="USD/(MW*h)",
                desc="Sell price for delivered heat",
            )
        else:
            self._sell_price_mode = None

        # Bookkeeping outputs so the DC-to-DH interconnection cost can be
        # inspected independently of the HP unit costs.
        self.add_output(
            "dh_pipe_diameter_mm",
            val=0.0,
            units="mm",
            desc="Sized nominal diameter of the DC-to-DH interconnection pipe",
        )
        self.add_output(
            "dh_connection_capex",
            val=0.0,
            units="USD",
            desc="Installed capex of the DC-to-DH interconnection (trench + pipe)",
        )
        self.add_output(
            "dh_connection_opex",
            val=0.0,
            units="USD/year",
            desc="Annual O&M of the DC-to-DH interconnection",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        capacity = float(np.asarray(inputs["system_capacity"]).item())
        heat_delivered = np.asarray(inputs["heat_delivered"], dtype=float)
        delivered_MWh_sim = float(heat_delivered.sum()) * (self.dt / 3600)
        delivered_MWh_annual = (
            delivered_MWh_sim / self.fraction_of_year_simulated
            if self.fraction_of_year_simulated > 0
            else 0.0
        )

        # HP unit costs
        capex_hp = float(np.asarray(inputs["capex_per_mw_th"]).item()) * capacity
        fixed_om_hp = float(np.asarray(inputs["fixed_opex_per_mw_th_per_year"]).item()) * capacity

        # DC-to-DH interconnection: pipe diameter is always sized from HP
        # capacity via a standard DH design rule (kept for reporting even
        # when the trench cost is quoted in bulk). By default, installed
        # cost has diameter-independent and diameter-linear terms; if the
        # user supplied a bulk ``$/ft`` trench price, that value overrides
        # the diameter-based model. O&M is a fixed fraction of installed
        # capex either way.
        d_mm = self.config.dh_pipe_diameter_coeff_mm_per_sqrt_mw * float(
            np.sqrt(max(capacity, 0.0))
        )
        if self.config.dh_trench_bulk_cost_usd_per_ft is not None:
            # 1 m = 3.28084 ft.
            cost_per_m = self.config.dh_trench_bulk_cost_usd_per_ft * 3.28084
        else:
            cost_per_m = (
                self.config.dh_pipe_fixed_cost_usd_per_m
                + self.config.dh_pipe_cost_per_mm_diameter_usd_per_m * d_mm
            )
        dh_capex = cost_per_m * self.config.dh_connection_distance_m
        dh_opex = dh_capex * self.config.dh_pipe_annual_om_fraction

        # Variable O&M on delivered heat (per year, replicated over plant life).
        variable_om_annual = (
            float(np.asarray(inputs["variable_opex_per_mwh_th"]).item()) * delivered_MWh_annual
        )
        variable_om_series = np.full(self.plant_life, variable_om_annual)

        # Revenue from delivered heat, expressed as negative VarOpEx. Consumed
        # by the finance model like any other cash flow.
        if self._sell_price_mode is None:
            revenue_series = np.zeros(self.plant_life)
        else:
            price = np.asarray(inputs["heat_sell_price"], dtype=float)
            if self._sell_price_mode == "constant":
                revenue_series = np.full(
                    self.plant_life, float(price.item()) * delivered_MWh_annual
                )
            else:  # per_year
                revenue_series = price * delivered_MWh_annual

        outputs["CapEx"] = capex_hp + dh_capex
        outputs["OpEx"] = fixed_om_hp + dh_opex
        outputs["VarOpEx"] = variable_om_series - revenue_series
        outputs["dh_pipe_diameter_mm"] = d_mm
        outputs["dh_connection_capex"] = dh_capex
        outputs["dh_connection_opex"] = dh_opex

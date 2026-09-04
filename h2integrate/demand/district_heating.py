"""District-heating demand and cost models for H2Integrate.

The performance model behaves like :class:`GenericDemandComponent` for the
``heat`` commodity, but additionally checks the incoming supply temperature
against a required minimum. When ``strict_temperature`` is true (the default),
heat delivered at insufficient supply temperatures is treated as unusable
(``heat_out = 0`` at those timesteps and the demand is counted as unmet).
"""

import numpy as np
from attrs import field, define, validators

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.demand.demand_base import DemandComponentBase, DemandComponentBaseConfig
from h2integrate.core.model_baseclasses import CostModelBaseClass, CostModelBaseConfig


@define(kw_only=True)
class DistrictHeatingDemandConfig(DemandComponentBaseConfig):
    """Configuration for :class:`DistrictHeatingDemand`.

    Attributes:
        min_supply_temp_C (float): Minimum acceptable supply temperature for
            the district-heating loop in degC.
        system_capacity_mw_th (float): Connected thermal capacity in MW.
        strict_temperature (bool): When True, timesteps whose supply
            temperature is below ``min_supply_temp_C`` yield zero delivered
            heat and full demand shortfall. When False, the temperature is
            reported and counted but delivery is not blocked.
    """

    min_supply_temp_C: float = field(validator=validators.gt(0))
    system_capacity_mw_th: float = field(validator=validators.gt(0))
    strict_temperature: bool = field(default=True)


class DistrictHeatingDemand(DemandComponentBase):
    """District-heating heat consumer with a supply-temperature check.

    Follows the :class:`GenericDemandComponent` pattern for the heat
    commodity: consumes ``heat_in`` against a ``heat_demand`` profile, and
    reports ``unmet_heat_demand_out``, ``unused_heat_out``, and ``heat_out``.
    Additionally reports temperature-shortfall metrics based on the incoming
    ``heat_supply_temp_C_in`` scalar.
    """

    _time_step_bounds = (3600, 3600)

    def setup(self):
        self.config = DistrictHeatingDemandConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=True,
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        self.add_input(
            "heat_supply_temp_C_in",
            val=self.config.min_supply_temp_C,
            units="degC",
            desc="Incoming supply temperature from the upstream heat source",
        )
        self.add_input(
            "min_supply_temp_C",
            val=self.config.min_supply_temp_C,
            units="degC",
            desc="Minimum acceptable district-heating supply temperature",
        )
        self.add_input(
            "system_capacity",
            val=self.config.system_capacity_mw_th,
            units="MW",
            desc="Connected thermal capacity",
        )

        self.add_output(
            "temperature_shortfall_flag",
            val=0.0,
            units="unitless",
            desc="1 if incoming supply temperature is below the minimum, 0 otherwise",
        )
        self.add_output(
            "temperature_shortfall_hours",
            val=0.0,
            units="h",
            desc=(
                "Total simulation-hours during which supply temperature is below "
                "the required minimum"
            ),
        )

    def compute(self, inputs, outputs):
        commodity_in = np.asarray(inputs[f"{self.commodity}_in"], dtype=float)
        commodity_demand = np.asarray(inputs[f"{self.commodity}_demand"], dtype=float)

        supply_temp_C = float(np.asarray(inputs["heat_supply_temp_C_in"]).item())
        min_supply_temp_C = float(np.asarray(inputs["min_supply_temp_C"]).item())
        temperature_ok = supply_temp_C >= min_supply_temp_C

        if not temperature_ok and self.config.strict_temperature:
            # Reject all incoming heat at insufficient temperature.
            commodity_in = np.zeros_like(commodity_in)

        outputs = self.calculate_outputs(commodity_in, commodity_demand, outputs)

        outputs["temperature_shortfall_flag"] = 0.0 if temperature_ok else 1.0
        # With steady incoming temperature this is either zero or the full
        # simulation length in hours; when the temperature is time-varying in
        # future extensions, this will need to be computed per timestep.
        hours_simulated = (self.dt / 3600) * self.n_timesteps
        outputs["temperature_shortfall_hours"] = 0.0 if temperature_ok else hours_simulated


@define(kw_only=True)
class DistrictHeatingDemandCostConfig(CostModelBaseConfig):
    """Configuration for :class:`DistrictHeatingDemandCostModel`.

    Attributes:
        system_capacity_mw_th (float): Connected thermal capacity in MW.
        capex_per_mw_th (float): Capital cost per MW of connected capacity.
        fixed_opex_per_mw_th_per_year (float): Fixed annual O&M.
        variable_opex_per_mwh_th (float): Variable O&M per delivered MWh_th.
    """

    system_capacity_mw_th: float = field(validator=validators.gt(0))
    capex_per_mw_th: float = field(validator=validators.ge(0))
    fixed_opex_per_mw_th_per_year: float = field(validator=validators.ge(0))
    variable_opex_per_mwh_th: float = field(default=0.0, validator=validators.ge(0))


class DistrictHeatingDemandCostModel(CostModelBaseClass):
    """Simple CapEx/OpEx cost model for a district-heating heat consumer."""

    _time_step_bounds = (3600, 3600)

    def setup(self):
        self.config = DistrictHeatingDemandCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        self.add_input(
            "system_capacity",
            val=self.config.system_capacity_mw_th,
            units="MW",
            desc="Connected thermal capacity",
        )
        self.add_input(
            "heat_out",
            val=0.0,
            shape=self.n_timesteps,
            units="MW",
            desc="Delivered heat from the demand performance model",
        )
        self.add_input(
            "capex_per_mw_th",
            val=self.config.capex_per_mw_th,
            units="USD/MW",
            desc="Capital cost per MW of connected thermal capacity",
        )
        self.add_input(
            "fixed_opex_per_mw_th_per_year",
            val=self.config.fixed_opex_per_mw_th_per_year,
            units="USD/(MW*year)",
            desc="Fixed O&M per MW of connected thermal capacity",
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

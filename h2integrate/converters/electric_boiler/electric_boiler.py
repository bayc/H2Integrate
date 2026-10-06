"""
Electric boiler for steam-plant models.

Demand-following, steady-state electric steam boiler: given a steam demand [kg/s] it
produces exactly that steam and back-calculates the electricity draw, blowdown and
boiler feedwater need. It mirrors :class:`NgBoilerPerformance` so a plant can swap fuel
for electricity while reusing the same steam header and recovery loop. Direct CO2 is
zero; an optional grid carbon intensity gives an indirect (Scope 2) CO2 figure.

The saturated-steam thermodynamics reuse the CoolProp/IAPWS-IF97 helpers from the
packaged steam-recovery models (valid at the low process-steam pressures used here),
not the 3-16 MPa polynomial correlations of the standalone dynamic model.

Components:
    ElectricBoilerPerformance - steam demand -> electricity, blowdown, feedwater, CO2.
    ElectricBoilerCost        - CapEx/OpEx (electricity cost handled by the feedstock).
"""

import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import CostModelBaseClass, CostModelBaseConfig
from h2integrate.core.commodity_stream_definitions import multivariable_streams
from h2integrate.converters.heat.steam_recovery_models import (
    STREAM,
    Tsat_from_P,
    h_from_TP,
    hg_from_P,
)

_C2K = 273.15


def _gt_zero(instance, attribute, value):
    if value <= 0:
        raise ValueError(f"{attribute.name} must be > 0, got {value}.")


def _gte_zero(instance, attribute, value):
    if value < 0:
        raise ValueError(f"{attribute.name} must be >= 0, got {value}.")


# ===========================================================================
# Electric boiler, demand-following steady-state map.
# ===========================================================================
@define(kw_only=True)
class ElectricBoilerPerformanceConfig(BaseConfig):
    """Configuration for the electric boiler performance model.

    Attributes:
        rated_heat_mw: Rated heat duty into steam [MW].
        steam_pressure_bar: Steam (HP header) pressure [bar].
        feedwater_temp_c: Boiler feedwater temperature [degC].
        min_load_frac: Minimum firing rate as a fraction of full load [-].
        blowdown_fraction: Continuous blowdown as a fraction of feedwater [-].
        eta_heater: Electric-to-thermal conversion efficiency [-].
        grid_co2_intensity: Indirect grid CO2 intensity [kg/kWh] (0 disables Scope 2 CO2).
    """

    rated_heat_mw: float = field(default=10.0, validator=_gt_zero)
    steam_pressure_bar: float = field(default=12.0, validator=_gt_zero)
    feedwater_temp_c: float = field(default=105.0, validator=_gt_zero)
    min_load_frac: float = field(default=0.0, validator=_gte_zero)
    blowdown_fraction: float = field(default=0.02, validator=_gte_zero)
    eta_heater: float = field(default=0.99, validator=_gt_zero)
    grid_co2_intensity: float = field(default=0.0, validator=_gte_zero)


class ElectricBoilerPerformance(om.ExplicitComponent):
    """Demand-following electric boiler: steam demand -> electricity, blowdown, feedwater."""

    _time_step_bounds = (3600, 3600)  # feedstock costing is hourly

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = ElectricBoilerPerformanceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.dt = int(self.options["plant_config"]["plant"]["simulation"]["dt"])

        c = self.config
        self._hs = hg_from_P(c.steam_pressure_bar)  # saturated-vapor enthalpy [kJ/kg]
        self._hf = h_from_TP(c.feedwater_temp_c + _C2K, c.steam_pressure_bar)  # feedwater h [kJ/kg]
        self._t_steam = Tsat_from_P(c.steam_pressure_bar) - _C2K  # saturated steam T [degC]
        self._s_full = (c.rated_heat_mw * 1e3) / max(self._hs - self._hf, 1e-6)  # rated steam kg/s

        self.add_input("steam_demand", val=0.0, shape=self.n, units="kg/s",
                       desc="Requested steam production")
        # Supplied electricity (required by the feedstock auto-connection; self-computes use).
        self.add_input("electricity_in", val=0.0, shape=self.n, units="kW",
                       desc="Electricity supplied by the connected electricity feedstock")

        for var_name, var_props in multivariable_streams[STREAM].items():
            self.add_output(f"{STREAM}:{var_name}_out", val=0.0, shape=self.n,
                            units=var_props.get("units"), desc=var_props.get("desc", ""))

        self.add_output("electricity_consumed", val=0.0, shape=self.n, units="kW",
                        desc="Electricity consumption (read by the electricity feedstock cost)")
        self.add_output("feedwater_demand", val=0.0, shape=self.n, units="kg/s",
                        desc="Boiler feedwater requirement (steam + blowdown)")
        self.add_output("blowdown", val=0.0, shape=self.n, units="kg/s", desc="Continuous blowdown")
        self.add_output("co2_out", val=0.0, shape=self.n, units="kg/h",
                        desc="Indirect (grid) CO2 emissions; 0 direct")
        self.add_output("total_electricity", val=0.0, units="kW*h",
                        desc="Electricity over the simulation")
        self.add_output("total_co2", val=0.0, units="kg", desc="Indirect CO2 over the simulation")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        c = self.config
        d = np.clip(inputs["steam_demand"], 0.0, self._s_full)   # delivered steam [kg/s]
        on = d > 1e-9

        # Enforce a minimum load when firing (electric boilers can also idle fully off).
        s_min = c.min_load_frac * self._s_full
        d = np.where(on & (d < s_min), s_min, d)

        heat_kW = d * (self._hs - self._hf)          # heat into steam [kW]
        elec_kW = heat_kW / c.eta_heater             # electricity draw [kW]

        # Blowdown and feedwater requirement (feedwater = steam + blowdown).
        feedwater = d / (1.0 - c.blowdown_fraction)
        blowdown = feedwater - d

        # Indirect (grid) CO2: kg/h = kg/kWh * kW.
        co2_kg_h = c.grid_co2_intensity * elec_kW

        outputs[f"{STREAM}:mass_flow_out"] = d
        outputs[f"{STREAM}:temperature_out"] = np.full(self.n, self._t_steam)
        outputs[f"{STREAM}:pressure_out"] = np.full(self.n, c.steam_pressure_bar)
        outputs[f"{STREAM}:quality_out"] = np.where(on, 1.0, 0.0)   # saturated vapor when firing
        outputs[f"{STREAM}:phase_out"] = np.ones(self.n)           # on the saturation line
        outputs["electricity_consumed"] = elec_kW
        outputs["feedwater_demand"] = feedwater
        outputs["blowdown"] = blowdown
        outputs["co2_out"] = co2_kg_h
        outputs["total_electricity"] = float(np.sum(elec_kW) * (self.dt / 3600.0))
        outputs["total_co2"] = float(np.sum(co2_kg_h) * (self.dt / 3600.0))


@define(kw_only=True)
class ElectricBoilerCostConfig(CostModelBaseConfig):
    """CapEx/OpEx config for the electric boiler (electricity cost is on the feedstock).

    Attributes:
        rated_heat_mw: Rated heat duty [MW], for sizing capex/opex.
        eta_heater: Electric-to-thermal conversion efficiency, to size the electric rating [-].
        capex_per_kw_electric: Overnight capital cost [USD/kW_electric].
        fixed_opex_per_kw_yr: Fixed O&M [USD/kW_electric/year].
    """

    rated_heat_mw: float = field(default=10.0, validator=_gt_zero)
    eta_heater: float = field(default=0.99, validator=_gt_zero)
    capex_per_kw_electric: float = field(default=120.0, validator=_gte_zero)
    fixed_opex_per_kw_yr: float = field(default=3.0, validator=_gte_zero)


class ElectricBoilerCost(CostModelBaseClass):
    """CapEx/OpEx cost model for the electric boiler."""

    _time_step_bounds = (3600, 3600)

    def setup(self):
        self.config = ElectricBoilerCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )
        super().setup()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        kw_e = self.config.rated_heat_mw * 1e3 / self.config.eta_heater  # electric rating [kW_e]
        outputs["CapEx"] = self.config.capex_per_kw_electric * kw_e
        outputs["OpEx"] = self.config.fixed_opex_per_kw_yr * kw_e
        if discrete_outputs is not None:
            discrete_outputs["cost_year"] = self.config.cost_year

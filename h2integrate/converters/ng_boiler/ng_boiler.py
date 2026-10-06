"""
Natural-gas firetube boiler for steam-plant models.

An H2Integrate-native, demand-following port of the standalone steady-state NG boiler
model (``NgFiretubeBoiler._steady_state``): given a steam demand [kg/s] it produces
exactly that steam and back-calculates fuel use, emissions, blowdown, boiler feedwater
need and parasitic electricity. The dynamic ODE model is intentionally not used - an
8760 annual run needs a per-timestep steady-state map.

Components:
    BoilerLeadLagDispatcher - splits total steam demand into lead/lag boiler shares.
    NgBoilerPerformance     - demand-following boiler physics (steam, NG, CO2/NOx/CO).
    NgBoilerCost            - CapEx/OpEx (fuel cost handled by the NG FeedstockCostModel).
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
_DESIGN_FR = 0.769  # design firing rate (fraction of full fire), from the validated model


def _gt_zero(instance, attribute, value):
    if value <= 0:
        raise ValueError(f"{attribute.name} must be > 0, got {value}.")


def _gte_zero(instance, attribute, value):
    if value < 0:
        raise ValueError(f"{attribute.name} must be >= 0, got {value}.")


# ===========================================================================
# Lead/lag dispatcher: the lead boiler fires first up to its rated steam, the
# lag boiler trims the remainder.
# ===========================================================================
@define(kw_only=True)
class BoilerLeadLagDispatcherConfig(BaseConfig):
    """Configuration for the lead/lag boiler dispatcher.

    Attributes:
        lead_rated_steam_kg_s: Rated steam output of the lead boiler [kg/s].
    """

    lead_rated_steam_kg_s: float = field(validator=_gt_zero)


class BoilerLeadLagDispatcher(om.ExplicitComponent):
    """Splits total steam demand into lead and lag boiler demands."""

    _time_step_bounds = (1, int(1e9))

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = BoilerLeadLagDispatcherConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.add_input("total_steam_demand", val=0.0, shape=self.n, units="kg/s")
        self.add_output("steam_demand_lead", val=0.0, shape=self.n, units="kg/s")
        self.add_output("steam_demand_lag", val=0.0, shape=self.n, units="kg/s")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total = inputs["total_steam_demand"]
        rated = self.config.lead_rated_steam_kg_s
        lead = np.minimum(total, rated)
        outputs["steam_demand_lead"] = lead
        outputs["steam_demand_lag"] = np.maximum(0.0, total - rated)


# ===========================================================================
# NG firetube boiler, demand-following steady-state map.
# ===========================================================================
@define(kw_only=True)
class NgBoilerPerformanceConfig(BaseConfig):
    """Configuration for the NG firetube boiler performance model.

    Attributes:
        rated_heat_mw: Rated heat duty into steam [MW].
        steam_pressure_bar: Steam (HP header) pressure [bar].
        feedwater_temp_c: Boiler feedwater temperature [degC].
        excess_air_pct: Combustion excess air [%].
        min_load_frac: Minimum firing rate as a fraction of full fire [-].
        blowdown_fraction: Continuous blowdown as a fraction of feedwater [-].
        fuel_lhv_mj_kg: Fuel lower heating value [MJ/kg].
        fuel_afr_stoic: Stoichiometric air/fuel mass ratio [-].
        thermal_efficiency: Thermal efficiency at the design point [-].
        parasitic_fraction: Parasitic electricity as a fraction of heat duty [-].
    """

    rated_heat_mw: float = field(default=10.0, validator=_gt_zero)
    steam_pressure_bar: float = field(default=12.0, validator=_gt_zero)
    feedwater_temp_c: float = field(default=105.0, validator=_gt_zero)
    excess_air_pct: float = field(default=10.0, validator=_gte_zero)
    min_load_frac: float = field(default=0.25, validator=_gte_zero)
    blowdown_fraction: float = field(default=0.03, validator=_gte_zero)
    fuel_lhv_mj_kg: float = field(default=50.05, validator=_gt_zero)
    fuel_afr_stoic: float = field(default=17.16, validator=_gt_zero)
    thermal_efficiency: float = field(default=0.915, validator=_gt_zero)
    parasitic_fraction: float = field(default=0.005, validator=_gte_zero)


class NgBoilerPerformance(om.ExplicitComponent):
    """Demand-following NG firetube boiler: steam demand -> fuel, emissions, blowdown."""

    _time_step_bounds = (3600, 3600)  # feedstock costing is hourly

    # Emission model constants (Zeldovich thermal NOx / Arrhenius CO), validated model.
    _A_NOX, _B_NOX = 3.681e6, 25168.0
    _A_CO, _B_CO = 10.24, 629.7

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = NgBoilerPerformanceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.dt = int(self.options["plant_config"]["plant"]["simulation"]["dt"])

        c = self.config
        self._hs = hg_from_P(c.steam_pressure_bar)  # saturated-vapor enthalpy [kJ/kg]
        self._hf = h_from_TP(c.feedwater_temp_c + _C2K, c.steam_pressure_bar)  # feedwater h [kJ/kg]
        self._t_steam = Tsat_from_P(c.steam_pressure_bar) - _C2K  # saturated steam T [degC]
        self._s_full = (c.rated_heat_mw * 1e3) / max(self._hs - self._hf, 1e-6)  # rated steam kg/s
        self._lambda = 1.0 + c.excess_air_pct / 100.0

        self.add_input("steam_demand", val=0.0, shape=self.n, units="kg/s",
                       desc="Requested steam production")
        # Supplied NG (required by the feedstock auto-connection; boiler self-computes use).
        self.add_input("natural_gas_in", val=0.0, shape=self.n, units="MMBtu/h",
                       desc="Natural gas supplied by the connected NG feedstock")

        for var_name, var_props in multivariable_streams[STREAM].items():
            self.add_output(f"{STREAM}:{var_name}_out", val=0.0, shape=self.n,
                            units=var_props.get("units"), desc=var_props.get("desc", ""))

        self.add_output("natural_gas_consumed", val=0.0, shape=self.n, units="MMBtu/h",
                        desc="Natural gas consumption (read by the NG feedstock cost model)")
        self.add_output("feedwater_demand", val=0.0, shape=self.n, units="kg/s",
                        desc="Boiler feedwater requirement (steam + blowdown)")
        self.add_output("blowdown", val=0.0, shape=self.n, units="kg/s", desc="Continuous blowdown")
        self.add_output("co2_out", val=0.0, shape=self.n, units="kg/h", desc="CO2 emissions")
        self.add_output("nox_out", val=0.0, shape=self.n, units="g/s", desc="NOx emissions")
        self.add_output("co_out", val=0.0, shape=self.n, units="g/s", desc="CO emissions")
        self.add_output("electricity_demand", val=0.0, shape=self.n, units="kW",
                        desc="Parasitic electricity (negative = consumed)")
        self.add_output("total_natural_gas", val=0.0, units="MMBtu", desc="NG over the simulation")
        self.add_output("total_co2", val=0.0, units="kg", desc="CO2 over the simulation")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        c = self.config
        d = np.clip(inputs["steam_demand"], 0.0, self._s_full)   # delivered steam [kg/s]
        on = d > 1e-9

        # Firing rate and part-load efficiency (quadratic penalty away from design).
        fr = np.where(on, np.clip(d / self._s_full, c.min_load_frac, 1.0), 0.0)
        lr = np.where(on, fr / _DESIGN_FR, 1.0)
        eta = np.clip(c.thermal_efficiency - 0.5 * (lr - 1.0) ** 2, 0.75, c.thermal_efficiency)

        q_net = d * (self._hs - self._hf)                         # heat into steam [kW]
        q_gross = np.where(on, q_net / eta, 0.0)                  # fuel heat [kW]
        lhv_kj = c.fuel_lhv_mj_kg * 1e3                           # [kJ/kg]
        mf = q_gross / lhv_kj                                     # fuel mass flow [kg/s]
        m_fg = (1.0 + self._lambda * c.fuel_afr_stoic) * mf       # flue-gas mass flow [kg/s]

        # Thermal NOx / CO from flame temperature (rises with firing rate).
        t_k = np.where(on, np.maximum(500.0, 1950.0 * fr) + _C2K, _C2K)
        o2_frac = 0.21 * (self._lambda - 1.0) / self._lambda
        p_o2 = max(0.01, o2_frac * 10.0)
        nox_ppm = np.where(on, self._A_NOX * np.exp(-self._B_NOX / t_k) * np.sqrt(p_o2)
                           * np.sqrt(0.15 / 0.10), 0.0)
        co_ppm = np.where(on, self._A_CO * np.exp(-t_k / self._B_CO)
                          / max(self._lambda - 1.0, 0.01), 0.0)

        # Blowdown and feedwater requirement (feedwater = steam + blowdown).
        feedwater = d / (1.0 - c.blowdown_fraction)
        blowdown = feedwater - d

        outputs[f"{STREAM}:mass_flow_out"] = d
        outputs[f"{STREAM}:temperature_out"] = np.full(self.n, self._t_steam)
        outputs[f"{STREAM}:pressure_out"] = np.full(self.n, c.steam_pressure_bar)
        outputs[f"{STREAM}:quality_out"] = np.where(on, 1.0, 0.0)   # saturated vapor when firing
        outputs[f"{STREAM}:phase_out"] = np.ones(self.n)           # on the saturation line
        # Natural gas as energy [MMBtu/h] (LHV basis): mf[kg/s]*3600*LHV[MJ/kg]/1055.06.
        ng_mmbtu_h = mf * 3600.0 * c.fuel_lhv_mj_kg / 1055.06
        outputs["natural_gas_consumed"] = ng_mmbtu_h
        outputs["feedwater_demand"] = feedwater
        outputs["blowdown"] = blowdown
        outputs["co2_out"] = mf * (44.01 / 16.04) * 3600.0
        outputs["nox_out"] = nox_ppm * 1e-6 * m_fg * 1e3
        outputs["co_out"] = co_ppm * 1e-6 * m_fg * 1e3
        outputs["electricity_demand"] = -c.parasitic_fraction * q_net
        outputs["total_natural_gas"] = float(np.sum(ng_mmbtu_h) * (self.dt / 3600.0))
        outputs["total_co2"] = float(np.sum(mf * (44.01 / 16.04)) * self.dt)


@define(kw_only=True)
class NgBoilerCostConfig(CostModelBaseConfig):
    """CapEx/OpEx config for the NG boiler (fuel cost is on the NG feedstock).

    Attributes:
        rated_heat_mw: Rated heat duty [MW], for sizing capex/opex.
        capex_per_kw: Overnight capital cost [USD/kW_th].
        fixed_opex_per_kw_yr: Fixed O&M [USD/kW_th/year].
    """

    rated_heat_mw: float = field(default=10.0, validator=_gt_zero)
    capex_per_kw: float = field(default=214.0, validator=_gte_zero)
    fixed_opex_per_kw_yr: float = field(default=4.3, validator=_gte_zero)


class NgBoilerCost(CostModelBaseClass):
    """CapEx/OpEx cost model for the NG firetube boiler."""

    _time_step_bounds = (3600, 3600)

    def setup(self):
        self.config = NgBoilerCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )
        super().setup()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        kw = self.config.rated_heat_mw * 1e3
        outputs["CapEx"] = self.config.capex_per_kw * kw
        outputs["OpEx"] = self.config.fixed_opex_per_kw_yr * kw
        if discrete_outputs is not None:
            discrete_outputs["cost_year"] = self.config.cost_year

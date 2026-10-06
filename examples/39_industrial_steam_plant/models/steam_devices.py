"""
Example-local steam devices for the industrial steam-plant example (example 39).

These are plain OpenMDAO ``ExplicitComponent`` models loaded via ``model_location``.
They reuse the CoolProp/IAPWS-IF97 helpers and the shared multivariable ``steam``
stream (mass_flow [kg/s], temperature [degC], pressure [bar], quality [-], phase [-])
from the packaged steam-recovery models, so the thermodynamics stay consistent with
the steam recovery loop.

Components:
    SteamLoadProfiles  - exogenous demand source (breaks the boiler<->process recycle;
                         defines the plant load from the three process demands).
    SteamHeaderSplitterPerformance - 1 steam inlet -> N outlets by per-outlet demand.
    PressureReducingValve          - isenthalpic throttle (HP steam -> LP steam).
    SteamProcessPerformance/Cost   - generic steam consumer: steam -> product +
                                     lower-quality condensate + steam loss.

The ``phase`` code (0 subcooled liquid, 1 saturated, 2 superheated vapor) is set on
every steam outlet.
"""

import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import CostModelBaseClass, CostModelBaseConfig
from h2integrate.core.commodity_stream_definitions import (
    multivariable_streams,
    add_multivariable_input,
)
from h2integrate.converters.heat.steam_recovery_models import (
    _C2K,
    STREAM,
    Tsat_from_P,
    h_from_TP,
    h_from_stream,
    phase_from_TPq,
    state_from_hP,
)


def _gt_zero(instance, attribute, value):
    if value <= 0:
        raise ValueError(f"{attribute.name} must be > 0, got {value}.")


def _gte_zero(instance, attribute, value):
    if value < 0:
        raise ValueError(f"{attribute.name} must be >= 0, got {value}.")


# ===========================================================================
# Steam load profiles: exogenous driver that defines the plant load from the
# three process demands. Having a pure source (no steam inlet) for the demand
# signals keeps the boiler -> header -> process -> loop graph acyclic (feed-forward),
# so no OpenMDAO solver is needed around the recycle.
# ===========================================================================
@define(kw_only=True)
class SteamLoadProfilesConfig(BaseConfig):
    """Configuration for the plant steam-load driver.

    Each process demand is a mean [kg/s]; a synthetic diurnal + seasonal swing is added
    (set the amplitude fractions to 0 for a constant load). These are placeholders until
    measured plant 8760 profiles are available.

    Attributes:
        hp_process_demand: Mean HP-process steam demand [kg/s].
        lp_process_demand: Mean LP-process steam demand [kg/s].
        lp_utility_demand: Mean LP-utility steam demand [kg/s].
        header_loss_fraction: Main-header steam loss as a fraction of process demand [-].
        diurnal_amplitude_frac: Peak diurnal swing as a fraction of the mean [-].
        seasonal_amplitude_frac: Peak seasonal swing as a fraction of the mean [-].
        random_seed: Seed for small multiplicative noise (None disables noise).
    """

    hp_process_demand: float = field(validator=_gt_zero)
    lp_process_demand: float = field(validator=_gt_zero)
    lp_utility_demand: float = field(validator=_gt_zero)
    header_loss_fraction: float = field(default=0.01, validator=_gte_zero)
    diurnal_amplitude_frac: float = field(default=0.25, validator=_gte_zero)
    seasonal_amplitude_frac: float = field(default=0.15, validator=_gte_zero)
    random_seed: int | None = field(default=42)


class SteamLoadProfiles(om.ExplicitComponent):
    """Emits per-consumer steam demands [kg/s] plus the total boiler steam demand."""

    _time_step_bounds = (1, int(1e9))

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = SteamLoadProfilesConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        for name, desc in [
            ("hp_process_demand", "HP-process steam demand"),
            ("lp_process_demand", "LP-process steam demand"),
            ("lp_utility_demand", "LP-utility steam demand"),
            ("header_loss_demand", "Main-header steam loss"),
            ("total_steam_demand", "Total boiler steam demand"),
        ]:
            self.add_output(name, val=0.0, shape=self.n, units="kg/s", desc=desc)

    def _synth(self, mean, rng):
        """Synthetic diurnal + seasonal load profile [kg/s] around ``mean``."""
        t = np.arange(self.n)
        diurnal = self.config.diurnal_amplitude_frac * np.sin(2.0 * np.pi * (t % 24) / 24.0)
        seasonal = self.config.seasonal_amplitude_frac * np.cos(2.0 * np.pi * t / max(self.n, 1))
        noise = rng.normal(0.0, 0.03, self.n) if rng is not None else 0.0
        return np.maximum(0.0, mean * (1.0 + diurnal + seasonal + noise))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        seed = self.config.random_seed
        rng = np.random.default_rng(seed) if seed is not None else None
        hp = self._synth(self.config.hp_process_demand, rng)
        lpp = self._synth(self.config.lp_process_demand, rng)
        lpu = self._synth(self.config.lp_utility_demand, rng)
        header_loss = self.config.header_loss_fraction * (hp + lpp + lpu)
        outputs["hp_process_demand"] = hp
        outputs["lp_process_demand"] = lpp
        outputs["lp_utility_demand"] = lpu
        outputs["header_loss_demand"] = header_loss
        outputs["total_steam_demand"] = hp + lpp + lpu + header_loss


# ===========================================================================
# Steam header splitter: one boiler-steam inlet split into N outlets by a
# prescribed per-outlet demand [kg/s]. Intensive properties (T, P, quality,
# phase) are preserved on every outlet; only mass flow is divided. Auto-indexed
# outputs (steam:<var>_out1, _out2, ...) because the tech name contains "splitter".
# ===========================================================================
@define(kw_only=True)
class SteamHeaderSplitterConfig(BaseConfig):
    """Configuration for the steam header splitter.

    Attributes:
        commodity: Multivariable stream name (must be 'steam').
        out_streams: Number of outlet ports.
    """

    commodity: str = field(default=STREAM)
    out_streams: int = field(default=2, validator=_gt_zero)

    def __attrs_post_init__(self):
        if self.commodity not in multivariable_streams:
            raise ValueError(f"Unknown stream '{self.commodity}'.")


class SteamHeaderSplitterPerformance(om.ExplicitComponent):
    """Splits one steam inlet into N outlets by per-outlet demand, preserving state."""

    _time_step_bounds = (1, int(1e9))

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = SteamHeaderSplitterConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        stream = self.config.commodity
        stream_def = multivariable_streams[stream]

        add_multivariable_input(self, stream, self.n)
        # One prescribed demand [kg/s] and one full set of steam outputs per outlet.
        for j in range(1, self.config.out_streams + 1):
            self.add_input(
                f"demand_out{j}", val=0.0, shape=self.n, units="kg/s",
                desc=f"Prescribed steam demand for outlet {j}",
            )
            for var_name, var_props in stream_def.items():
                self.add_output(
                    f"{stream}:{var_name}_out{j}",
                    val=0.0,
                    shape=self.n,
                    units=var_props.get("units"),
                    desc=f"Outlet {j}: {var_props.get('desc', '')}",
                )
        self.add_output(
            "header_loss_flow", val=0.0, shape=self.n, units="kg/s",
            desc="Unallocated inlet steam lost at the header",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        stream = self.config.commodity
        ns = self.config.out_streams
        m_in = inputs[f"{stream}:mass_flow_in"]
        demands = [inputs[f"demand_out{j}"] for j in range(1, ns + 1)]
        total_demand = sum(demands)

        # Scale outlet flows to the available inlet if demand exceeds supply.
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(
                total_demand > m_in, m_in / np.where(total_demand > 0, total_demand, 1.0), 1.0
            )

        allocated = np.zeros_like(m_in)
        for j in range(1, ns + 1):
            out_flow = demands[j - 1] * scale
            allocated = allocated + out_flow
            outputs[f"{stream}:mass_flow_out{j}"] = out_flow
            # Intensive properties are identical to the inlet on every outlet.
            for var in ("temperature", "pressure", "quality", "phase"):
                outputs[f"{stream}:{var}_out{j}"] = inputs[f"{stream}:{var}_in"]
        # Any inlet steam not sent to an outlet is lost at the header.
        outputs["header_loss_flow"] = np.maximum(0.0, m_in - allocated)


# ===========================================================================
# Pressure reducing valve (PRV): isenthalpic throttle from the inlet pressure
# down to a set outlet pressure. Enthalpy and mass are conserved; temperature,
# quality and phase are recovered from (h_in, P_out) with CoolProp.
# ===========================================================================
@define(kw_only=True)
class PressureReducingValveConfig(BaseConfig):
    """Configuration for a pressure reducing valve.

    Attributes:
        outlet_pressure: Reduced downstream pressure [bar].
    """

    outlet_pressure: float = field(validator=_gt_zero)


class PressureReducingValve(om.ExplicitComponent):
    """Isenthalpic steam throttle to a set outlet pressure."""

    _time_step_bounds = (1, int(1e9))

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = PressureReducingValveConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        add_multivariable_input(self, STREAM, self.n)
        for var_name, var_props in multivariable_streams[STREAM].items():
            self.add_output(
                f"{STREAM}:{var_name}_out", val=0.0, shape=self.n,
                units=var_props.get("units"), desc=var_props.get("desc", ""),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m = inputs[f"{STREAM}:mass_flow_in"]
        T_in = inputs[f"{STREAM}:temperature_in"]
        P_in = inputs[f"{STREAM}:pressure_in"]
        q_in = inputs[f"{STREAM}:quality_in"]
        ph_in = inputs[f"{STREAM}:phase_in"]

        # Throttling conserves enthalpy and mass; only pressure is set.
        h = h_from_stream(T_in, P_in, q_in, ph_in)
        P_out = np.full(self.n, self.config.outlet_pressure)
        T_out_K, q_out, ph_out = state_from_hP(h, P_out)

        outputs[f"{STREAM}:mass_flow_out"] = m
        outputs[f"{STREAM}:temperature_out"] = T_out_K - _C2K
        outputs[f"{STREAM}:pressure_out"] = P_out
        outputs[f"{STREAM}:quality_out"] = q_out
        outputs[f"{STREAM}:phase_out"] = ph_out


# ===========================================================================
# Generic steam process: consumes supplied steam to make a product, loses a
# fraction of the steam (vent/leak), and returns lower-quality condensate.
# Used for HP Process, LP Process, and LP Utility with different configs.
# ===========================================================================
@define(kw_only=True)
class SteamProcessConfig(BaseConfig):
    """Configuration for a generic steam-consuming process.

    Attributes:
        loss_fraction: Fraction of inlet steam lost (vent/leak), not returned [-].
        product_yield: Product mass produced per unit steam consumed [kg/kg].
        outlet_pressure: Condensate outlet pressure [bar].
        outlet_temperature: Condensate outlet temperature [degC] (subcooled). Set this
            or ``outlet_quality``.
        outlet_quality: Condensate outlet quality [-] (saturated). Set this or
            ``outlet_temperature``.
    """

    loss_fraction: float = field(default=0.02, validator=_gte_zero)
    product_yield: float = field(default=1.0, validator=_gte_zero)
    outlet_pressure: float = field(validator=_gt_zero)
    outlet_temperature: float | None = field(default=None)
    outlet_quality: float | None = field(default=None)

    def __attrs_post_init__(self):
        if self.outlet_temperature is None and self.outlet_quality is None:
            raise ValueError("Set either 'outlet_temperature' or 'outlet_quality'.")


class SteamProcessPerformance(om.ExplicitComponent):
    """Steam consumer that produces a product and returns lower-quality condensate."""

    _time_step_bounds = (1, int(1e9))

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = SteamProcessConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.dt = int(self.options["plant_config"]["plant"]["simulation"]["dt"])

        add_multivariable_input(self, STREAM, self.n)
        for var_name, var_props in multivariable_streams[STREAM].items():
            self.add_output(
                f"{STREAM}:{var_name}_out", val=0.0, shape=self.n,
                units=var_props.get("units"), desc=var_props.get("desc", ""),
            )
        self.add_output("product", val=0.0, shape=self.n, units="kg/s", desc="Product output rate")
        self.add_output("steam_loss", val=0.0, shape=self.n, units="kg/s", desc="Vented steam")
        self.add_output("heat_delivered", val=0.0, shape=self.n, units="kW", desc="Heat to product")
        self.add_output("total_product", val=0.0, units="kg", desc="Product over the simulation")
        self.add_output("total_steam_loss", val=0.0, units="kg", desc="Steam loss over the run")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m_in = inputs[f"{STREAM}:mass_flow_in"]
        T_in = inputs[f"{STREAM}:temperature_in"]
        P_in = inputs[f"{STREAM}:pressure_in"]
        q_in = inputs[f"{STREAM}:quality_in"]
        ph_in = inputs[f"{STREAM}:phase_in"]
        h_in = h_from_stream(T_in, P_in, q_in, ph_in)

        m_loss = self.config.loss_fraction * m_in
        m_cond = m_in - m_loss

        # Condensate outlet state fixed by config (subcooled by temperature or saturated quality).
        P_out = np.full(self.n, self.config.outlet_pressure)
        if self.config.outlet_quality is not None:
            q_out = float(self.config.outlet_quality)
            T_out = Tsat_from_P(self.config.outlet_pressure) - _C2K
            ph_out = 1.0
            h_out = h_from_stream(T_out, self.config.outlet_pressure, q_out, ph_out)[0]
        else:
            T_out = float(self.config.outlet_temperature)
            ph_out = phase_from_TPq(T_out, self.config.outlet_pressure, 0.0)
            h_out = h_from_TP(T_out + _C2K, self.config.outlet_pressure)
            q_out = 0.0 if ph_out == 0.0 else 1.0

        outputs[f"{STREAM}:mass_flow_out"] = m_cond
        outputs[f"{STREAM}:temperature_out"] = np.full(self.n, T_out)
        outputs[f"{STREAM}:pressure_out"] = P_out
        outputs[f"{STREAM}:quality_out"] = np.full(self.n, q_out)
        outputs[f"{STREAM}:phase_out"] = np.full(self.n, ph_out)
        outputs["product"] = m_in * self.config.product_yield
        outputs["steam_loss"] = m_loss
        # Heat handed to the product = enthalpy given up by the condensing steam [kW].
        outputs["heat_delivered"] = m_cond * (h_in - h_out)
        outputs["total_product"] = float(np.sum(m_in * self.config.product_yield) * self.dt)
        outputs["total_steam_loss"] = float(np.sum(m_loss) * self.dt)


@define(kw_only=True)
class SteamProcessCostConfig(CostModelBaseConfig):
    """CapEx/OpEx cost model config for a steam process.

    Attributes:
        capex: Overnight capital expenditure [USD].
        opex: Fixed annual operating expenditure [USD/year].
    """

    capex: float = field(default=0.0, validator=_gte_zero)
    opex: float = field(default=0.0, validator=_gte_zero)


class SteamProcessCost(CostModelBaseClass):
    """Simple CapEx/OpEx cost model for a steam process unit."""

    _time_step_bounds = (1, int(1e9))

    def setup(self):
        self.config = SteamProcessCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )
        super().setup()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["CapEx"] = self.config.capex
        outputs["OpEx"] = self.config.opex
        if discrete_outputs is not None:
            discrete_outputs["cost_year"] = self.config.cost_year


# ===========================================================================
# Make-up water from losses: sizes the make-up water to replace ALL tracked
# water losses (boiler blowdowns, deaerator vent, header loss, process losses,
# LP-utility drain) so the plant water balance closes (make-up in = losses out).
# It also injects the make-up water into the returning condensate stream.
# ===========================================================================
@define(kw_only=True)
class MakeupWaterFromLossesConfig(BaseConfig):
    """Configuration for the loss-driven make-up water intake.

    Attributes:
        n_loss_inputs: Number of loss mass-flow inputs [kg/s] to sum.
        makeup_temperature: Temperature of the supplied make-up water [degC].
        makeup_pressure: Pressure of the make-up water and the mixed outlet [bar].
    """

    n_loss_inputs: int = field(validator=_gt_zero)
    makeup_temperature: float = field(default=15.0)
    makeup_pressure: float = field(default=1.05, validator=_gt_zero)


class MakeupWaterFromLosses(om.ExplicitComponent):
    """Make-up water = sum of tracked losses, mixed into the condensate stream."""

    _time_step_bounds = (1, int(1e9))

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = MakeupWaterFromLossesConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.dt = int(self.options["plant_config"]["plant"]["simulation"]["dt"])

        add_multivariable_input(self, STREAM, self.n)
        for var_name, var_props in multivariable_streams[STREAM].items():
            self.add_output(f"{STREAM}:{var_name}_out", val=0.0, shape=self.n,
                            units=var_props.get("units"), desc=var_props.get("desc", ""))
        for k in range(1, self.config.n_loss_inputs + 1):
            self.add_input(f"loss{k}", val=0.0, shape=self.n, units="kg/s",
                           desc=f"Tracked water loss stream {k}")
        # Make-up water supply (required by the feedstock auto-connection).
        self.add_input("water_in", val=0.0, shape=self.n, units="kg/h",
                       desc="Make-up water supplied by the connected water feedstock")
        self.add_output("water_consumed", val=0.0, shape=self.n, units="kg/h",
                        desc="Make-up water required to replace total losses")
        self.add_output("makeup_flow", val=0.0, shape=self.n, units="kg/s",
                        desc="Make-up water flow (= total losses)")
        self.add_output("total_losses", val=0.0, shape=self.n, units="kg/s",
                        desc="Total tracked water losses")
        self.add_output("total_water_consumed", val=0.0, units="kg", desc="Make-up over the run")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m_cond = inputs[f"{STREAM}:mass_flow_in"]
        T_cond = inputs[f"{STREAM}:temperature_in"]
        P_cond = inputs[f"{STREAM}:pressure_in"]
        q_cond = inputs[f"{STREAM}:quality_in"]
        ph_cond = inputs[f"{STREAM}:phase_in"]

        # Make-up replaces every tracked water loss leaving the plant.
        m_makeup = sum(inputs[f"loss{k}"] for k in range(1, self.config.n_loss_inputs + 1))

        h_cond = h_from_stream(T_cond, P_cond, q_cond, ph_cond)
        P_out = np.full(self.n, self.config.makeup_pressure)
        h_makeup = h_from_TP(self.config.makeup_temperature + _C2K, self.config.makeup_pressure)

        m_out = m_cond + m_makeup
        with np.errstate(divide="ignore", invalid="ignore"):
            h_out = np.where(m_out > 0, (m_cond * h_cond + m_makeup * h_makeup) / m_out, h_makeup)
        T_out_K, q_out, ph_out = state_from_hP(h_out, P_out)

        outputs[f"{STREAM}:mass_flow_out"] = m_out
        outputs[f"{STREAM}:temperature_out"] = T_out_K - _C2K
        outputs[f"{STREAM}:pressure_out"] = P_out
        outputs[f"{STREAM}:quality_out"] = q_out
        outputs[f"{STREAM}:phase_out"] = ph_out
        outputs["makeup_flow"] = m_makeup
        outputs["total_losses"] = m_makeup
        outputs["water_consumed"] = m_makeup * 3600.0
        outputs["total_water_consumed"] = float(np.sum(m_makeup) * self.dt)

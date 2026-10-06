"""
Thermodynamic models for the Steam Recovery Loop of an industrial process using
steam, built as H2Integrate custom technology models.

The loop is decomposed into connected converters plus boundary source/sink stubs,
all exchanging the shared multivariable ``steam`` stream (mass_flow [kg/s],
temperature [degC], pressure [bar], quality [-], phase [-]). Make-up water is supplied
by a ``water`` feedstock; its required rate is computed by ``makeup_water_intake`` from
the loop losses and exposed as a ``water_consumed`` demand.

Flow of the shared ``steam`` stream through the loop:

    hp_condensate_source, lp_condensate_source
        -> condensate_surge_tank_combiner
        -> makeup_water_intake   (<- make-up water from the ``water`` feedstock)
        -> cst_pump -> deaerator -> bfw_pump -> boiler_feedwater_sink

Water/steam properties come from CoolProp (IAPWS-IF97), a built-in H2Integrate
dependency. The shared ``steam`` stream carries (T, P, quality, phase) but not specific
enthalpy, so each component reconstructs enthalpy internally from that state for its
energy balances. The ``phase`` code (0 subcooled liquid, 1 saturated, 2 superheated
vapor) disambiguates the single-phase branches, which quality alone cannot. All
components are plain OpenMDAO ``ExplicitComponent`` models (the same pattern H2Integrate
uses for its ``gas_stream_combiner``), configured with ``attrs``/``BaseConfig`` and
paired with ``CostModelBaseClass`` cost models.
"""

import numpy as np
import openmdao.api as om
import CoolProp.CoolProp as CP
from attrs import field, define, validators

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import CostModelBaseClass, CostModelBaseConfig
from h2integrate.core.commodity_stream_definitions import (
    multivariable_streams,
    add_multivariable_input,
    add_multivariable_output,
)


STREAM = "steam"
_TS_BOUNDS = (1, int(1e9))  # (min, max) time-step lengths in seconds this model supports
_C2K = 273.15  # degC -> K offset (the shared "steam" stream reports temperature in degC)


# ---------------------------------------------------------------------------
# CoolProp (IAPWS-IF97) property helpers. Pressure in bar, enthalpy in kJ/kg,
# temperature in K. CoolProp works in Pa and J/kg internally.
# ---------------------------------------------------------------------------
def h_from_TP(T_K, P_bar):
    """Specific enthalpy [kJ/kg] of water from temperature [K] and pressure [bar]."""
    return CP.PropsSI("Hmass", "T", float(T_K), "P", float(P_bar) * 1e5, "Water") / 1e3


def hf_from_P(P_bar):
    """Saturated-liquid specific enthalpy [kJ/kg] at pressure [bar]."""
    return CP.PropsSI("Hmass", "P", float(P_bar) * 1e5, "Q", 0.0, "Water") / 1e3


def hg_from_P(P_bar):
    """Saturated-vapor specific enthalpy [kJ/kg] at pressure [bar]."""
    return CP.PropsSI("Hmass", "P", float(P_bar) * 1e5, "Q", 1.0, "Water") / 1e3


def Tsat_from_P(P_bar):
    """Saturation temperature [K] at pressure [bar]."""
    return CP.PropsSI("T", "P", float(P_bar) * 1e5, "Q", 0.0, "Water")


def rho_satliq_from_P(P_bar):
    """Saturated-liquid density [kg/m^3] at pressure [bar] (pump inlet is at saturation)."""
    return CP.PropsSI("Dmass", "P", float(P_bar) * 1e5, "Q", 0.0, "Water")


def _state_from_hP_scalar(h_kJkg, P_bar):
    """Return (temperature [K], vapor_quality [-], phase [-]) from enthalpy [kJ/kg], pressure [bar].

    ``phase`` is 0 (subcooled liquid), 1 (saturated), or 2 (superheated vapor).
    """
    P = float(P_bar) * 1e5
    h = float(h_kJkg) * 1e3
    hf = CP.PropsSI("Hmass", "P", P, "Q", 0.0, "Water")
    hg = CP.PropsSI("Hmass", "P", P, "Q", 1.0, "Water")
    tol = 1.0  # J/kg tolerance so numerically-saturated states classify as saturated
    if h < hf - tol:  # subcooled liquid
        return CP.PropsSI("T", "P", P, "Hmass", h, "Water"), 0.0, 0.0
    if h > hg + tol:  # superheated vapor
        return CP.PropsSI("T", "P", P, "Hmass", h, "Water"), 1.0, 2.0
    q = float(np.clip((h - hf) / (hg - hf), 0.0, 1.0))  # saturated (incl. sat. liquid/vapor)
    return CP.PropsSI("T", "P", P, "Q", q, "Water"), q, 1.0


def state_from_hP(h_kJkg, P_bar):
    """Vectorized (temperature [K], vapor_quality [-], phase [-]) from enthalpy and pressure arrays.

    ``phase`` is 0 (subcooled liquid), 1 (saturated), or 2 (superheated vapor).
    Short-circuits the common case where both inputs are constant over the horizon.
    """
    h = np.atleast_1d(np.asarray(h_kJkg, dtype=float))
    P = np.atleast_1d(np.asarray(P_bar, dtype=float))
    n = max(h.size, P.size)
    h = np.broadcast_to(h, (n,))
    P = np.broadcast_to(P, (n,))
    if np.ptp(h) == 0.0 and np.ptp(P) == 0.0:
        T0, q0, ph0 = _state_from_hP_scalar(h[0], P[0])
        return np.full(n, T0), np.full(n, q0), np.full(n, ph0)
    T = np.empty(n)
    q = np.empty(n)
    ph = np.empty(n)
    for i in range(n):
        T[i], q[i], ph[i] = _state_from_hP_scalar(h[i], P[i])
    return T, q, ph


def _rho_satliq_vec(P_bar):
    """Vectorized saturated-liquid density [kg/m^3] over a pressure array [bar]."""
    P = np.atleast_1d(np.asarray(P_bar, dtype=float))
    if np.ptp(P) == 0.0:
        return np.full(P.size, rho_satliq_from_P(P[0]))
    return np.array([rho_satliq_from_P(p) for p in P])


def _h_from_state_scalar(T_degC, P_bar, quality, phase):
    """Specific enthalpy [kJ/kg] from a single stream state (T [degC], P [bar], quality, phase).

    Saturated states (``phase == 1``) are fixed by (P, quality); subcooled (0) and
    superheated (2) states are fixed by (T, P). Enthalpy is reconstructed here because the
    shared ``steam`` stream carries (T, P, quality, phase) but not specific enthalpy.
    """
    if phase == 1.0:
        return hf_from_P(P_bar) + quality * (hg_from_P(P_bar) - hf_from_P(P_bar))
    T_K = T_degC + _C2K
    # Guard the (T, P) degeneracy at the saturation line (e.g. numerically-saturated vapor).
    if abs(T_K - Tsat_from_P(P_bar)) < 1e-2:
        return hg_from_P(P_bar) if phase >= 2.0 else hf_from_P(P_bar)
    return h_from_TP(T_K, P_bar)


def h_from_stream(T_degC, P_bar, quality, phase):
    """Vectorized specific enthalpy [kJ/kg] from stream state arrays (T [degC], P [bar], x, phase).

    Short-circuits the common case where the carried state is constant over the horizon.
    """
    T = np.atleast_1d(np.asarray(T_degC, dtype=float))
    P = np.atleast_1d(np.asarray(P_bar, dtype=float))
    x = np.atleast_1d(np.asarray(quality, dtype=float))
    ph = np.atleast_1d(np.asarray(phase, dtype=float))
    n = max(T.size, P.size, x.size, ph.size)
    T = np.broadcast_to(T, (n,))
    P = np.broadcast_to(P, (n,))
    x = np.broadcast_to(x, (n,))
    ph = np.broadcast_to(ph, (n,))
    if np.ptp(T) == 0.0 and np.ptp(P) == 0.0 and np.ptp(x) == 0.0 and np.ptp(ph) == 0.0:
        return np.full(n, _h_from_state_scalar(T[0], P[0], x[0], ph[0]))
    return np.array([_h_from_state_scalar(T[i], P[i], x[i], ph[i]) for i in range(n)])


def phase_from_TPq(T_degC, P_bar, quality):
    """Phase code (0 subcooled / 1 saturated / 2 superheated) from (T [degC], P [bar], quality).

    Saturated when 0 < quality < 1 or T is at T_sat(P); otherwise subcooled if T < T_sat,
    superheated if T > T_sat.
    """
    if 0.0 < quality < 1.0:
        return 1.0
    T_sat_C = Tsat_from_P(P_bar) - _C2K
    if T_degC < T_sat_C - 1e-3:
        return 0.0
    if T_degC > T_sat_C + 1e-3:
        return 2.0
    return 1.0


# ===========================================================================
# Boundary source: emits a steam stream from fixed conditions with an
# optional synthetic load profile on the mass flow. Real 8760 profiles can be
# swapped in later via the load_profile / min_mass_flow / max_mass_flow inputs.
# ===========================================================================
@define(kw_only=True)
class SteamWaterStreamSourceConfig(BaseConfig):
    """Configuration for a boundary steam/water stream source.

    Attributes:
        mass_flow: Base (mean) mass flow rate [kg/s].
        pressure: Stream pressure [bar].
        temperature: Stream temperature [degC]. Used when ``vapor_quality`` is not set.
        vapor_quality: If set (0-1), stream is at saturation at ``pressure`` with this quality.
        load_profile: Synthetic profile shape: 'constant', 'diurnal', or 'random'.
        min_mass_flow: Minimum mass flow [kg/s] for non-constant profiles.
        max_mass_flow: Maximum mass flow [kg/s] for non-constant profiles.
        random_seed: Seed for the 'random' profile (reproducibility).
    """

    mass_flow: float = field(validator=validators.gt(0))
    pressure: float = field(validator=validators.gt(0))
    temperature: float | None = field(default=None)
    vapor_quality: float | None = field(default=None)
    load_profile: str = field(default="constant")
    min_mass_flow: float | None = field(default=None)
    max_mass_flow: float | None = field(default=None)
    random_seed: int | None = field(default=None)


class SteamWaterStreamSource(om.ExplicitComponent):
    """Boundary condition producer for a ``steam`` stream."""

    _time_step_bounds = _TS_BOUNDS

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = SteamWaterStreamSourceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        add_multivariable_output(self, STREAM, self.n_timesteps)

    def _flow_profile(self):
        """Build the mass-flow time series [kg/s] for the configured profile shape.

        Semantics note: for the non-constant shapes the envelope is
        ``[min_mass_flow, max_mass_flow]``; ``mass_flow`` is only the fallback used when a
        bound is omitted. A 'diurnal' series therefore has mean
        ``(min_mass_flow + max_mass_flow) / 2`` (not necessarily ``mass_flow``). Replace
        this synthetic generator with the measured 8760 profile when it becomes available.
        """
        n = self.n_timesteps
        base = self.config.mass_flow                 # base/fallback mean flow [kg/s]
        shape = self.config.load_profile
        if shape == "constant" or n <= 1:
            return np.full(n, base)
        lo = self.config.min_mass_flow if self.config.min_mass_flow is not None else base
        hi = self.config.max_mass_flow if self.config.max_mass_flow is not None else base
        if shape == "diurnal":
            # Smooth 24 h cosine swing from lo (hour 0) to hi (hour 12):
            #     m(t) = lo + (hi - lo) * 0.5 * (1 - cos(2*pi*(t mod 24)/24))
            hours = np.arange(n)
            frac = 0.5 * (1.0 - np.cos(2.0 * np.pi * (hours % 24) / 24.0))
            return lo + (hi - lo) * frac
        if shape == "random":
            rng = np.random.default_rng(self.config.random_seed)
            return rng.uniform(lo, hi, n)
        raise ValueError(f"Unknown load_profile '{shape}' (use constant/diurnal/random).")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.n_timesteps
        flow = self._flow_profile()      # mass flow rate time series [kg/s]
        P = self.config.pressure         # stream pressure [bar]

        if self.config.vapor_quality is not None:
            # Saturated mixture set by quality; state fixed by (P, x):   T = T_sat(P).
            q = float(self.config.vapor_quality)                   # vapor mass fraction [-]
            T = Tsat_from_P(P) - _C2K                              # temperature [degC]
            ph = 1.0                                               # saturated
        elif self.config.temperature is not None:
            # Single-phase (sub-cooled/superheated) state fixed by (T, P).
            T = float(self.config.temperature)                     # temperature [degC]
            h = h_from_TP(T + _C2K, P)                             # specific enthalpy [kJ/kg]
            _, q, ph = _state_from_hP_scalar(h, P)                 # quality, phase [-]
        else:
            raise ValueError("Source requires either 'temperature' or 'vapor_quality'.")

        # Intensive properties are held constant; only the mass flow varies over time.
        outputs[f"{STREAM}:mass_flow_out"] = flow
        outputs[f"{STREAM}:temperature_out"] = np.full(n, T)
        outputs[f"{STREAM}:pressure_out"] = np.full(n, P)
        outputs[f"{STREAM}:quality_out"] = np.full(n, q)
        outputs[f"{STREAM}:phase_out"] = np.full(n, ph)


# ===========================================================================
# Condensate Surge Tank, modeled as a combiner (H2Integrate auto-indexes the
# numbered inlets because the technology name contains "combiner"). Sums mass
# flow, conserves energy via mass-weighted enthalpy, and flashes to the tank
# pressure with CoolProp to recover outlet temperature and vapor quality.
# ===========================================================================
@define(kw_only=True)
class CondensateSurgeTankConfig(BaseConfig):
    """Configuration for the condensate surge tank (mixing) model.

    Attributes:
        commodity: Multivariable stream name (must be 'steam').
        in_streams: Number of inlet streams to combine.
        tank_pressure: Operating pressure of the tank [bar].
        heat_loss_fraction: Fraction of mixed sensible enthalpy lost to ambient [-].
    """

    commodity: str = field(default=STREAM)
    in_streams: int = field(default=3, validator=validators.gt(0))
    tank_pressure: float = field(default=1.05, validator=validators.gt(0))
    heat_loss_fraction: float = field(default=0.0, validator=validators.ge(0))

    def __attrs_post_init__(self):
        if self.commodity not in multivariable_streams:
            raise ValueError(f"Unknown stream '{self.commodity}'.")


class CondensateSurgeTankCombinerPerformance(om.ExplicitComponent):
    """Adiabatic (optionally lossy) mixing of the returning condensate streams."""

    _time_step_bounds = _TS_BOUNDS

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = CondensateSurgeTankConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        # Multivariable stream this tank mixes (from config; default steam).
        stream = self.config.commodity
        stream_def = multivariable_streams[stream]

        # One numbered inlet port per return. H2Integrate wires sources to
        # ``<stream>:<var>_in1``, ``_in2`` ... automatically because the technology name
        # contains "combiner"; ``in_streams`` must equal the number of sources connected here.
        for i in range(1, self.config.in_streams + 1):
            for var_name, var_props in stream_def.items():
                self.add_input(
                    f"{stream}:{var_name}_in{i}",
                    val=0.0,
                    shape=self.n_timesteps,
                    units=var_props.get("units"),
                    desc=f"Inlet {i}: {var_props.get('desc', '')}",
                )
        add_multivariable_output(self, stream, self.n_timesteps)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.n_timesteps
        ns = self.config.in_streams          # number of inlet ports
        stream = self.config.commodity

        # Per-inlet mass flow [kg/s]; specific enthalpy [kJ/kg] reconstructed from the carried
        # state (T [degC], P [bar], quality, phase) because the "steam" stream omits enthalpy.
        flows = [inputs[f"{stream}:mass_flow_in{i}"] for i in range(1, ns + 1)]
        enthalpies = [
            h_from_stream(
                inputs[f"{stream}:temperature_in{i}"],
                inputs[f"{stream}:pressure_in{i}"],
                inputs[f"{stream}:quality_in{i}"],
                inputs[f"{stream}:phase_in{i}"],
            )
            for i in range(1, ns + 1)
        ]

        # Mass balance:   m_out = sum_i m_i
        total_flow = sum(flows)

        # Energy balance for adiabatic mixing. The mixed specific enthalpy is the
        # mass-weighted average, which exactly conserves total enthalpy:
        #     h_mix = sum_i (m_i * h_i) / sum_i m_i
        weighted_h = sum(flows[i] * enthalpies[i] for i in range(ns))
        with np.errstate(divide="ignore", invalid="ignore"):
            h_mix = np.where(total_flow > 0, weighted_h / total_flow, 0.0)

        # Optional ambient loss removes a fraction of the mixed enthalpy (datum ~ IAPWS
        # triple point, so this is approximately a fractional sensible-heat loss):
        #     h_mix <- h_mix * (1 - f_loss)
        h_mix = h_mix * (1.0 - self.config.heat_loss_fraction)

        # Flash the mixed liquid to the tank pressure and recover the outlet temperature
        # and vapor quality from (h_mix, P_tank) with the IAPWS-IF97 equation of state.
        P_out = np.full(n, self.config.tank_pressure)   # tank operating pressure [bar]
        T_out_K, q_out, ph_out = state_from_hP(h_mix, P_out)  # outlet T [K], quality, phase

        outputs[f"{stream}:mass_flow_out"] = total_flow
        outputs[f"{stream}:temperature_out"] = T_out_K - _C2K
        outputs[f"{stream}:pressure_out"] = P_out
        outputs[f"{stream}:quality_out"] = q_out
        outputs[f"{stream}:phase_out"] = ph_out


# ===========================================================================
# Make-up water intake: tops up the returning condensate with make-up water to
# replace loop losses (deaerator vent + boiler blowdown). The make-up requirement
# is computed feed-forward from the incoming condensate flow and reported as a
# ``water_consumed`` demand that a connected ``water`` feedstock supplies and costs.
# ===========================================================================
@define(kw_only=True)
class MakeupWaterIntakeConfig(BaseConfig):
    """Configuration for the make-up water intake.

    Attributes:
        vent_fraction: Deaerator vent loss as a fraction of feedwater flow [-].
        blowdown_fraction: Boiler blowdown loss as a fraction of feedwater flow [-].
        makeup_temperature: Temperature of the supplied make-up water [degC].
        makeup_pressure: Pressure of the make-up water and the mixed outlet [bar].
    """

    vent_fraction: float = field(default=0.001, validator=validators.ge(0))
    blowdown_fraction: float = field(default=0.01, validator=validators.ge(0))
    makeup_temperature: float = field(default=15.0)
    makeup_pressure: float = field(default=1.05, validator=validators.gt(0))

    def __attrs_post_init__(self):
        total_loss = self.vent_fraction + self.blowdown_fraction
        if not 0.0 <= total_loss < 1.0:
            raise ValueError(
                "vent_fraction + blowdown_fraction must be in [0, 1); "
                f"got {total_loss}."
            )


class MakeupWaterIntake(om.ExplicitComponent):
    """Adds loss-replacement make-up water to the returning condensate stream.

    Make-up replaces the water leaving the loop each pass. With total loss fraction
    ``f = vent_fraction + blowdown_fraction`` of the feedwater (where
    ``feedwater = condensate + make-up``), closing the mass balance gives the
    feed-forward make-up rate ``m_makeup = f / (1 - f) * m_condensate``. The result is
    exposed as ``water_consumed`` (kg/h) for a connected ``water`` feedstock.
    """

    _time_step_bounds = _TS_BOUNDS

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = MakeupWaterIntakeConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.dt = int(self.options["plant_config"]["plant"]["simulation"]["dt"])

        add_multivariable_input(self, STREAM, self.n_timesteps)
        add_multivariable_output(self, STREAM, self.n_timesteps)

        # Make-up water supplied by the feedstock (required by the feedstock auto-connection).
        self.add_input(
            "water_in",
            val=0.0,
            shape=self.n_timesteps,
            units="kg/h",
            desc="Make-up water supplied by the connected water feedstock",
        )
        # Demand signal read by the water feedstock cost model.
        self.add_output(
            "water_consumed",
            val=0.0,
            shape=self.n_timesteps,
            units="kg/h",
            desc="Make-up water required to replace loop losses",
        )
        self.add_output(
            "total_water_consumed", val=0.0, units="kg", desc="Make-up water over the simulation"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m_cond = inputs[f"{STREAM}:mass_flow_in"]        # returning condensate flow [kg/s]
        T_cond = inputs[f"{STREAM}:temperature_in"]      # condensate temperature [degC]
        P_cond = inputs[f"{STREAM}:pressure_in"]         # condensate pressure [bar]
        q_cond = inputs[f"{STREAM}:quality_in"]          # condensate vapor quality [-]
        ph_cond = inputs[f"{STREAM}:phase_in"]           # condensate phase [-]

        # Feed-forward loss replacement:  m_makeup = f/(1-f) * m_condensate  [kg/s].
        total_loss = self.config.vent_fraction + self.config.blowdown_fraction
        m_makeup = (total_loss / (1.0 - total_loss)) * m_cond

        # Enthalpies [kJ/kg]: condensate reconstructed from its state; make-up as sub-cooled liquid.
        h_cond = h_from_stream(T_cond, P_cond, q_cond, ph_cond)
        P_out = np.full(self.n_timesteps, self.config.makeup_pressure)
        h_makeup = h_from_TP(self.config.makeup_temperature + _C2K, self.config.makeup_pressure)

        # Adiabatic mix of condensate + make-up, then flash to the intake pressure.
        m_out = m_cond + m_makeup
        with np.errstate(divide="ignore", invalid="ignore"):
            h_out = np.where(m_out > 0, (m_cond * h_cond + m_makeup * h_makeup) / m_out, h_makeup)
        T_out_K, q_out, ph_out = state_from_hP(h_out, P_out)

        outputs[f"{STREAM}:mass_flow_out"] = m_out
        outputs[f"{STREAM}:temperature_out"] = T_out_K - _C2K
        outputs[f"{STREAM}:pressure_out"] = P_out
        outputs[f"{STREAM}:quality_out"] = q_out
        outputs[f"{STREAM}:phase_out"] = ph_out
        # kg/s -> kg/h for the water feedstock, which operates in kg/h.
        outputs["water_consumed"] = m_makeup * 3600.0
        outputs["total_water_consumed"] = float(np.sum(m_makeup) * self.dt)


# ===========================================================================
# Water pump (reused for the CST pump and the BFW pump). Raises the stream to
# a discharge pressure, computing hydraulic work, the resulting enthalpy rise,
# and the electrical power demand.
# ===========================================================================
@define(kw_only=True)
class WaterPumpConfig(BaseConfig):
    """Configuration for a liquid pump.

    Attributes:
        discharge_pressure: Pump discharge pressure [bar].
        pump_efficiency: Hydraulic (isentropic) efficiency [-].
        motor_efficiency: Electric motor efficiency [-].
    """

    discharge_pressure: float = field(validator=validators.gt(0))
    pump_efficiency: float = field(default=0.7, validator=validators.gt(0))
    motor_efficiency: float = field(default=0.9, validator=validators.gt(0))


class WaterPumpPerformance(om.ExplicitComponent):
    """Incompressible-liquid pump with electrical demand output."""

    _time_step_bounds = _TS_BOUNDS

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = WaterPumpConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.dt = int(self.options["plant_config"]["plant"]["simulation"]["dt"])

        add_multivariable_input(self, STREAM, self.n_timesteps)
        add_multivariable_output(self, STREAM, self.n_timesteps)
        self.add_output(
            "electricity_demand",
            val=0.0,
            shape=self.n_timesteps,
            units="kW",
            desc="Electrical power demand of the pump motor",
        )
        self.add_output(
            "total_electricity", val=0.0, units="kW*h", desc="Pump electricity over the simulation"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m = inputs[f"{STREAM}:mass_flow_in"]              # inlet mass flow rate [kg/s]
        P_in = inputs[f"{STREAM}:pressure_in"]            # inlet (suction) pressure [bar]
        T_in = inputs[f"{STREAM}:temperature_in"]         # inlet temperature [degC]
        q_in = inputs[f"{STREAM}:quality_in"]             # inlet vapor quality [-]
        ph_in = inputs[f"{STREAM}:phase_in"]              # inlet phase [-]
        P_dis = self.config.discharge_pressure            # discharge (set) pressure [bar]

        # Pumps handle liquid only: reject vapor (phase 2) or any two-phase vapor content.
        if np.any(ph_in >= 2.0) or np.any(q_in > 1e-3):
            raise ValueError(
                f"{self.__class__.__name__} received vapor at the inlet "
                "(phase 2 or quality > 0); only liquid may be pumped."
            )

        # Inlet enthalpy reconstructed from the carried state (T, P, quality, phase) [kJ/kg].
        h_in = h_from_stream(T_in, P_in, q_in, ph_in)

        # Liquid density approximated by the saturated-liquid value at the inlet pressure:
        # exact for the deaerator's saturated outlet, ~1% high for a sub-cooled inlet
        # (negligible effect on the resulting pump work).
        rho = _rho_satliq_vec(P_in)                       # liquid density [kg/m^3]
        v = 1.0 / rho                                     # specific volume [m^3/kg]

        # Incompressible-liquid pump work per unit mass:
        #     w_ideal  = v * dP            (reversible)         [J/kg]
        #     w_actual = w_ideal / eta_p   (hydraulic losses)   [J/kg]
        dP = (P_dis - P_in) * 1e5                          # pressure rise [Pa]
        w_ideal = v * dP
        w_actual = w_ideal / self.config.pump_efficiency

        # Steady-flow energy balance for a near-incompressible liquid: the enthalpy rise
        # equals the actual specific work,   h_out = h_in + w_actual   [kJ/kg].
        h_out = h_in + w_actual / 1e3

        # Discharge state (temperature, quality, phase) from (h_out, P_dis) via IAPWS-IF97.
        P_out = np.full(self.n_timesteps, P_dis)
        T_out_K, q_out, ph_out = state_from_hP(h_out, P_out)

        # Electrical power drawn by the motor,  P_el = m_dot * w_actual / eta_motor  [kW]
        # (m in kg/s already; /1e3: W -> kW).
        p_el_kw = m * w_actual / self.config.motor_efficiency / 1e3

        outputs[f"{STREAM}:mass_flow_out"] = m
        outputs[f"{STREAM}:temperature_out"] = T_out_K - _C2K
        outputs[f"{STREAM}:pressure_out"] = P_out
        outputs[f"{STREAM}:quality_out"] = q_out
        outputs[f"{STREAM}:phase_out"] = ph_out
        outputs["electricity_demand"] = p_el_kw
        outputs["total_electricity"] = float(np.sum(p_el_kw) * (self.dt / 3600.0))


# ===========================================================================
# Deaerator: direct-contact feedwater heater. Solves the deaeration steam draw
# needed to bring the feedwater to saturation at the deaerator pressure, with a
# small continuous vent of saturated vapor. Steam is exposed as a demand draw
# from the (out-of-boundary) header at the configured supply conditions.
# ===========================================================================
@define(kw_only=True)
class DeaeratorConfig(BaseConfig):
    """Configuration for the deaerator model.

    Attributes:
        da_pressure: Deaerator operating pressure [bar].
        steam_supply_pressure: Pressure of the deaeration steam header [bar].
        steam_supply_quality: Quality of the supplied deaeration steam [-].
        vent_rate_fraction: Continuous vent as a fraction of feedwater mass flow [-].
    """

    da_pressure: float = field(default=1.36, validator=validators.gt(0))
    steam_supply_pressure: float = field(default=11.36, validator=validators.gt(0))
    steam_supply_quality: float = field(default=1.0, validator=validators.ge(0))
    vent_rate_fraction: float = field(default=0.001, validator=validators.ge(0))


class DeaeratorPerformance(om.ExplicitComponent):
    """Deaerator solving the deaeration steam demand to reach saturation."""

    _time_step_bounds = _TS_BOUNDS

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = DeaeratorConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.dt = int(self.options["plant_config"]["plant"]["simulation"]["dt"])

        add_multivariable_input(self, STREAM, self.n_timesteps)
        add_multivariable_output(self, STREAM, self.n_timesteps)
        self.add_output(
            "steam_demand",
            val=0.0,
            shape=self.n_timesteps,
            units="kg/s",
            desc="Deaeration steam draw from the header",
        )
        self.add_output(
            "vent_flow", val=0.0, shape=self.n_timesteps, units="kg/s", desc="Vent (loss) flow"
        )
        self.add_output(
            "heat_duty",
            val=0.0,
            shape=self.n_timesteps,
            units="kW",
            desc="Thermal duty delivered by the deaeration steam",
        )
        self.add_output(
            "total_steam_demand", val=0.0, units="kg", desc="Deaeration steam over the simulation"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m_w = inputs[f"{STREAM}:mass_flow_in"]          # incoming feedwater mass flow [kg/s]
        T_w = inputs[f"{STREAM}:temperature_in"]        # incoming feedwater temperature [degC]
        P_w = inputs[f"{STREAM}:pressure_in"]           # incoming feedwater pressure [bar]
        q_w = inputs[f"{STREAM}:quality_in"]            # incoming feedwater vapor quality [-]
        ph_w = inputs[f"{STREAM}:phase_in"]             # incoming feedwater phase [-]
        # Incoming feedwater enthalpy reconstructed from the carried state [kJ/kg].
        h_w = h_from_stream(T_w, P_w, q_w, ph_w)

        # Deaerated boiler feedwater leaves as saturated liquid at the deaerator pressure.
        P_da = self.config.da_pressure   # deaerator operating pressure [bar]
        h_bfw = hf_from_P(P_da)          # BFW (saturated-liquid) enthalpy  h_f(P_da) [kJ/kg]
        T_bfw = Tsat_from_P(P_da) - _C2K # BFW saturation temperature       T_sat(P_da) [degC]
        h_vent = hg_from_P(P_da)         # vented (saturated-vapor) enthalpy h_g(P_da) [kJ/kg]

        # Deaeration steam enthalpy from the header, set by its quality (lever rule):
        #     h_s = h_f(P_s) + x_s * (h_g(P_s) - h_f(P_s))
        P_s = self.config.steam_supply_pressure   # steam header pressure [bar]
        h_s = hf_from_P(P_s) + self.config.steam_supply_quality * (hg_from_P(P_s) - hf_from_P(P_s))

        # Continuous vent (non-condensable purge) as a fraction of the feedwater flow [kg/s].
        m_vent = self.config.vent_rate_fraction * m_w

        # Steady mass and energy balances about the deaerator:
        #     mass:    m_bfw = m_w + m_steam - m_vent
        #     energy:  m_w*h_w + m_steam*h_s = m_bfw*h_bfw + m_vent*h_vent
        # Eliminating m_bfw and solving for the steam draw (requires h_s > h_bfw):
        #     m_steam = [ m_w*(h_bfw - h_w) + m_vent*(h_vent - h_bfw) ] / (h_s - h_bfw)
        m_steam = (m_w * (h_bfw - h_w) + m_vent * (h_vent - h_bfw)) / (h_s - h_bfw)
        m_bfw = m_w + m_steam - m_vent   # deaerated BFW delivered downstream [kg/s]

        n = self.n_timesteps
        outputs[f"{STREAM}:mass_flow_out"] = m_bfw
        outputs[f"{STREAM}:temperature_out"] = np.full(n, T_bfw)
        outputs[f"{STREAM}:pressure_out"] = np.full(n, P_da)
        outputs[f"{STREAM}:quality_out"] = np.zeros(n)
        outputs[f"{STREAM}:phase_out"] = np.ones(n)   # deaerated BFW leaves as saturated liquid
        outputs["steam_demand"] = m_steam
        outputs["vent_flow"] = m_vent
        # Thermal duty supplied by the steam, referenced to the BFW enthalpy:
        #     Q = m_steam * (h_s - h_bfw)      (kg/s * kJ/kg -> kW)
        outputs["heat_duty"] = m_steam * (h_s - h_bfw)
        outputs["total_steam_demand"] = float(np.sum(m_steam) * self.dt)


# ===========================================================================
# Boundary sink: consumes the final boiler feedwater stream and reports totals.
# ===========================================================================
class SteamWaterStreamSink(om.ExplicitComponent):
    """Consumes a ``steam`` stream and reports aggregate boundary KPIs."""

    _time_step_bounds = _TS_BOUNDS

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.dt = int(self.options["plant_config"]["plant"]["simulation"]["dt"])
        add_multivariable_input(self, STREAM, self.n_timesteps)

        self.add_output("total_mass_received", val=0.0, units="kg", desc="Total mass over the run")
        self.add_output("mean_mass_flow", val=0.0, units="kg/s", desc="Mean mass flow rate")
        self.add_output(
            "mean_temperature", val=0.0, units="degC", desc="Mass-weighted mean temperature"
        )
        self.add_output("mean_pressure", val=0.0, units="bar", desc="Mass-weighted mean pressure")
        self.add_output(
            "total_thermal_energy", val=0.0, units="kW*h", desc="Total stream enthalpy over the run"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m = inputs[f"{STREAM}:mass_flow_in"]            # boiler feedwater mass flow [kg/s]
        T = inputs[f"{STREAM}:temperature_in"]          # feedwater temperature [degC]
        P = inputs[f"{STREAM}:pressure_in"]             # feedwater pressure [bar]
        q = inputs[f"{STREAM}:quality_in"]              # feedwater vapor quality [-]
        ph = inputs[f"{STREAM}:phase_in"]               # feedwater phase [-]
        h = h_from_stream(T, P, q, ph)                  # feedwater specific enthalpy [kJ/kg]

        # Mass delivered each timestep and its total over the horizon:
        #     m_step = m_dot * dt ,   M_total = sum(m_step)   [kg]  (m in kg/s, dt in s)
        mass_per_step = m * self.dt
        total_mass = float(np.sum(mass_per_step))
        weight = total_mass if total_mass > 0 else 1.0   # guard against divide-by-zero

        outputs["total_mass_received"] = total_mass
        outputs["mean_mass_flow"] = float(np.mean(m))
        # Mass-weighted mean of an intensive property phi:  mean = sum(phi*m_step)/sum(m_step)
        outputs["mean_temperature"] = float(np.sum(T * mass_per_step) / weight)
        outputs["mean_pressure"] = float(np.sum(P * mass_per_step) / weight)
        # Total enthalpy carried by the stream:  E = sum(m_step * h)  (kg*kJ/kg=kJ; /3600 -> kWh)
        outputs["total_thermal_energy"] = float(np.sum(mass_per_step * h) / 3600.0)


# ===========================================================================
# Cost model shared by the four loop units. CapEx/OpEx are provided directly;
# swap in a sizing-based estimate later if desired.
# ===========================================================================
@define(kw_only=True)
class SteamRecoveryCostConfig(CostModelBaseConfig):
    """Configuration for a steam-recovery unit cost model.

    Attributes:
        capex: Overnight capital expenditure [USD].
        opex: Fixed annual operating expenditure [USD/year].
    """

    capex: float = field(default=0.0, validator=validators.ge(0))
    opex: float = field(default=0.0, validator=validators.ge(0))


class SteamRecoveryCostModel(CostModelBaseClass):
    """Simple CapEx/OpEx cost model for a steam-recovery-loop unit."""

    _time_step_bounds = _TS_BOUNDS

    def setup(self):
        self.config = SteamRecoveryCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )
        super().setup()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["CapEx"] = self.config.capex
        outputs["OpEx"] = self.config.opex
        if discrete_outputs is not None:
            discrete_outputs["cost_year"] = self.config.cost_year

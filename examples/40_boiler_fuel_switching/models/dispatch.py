"""
Example-local per-hour cost-merit boiler dispatcher (example 40).

With two fixed-size boilers and no storage, each hour is independent, so the
least-cost dispatch is a simple per-hour merit order: serve demand from whichever
boiler has the lower marginal cost of steam first (up to its rated output), then
spill the remainder to the other. This is not a heuristic - for the no-storage case
it is the exact cost optimum. (Add storage later and this becomes a MILP.)

Marginal cost of steam is compared on a $/kWh-thermal basis:
    NG boiler   : ng_price [$/MMBtu] / 293.071 [kWh/MMBtu] / ng_thermal_efficiency  (scalar)
    electric    : lmp [$/kWh] / elec_efficiency                                     (8760 array)
Both boilers feed the same header from the same feedwater, so the $/kWh-thermal order
is identical to the $/kg-steam order.
"""

import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs


_KWH_PER_MMBTU = 293.071  # 1 MMBtu = 293.071 kWh


def _gt_zero(instance, attribute, value):
    if value <= 0:
        raise ValueError(f"{attribute.name} must be > 0, got {value}.")


@define(kw_only=True)
class CostMeritBoilerDispatcherConfig(BaseConfig):
    """Configuration for the per-hour cost-merit boiler dispatcher.

    Attributes:
        ng_price: Natural gas price [USD/MMBtu].
        ng_thermal_efficiency: NG boiler thermal efficiency [-].
        elec_efficiency: Electric boiler electric-to-thermal efficiency [-].
        ng_rated_steam_kg_s: NG boiler rated steam output [kg/s].
        elec_rated_steam_kg_s: Electric boiler rated steam output [kg/s].
        electricity_price: Hourly electricity price [USD/kWh] (scalar or length n_timesteps).
    """

    ng_price: float = field(validator=_gt_zero)
    ng_thermal_efficiency: float = field(validator=_gt_zero)
    elec_efficiency: float = field(validator=_gt_zero)
    ng_rated_steam_kg_s: float = field(validator=_gt_zero)
    elec_rated_steam_kg_s: float = field(validator=_gt_zero)
    electricity_price: float | list = field()


class CostMeritBoilerDispatcher(om.ExplicitComponent):
    """Splits total steam demand between an NG and an electric boiler by hourly marginal cost."""

    _time_step_bounds = (1, int(1e9))

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = CostMeritBoilerDispatcherConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        self.n = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])

        c = self.config
        # NG marginal cost is constant; electric tracks the hourly LMP.
        self._ng_marg = c.ng_price / _KWH_PER_MMBTU / c.ng_thermal_efficiency  # $/kWh_th
        lmp = np.asarray(c.electricity_price, dtype=float)
        if lmp.ndim == 0:
            lmp = np.full(self.n, float(lmp))
        elif lmp.shape[0] != self.n:
            raise ValueError(
                f"electricity_price length ({lmp.shape[0]}) must be 1 or n_timesteps ({self.n})."
            )
        self._elec_marg = lmp / c.elec_efficiency  # $/kWh_th

        self.add_input("total_steam_demand", val=0.0, shape=self.n, units="kg/s")
        self.add_output("steam_demand_ng", val=0.0, shape=self.n, units="kg/s")
        self.add_output("steam_demand_elec", val=0.0, shape=self.n, units="kg/s")
        self.add_output("unmet_steam", val=0.0, shape=self.n, units="kg/s",
                        desc="Demand neither boiler could meet (both at rated)")
        self.add_output("electric_is_cheaper", val=0.0, shape=self.n, units="unitless",
                        desc="1 when the electric boiler has the lower marginal cost")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        c = self.config
        total = np.clip(inputs["total_steam_demand"], 0.0, None)
        ng_rated = c.ng_rated_steam_kg_s
        elec_rated = c.elec_rated_steam_kg_s

        elec_cheaper = self._elec_marg <= self._ng_marg

        # Electric-first branch.
        e_first = np.minimum(total, elec_rated)
        ng_after_e = np.minimum(np.maximum(total - e_first, 0.0), ng_rated)
        # NG-first branch.
        ng_first = np.minimum(total, ng_rated)
        e_after_ng = np.minimum(np.maximum(total - ng_first, 0.0), elec_rated)

        steam_elec = np.where(elec_cheaper, e_first, e_after_ng)
        steam_ng = np.where(elec_cheaper, ng_after_e, ng_first)

        outputs["steam_demand_ng"] = steam_ng
        outputs["steam_demand_elec"] = steam_elec
        outputs["unmet_steam"] = np.maximum(total - steam_ng - steam_elec, 0.0)
        outputs["electric_is_cheaper"] = elec_cheaper.astype(float)

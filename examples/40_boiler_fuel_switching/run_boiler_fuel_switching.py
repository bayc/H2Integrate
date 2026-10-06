"""
Example 40: Boiler fuel switching (NG vs electric).

One natural-gas firetube boiler and one electric boiler meet a shared steam load. A
per-hour cost-merit dispatcher routes each hour's demand to the cheaper source first
(NG at a fixed gas price, electric at a synthetic hourly LMP), then spills the remainder
to the other. With fixed sizes and no storage, this is the exact least-cost dispatch.
The steam header, PRVs, processes and full recovery loop are reused from example 39.

Run from any working directory with the environment that has H2Integrate installed:
    python run_boiler_fuel_switching.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from h2integrate import H2IntegrateModel
from h2integrate.converters.heat.steam_recovery_models import h_from_TP, hg_from_P


_HERE = Path(__file__).parent
model = H2IntegrateModel(str(_HERE / "40_boiler_fuel_switching.yaml"))
model.setup()
model.run()

gv = model.prob.get_val


def mean_kg_s(name):
    return float(np.mean(gv(name)))


# --- Steam production / demand ---
total_demand = gv("steam_load.total_steam_demand", units="kg/s")
ng = gv("boiler_ng.steam:mass_flow_out", units="kg/s")
el = gv("boiler_electric.steam:mass_flow_out", units="kg/s")
unmet = gv("cost_merit_dispatcher.unmet_steam", units="kg/s")
print("\nSteam production (mean kg/s):")
print(f"  total process+header demand : {total_demand.mean():.3f}")
print(f"  NG boiler                   : {ng.mean():.3f}")
print(f"  electric boiler             : {el.mean():.3f}")
print(f"  boilers total               : {ng.mean() + el.mean():.3f}")
print(f"  unmet (both at rated)       : {unmet.mean():.4f}")

# --- Hourly dispatch / switching ---
elec_cheaper = gv("cost_merit_dispatcher.electric_is_cheaper") > 0.5
lmp = np.asarray(gv("electricity_feedstock.price"))
ng_steam_kg = ng.sum()
el_steam_kg = el.sum()
share_el = el_steam_kg / max(ng_steam_kg + el_steam_kg, 1e-9)
print("\nHourly dispatch (fuel switching):")
print(f"  hours electric is cheaper   : {int(elec_cheaper.sum()):,} / {elec_cheaper.size}")
print(f"  hours NG is cheaper         : {int((~elec_cheaper).sum()):,}")
print(f"  annual steam share electric : {100.0 * share_el:.1f} %")
print(f"  annual steam share NG       : {100.0 * (1.0 - share_el):.1f} %")
print(f"  mean LMP when electric leads: {lmp[elec_cheaper].mean():.4f} USD/kWh")
print(f"  mean LMP when NG leads      : {lmp[~elec_cheaper].mean():.4f} USD/kWh")

# --- Energy and variable cost ---
ng_mmbtu = gv("boiler_ng.total_natural_gas", units="MMBtu")[0]
el_kwh = gv("boiler_electric.total_electricity", units="kW*h")[0]
ng_cost = gv("natural_gas_feedstock.VarOpEx", units="USD/year")[0]
el_cost = gv("electricity_feedstock.VarOpEx", units="USD/year")[0]
water_cost = gv("water_feedstock.VarOpEx", units="USD/year")[0]
print("\nEnergy and variable cost (annual):")
print(f"  natural gas   : {ng_mmbtu:,.0f} MMBtu/yr   cost {ng_cost:,.0f} USD/yr")
print(f"  electricity   : {el_kwh / 1e3:,.0f} MWh/yr     cost {el_cost:,.0f} USD/yr")
print(f"  make-up water :                    cost {water_cost:,.0f} USD/yr")
print(f"  => total variable cost      : {ng_cost + el_cost + water_cost:,.0f} USD/yr")

# --- Capital / fixed O&M ---
print("\nBoiler capital and fixed O&M:")
for b in ("boiler_ng", "boiler_electric"):
    capex = gv(f"{b}.CapEx", units="USD")[0]
    opex = gv(f"{b}.OpEx", units="USD/year")[0]
    print(f"  {b:15s} : CapEx {capex:,.0f} USD   OpEx {opex:,.0f} USD/yr")

# --- Emissions ---
ng_co2 = gv("boiler_ng.total_co2", units="kg")[0]
el_co2 = gv("boiler_electric.total_co2", units="kg")[0]
print("\nCO2 (annual):")
print(f"  NG boiler (direct)       : {ng_co2 / 1000.0:,.0f} tonne/yr")
print(f"  electric boiler (grid)   : {el_co2 / 1000.0:,.0f} tonne/yr")

# --- Water balance: make-up vs tracked losses ---
losses = {
    "NG boiler blowdown": mean_kg_s("boiler_ng.blowdown"),
    "electric boiler blowdown": mean_kg_s("boiler_electric.blowdown"),
    "hp_process loss": mean_kg_s("hp_process.steam_loss"),
    "lp_process loss": mean_kg_s("lp_process.steam_loss"),
    "lp_utility loss": mean_kg_s("lp_utility.steam_loss"),
    "lp_utility drain": mean_kg_s("lp_utility.steam:mass_flow_out"),
    "header loss": mean_kg_s("steam_header_splitter.header_loss_flow"),
}
makeup = mean_kg_s("makeup_water_intake.makeup_flow")
print("\nWater balance (mean kg/s):")
for k, v in losses.items():
    print(f"  loss: {k:26s} {v:.4f}")
print(f"  => total tracked losses  {sum(losses.values()):.4f}")
print(f"     make-up water in      {makeup:.4f}")

# --- Boiler feedwater loop check ---
bfw = gv("boiler_feedwater_sink.mean_mass_flow", units="kg/s")[0]
cond = mean_kg_s("condensate_surge_tank_combiner.steam:mass_flow_out")
da_steam = mean_kg_s("deaerator.steam_demand")
vent = mean_kg_s("deaerator.vent_flow")
print("\nRecovery loop (mean kg/s):")
print(f"  condensate returned : {cond:.3f}")
print(f"  make-up added       : {makeup:.3f}")
print(f"  deaeration steam  + : {da_steam:.3f}")
print(f"  deaerator vent    - : {vent:.4f}")
print(f"  boiler feedwater    : {bfw:.3f}")

# --- Plot: annual thermal load split (left axis) and energy prices (right axis) ---
perf = model.technology_config["technologies"]["boiler_ng"]["model_inputs"][
    "performance_parameters"
]
# Steam enthalpy rise [kJ/kg] converts steam mass flow [kg/s] to thermal power [MW].
dh = hg_from_P(perf["steam_pressure_bar"]) - h_from_TP(
    perf["feedwater_temp_c"] + 273.15, perf["steam_pressure_bar"]
)
load_total = total_demand * dh / 1e3
load_ng = ng * dh / 1e3
load_el = el * dh / 1e3

# Marginal cost of delivered steam [USD/kWh_th], exactly as the dispatcher compares them:
# NG = gas price / (kWh per MMBtu) / thermal efficiency; electric = LMP / heater efficiency.
disp = model.technology_config["technologies"]["cost_merit_dispatcher"]["model_inputs"][
    "performance_parameters"
]
ng_marg = disp["ng_price"] / 293.071 / disp["ng_thermal_efficiency"]  # constant [USD/kWh_th]
elec_marg = lmp / disp["elec_efficiency"]                             # array [USD/kWh_th]

dt_s = int(model.plant_config["plant"]["simulation"]["dt"])
t = pd.date_range("2024-01-01", periods=total_demand.size, freq=f"{dt_s}s")

# Aggregate to daily for a readable annual view; keep the daily electricity-price
# range so the intraday dips below the NG price (which drive switching) stay visible.
df = pd.DataFrame(
    {"total": load_total, "ng": load_ng, "el": load_el, "cost": elec_marg}, index=t
)
daily = df.resample("D").mean()
cost_lo = df["cost"].resample("D").min()
cost_hi = df["cost"].resample("D").max()

fig, ax = plt.subplots(figsize=(14, 6))
ax.stackplot(daily.index, daily["el"], daily["ng"],
             labels=["Met by electric boiler", "Met by NG boiler"],
             colors=["#4C72B0", "#DD8452"], alpha=0.85)
ax.plot(daily.index, daily["total"], color="black", lw=1.0, label="Thermal load")
ax.set_xlabel("Time")
ax.set_ylabel("Thermal load [MW]  (daily mean)")
ax.set_ylim(bottom=0)
ax.margins(x=0)

ax2 = ax.twinx()
ax2.fill_between(daily.index, cost_lo, cost_hi, color="#2E7D32", alpha=0.15,
                 label="Electric marginal cost (daily range)")
ax2.plot(daily.index, daily["cost"], color="#2E7D32", lw=1.0, label="Electric marginal cost")
ax2.axhline(ng_marg, color="#C62828", lw=1.4, ls="--", label="NG marginal cost")
ax2.set_ylabel("Marginal cost of steam [USD/kWh_th]")
ax2.set_ylim(bottom=0)

handles = [*ax.get_legend_handles_labels()[0], *ax2.get_legend_handles_labels()[0]]
labels = [*ax.get_legend_handles_labels()[1], *ax2.get_legend_handles_labels()[1]]
ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=6, frameon=False)
fig.tight_layout()

out_dir = _HERE / "outputs"
out_dir.mkdir(exist_ok=True)
stem = out_dir / "fuel_switching_timeseries"
for ext in ("png", "jpg", "eps"):
    fig.savefig(f"{stem}.{ext}", dpi=150)
print(f"\nSaved plot: {stem}.png / .jpg / .eps")
plt.show()


"""
Example 39: Full industrial steam plant.

Two lead/lag NG firetube boilers feed a main steam header, which supplies an HP
process directly and an LP process / LP utility through pressure reducing valves.
HP and LP process condensate return to the steam recovery loop; the LP-utility
condensate drains. Natural gas and make-up water are feedstocks. Blowdowns, header
loss, process losses and the drain are tracked, and make-up water is sized to
replace them so the plant water balance closes.

Run from this folder with the environment that has H2Integrate installed:
    python run_industrial_steam_plant.py
"""

import numpy as np
from pathlib import Path

from h2integrate import H2IntegrateModel


# Resolve the config relative to this file so the script runs from any working directory.
_HERE = Path(__file__).parent
model = H2IntegrateModel(str(_HERE / "39_industrial_steam_plant.yaml"))
model.setup()
model.run()

gv = model.prob.get_val


def total_kg_s(name):
    return float(np.mean(gv(name)))


# --- Steam production / demand ---
total_demand = gv("steam_load.total_steam_demand", units="kg/s")
b1 = gv("boiler_1.steam:mass_flow_out", units="kg/s")
b2 = gv("boiler_2.steam:mass_flow_out", units="kg/s")
print("\nSteam production (mean kg/s):")
print(f"  total process+header demand : {total_demand.mean():.3f}")
print(f"  boiler_1 (lead)             : {b1.mean():.3f}")
print(f"  boiler_2 (lag)              : {b2.mean():.3f}")
print(f"  boilers total              : {b1.mean() + b2.mean():.3f}")

# --- Fuel + emissions ---
ng1 = gv("boiler_1.total_natural_gas", units="MMBtu")
ng2 = gv("boiler_2.total_natural_gas", units="MMBtu")
co2 = gv("boiler_1.total_co2", units="kg") + gv("boiler_2.total_co2", units="kg")
ng_cost = gv("natural_gas_feedstock_1.VarOpEx", units="USD/year")[0] + gv(
    "natural_gas_feedstock_2.VarOpEx", units="USD/year"
)[0]
print("\nFuel and emissions (annual):")
print(f"  natural gas   : {ng1[0] + ng2[0]:,.0f} MMBtu/yr")
print(f"  NG cost       : {ng_cost:,.0f} USD/yr")
print(f"  CO2 emitted   : {co2[0] / 1000.0:,.0f} tonne/yr")

# --- Products ---
print("\nProcess product (annual):")
for p in ("hp_process", "lp_process", "lp_utility"):
    tp = gv(f"{p}.total_product", units="kg")
    print(f"  {p:11s} : {tp[0] / 1000.0:,.0f} tonne/yr")

# --- Water balance: make-up vs tracked losses ---
losses = {
    "boiler_1 blowdown": total_kg_s("boiler_1.blowdown"),
    "boiler_2 blowdown": total_kg_s("boiler_2.blowdown"),
    "hp_process loss": total_kg_s("hp_process.steam_loss"),
    "lp_process loss": total_kg_s("lp_process.steam_loss"),
    "lp_utility loss": total_kg_s("lp_utility.steam_loss"),
    "lp_utility drain": total_kg_s("lp_utility.steam:mass_flow_out"),
    "header loss": total_kg_s("steam_header_splitter.header_loss_flow"),
}
makeup = total_kg_s("makeup_water_intake.makeup_flow")
water_cost = gv("water_feedstock.VarOpEx", units="USD/year")[0]
print("\nWater balance (mean kg/s):")
for k, v in losses.items():
    print(f"  loss: {k:20s} {v:.4f}")
print(f"  => total tracked losses  {sum(losses.values()):.4f}")
print(f"     make-up water in      {makeup:.4f}")
print(f"     make-up water cost    {water_cost:,.0f} USD/yr")

# --- Boiler feedwater loop check ---
bfw = gv("boiler_feedwater_sink.mean_mass_flow", units="kg/s")[0]
cond = total_kg_s("condensate_surge_tank_combiner.steam:mass_flow_out")
da_steam = total_kg_s("deaerator.steam_demand")
vent = total_kg_s("deaerator.vent_flow")
print("\nRecovery loop (mean kg/s):")
print(f"  condensate returned : {cond:.3f}")
print(f"  make-up added       : {makeup:.3f}")
print(f"  deaeration steam  + : {da_steam:.3f}")
print(f"  deaerator vent    - : {vent:.4f}")
print(f"  boiler feedwater    : {bfw:.3f}")

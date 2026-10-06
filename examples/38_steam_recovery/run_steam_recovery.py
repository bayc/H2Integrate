"""
Example 38: Steam recovery loop with make-up water from a water feedstock.

Returning HP and LP condensate is collected in the condensate surge tank, topped
up with make-up water, pumped, deaerated, and delivered as boiler feedwater.

The ``makeup_water_intake`` computes how much make-up water the loop needs at each
time step (to replace deaerator vent + boiler blowdown losses) and exposes it as a
``water_consumed`` demand. A ``water_feedstock`` (FeedstockPerformanceModel +
FeedstockCostModel) supplies and prices that make-up water, following the same
feedstock convention used elsewhere in H2Integrate (e.g. examples 21 and 36).
"""

import numpy as np

from h2integrate import H2IntegrateModel


model = H2IntegrateModel("38_steam_recovery.yaml")
model.setup()
model.run()

# --- Make-up water demand computed by the loop -----------------------------
water_consumed = model.prob.get_val("makeup_water_intake.water_consumed", units="kg/h")
total_makeup = model.prob.get_val("makeup_water_intake.total_water_consumed", units="kg")
print("\nMake-up water demand (computed by the loop):")
print(f"  water_consumed: mean={water_consumed.mean():.1f} kg/h")
print(f"  water_consumed: peak={water_consumed.max():.1f} kg/h")
print(f"  total over simulation: {total_makeup[0]:,.0f} kg")

# --- Water feedstock supply + cost -----------------------------------------
fs_total = model.prob.get_val("water_feedstock.total_water_consumed", units="kg")
fs_varopex = model.prob.get_val("water_feedstock.VarOpEx", units="USD/year")
fs_cf = model.prob.get_val("water_feedstock.capacity_factor")
print("\nWater feedstock:")
print(f"  total water consumed: {fs_total[0]:,.0f} kg")
print(f"  variable water cost:  {fs_varopex[0]:,.0f} USD/year")
print(f"  capacity factor:      {np.mean(fs_cf):.3f}")

# --- Loop mass balance (kg/s), averaged over the horizon -------------------
cond = model.prob.get_val("condensate_surge_tank_combiner.steam:mass_flow_out", units="kg/s")
makeup = model.prob.get_val("makeup_water_intake.water_consumed", units="kg/s")
steam_draw = model.prob.get_val("deaerator.steam_demand", units="kg/s")
vent = model.prob.get_val("deaerator.vent_flow", units="kg/s")
bfw = model.prob.get_val("boiler_feedwater_sink.mean_mass_flow", units="kg/s")

expected_bfw = cond + makeup + steam_draw - vent
print("\nLoop mass balance (mean kg/s):")
print(f"  condensate returns:      {cond.mean():.4f}")
print(f"  make-up water:           {makeup.mean():.4f}")
print(f"  deaeration steam:      + {steam_draw.mean():.4f}")
print(f"  deaerator vent:        - {vent.mean():.4f}")
print(f"  => expected BFW:         {expected_bfw.mean():.4f}")
print(f"     reported BFW:         {bfw[0]:.4f}")

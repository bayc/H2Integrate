"""Data center size and heat-pump comparison for an urban Duluth, MN site.

For each of three data center sizes (10, 20, 30 MW IT workload) this script
runs two H2Integrate cases:

* ``dc_only``      -- data center + grid buy + water feedstock.
* ``dc_hp``        -- same, plus an electric heat pump (Carnot model) that
  upgrades the recovered waste heat to 140 F (60 C). The heat pump draws
  from a dedicated grid interconnection so its electricity cost is tracked
  separately from the data-center electricity cost.

The script reports, per case:

* total lifetime cost = sum(CapEx) + sum(OpEx * plant_life) + sum(VarOpEx over years).
* annual electricity consumption (facility + heat-pump).
* annual water consumption.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

DC_SIZES_MW = [10, 20, 30]
PLANT_LIFE = 15
N_TIMESTEPS = 8760
COST_YEAR = 2026
ELECTRICITY_PRICE_USD_PER_KWH = 0.04574
WATER_PRICE_USD_PER_GAL = 0.01

# Heat-pump value analysis
# Reference thermal-energy price levels for the breakeven analysis.
# Typical U.S. district-heating heat sale prices span roughly $20-70/MWh_th;
# avoiding a natural-gas boiler at $5/MMBtu with 85% efficiency is about
# $20/MWh_th; retail residential-scale thermal energy tends to be $40-60+.
REFERENCE_HEAT_VALUES_USD_PER_MWH_TH = {
    r"NG boiler avoided (\$5/MMBtu, 85% eff)": 20.0,
    "Wholesale DH sale": 30.0,
    "Retail thermal energy": 60.0,
}
HEAT_VALUE_SWEEP_USD_PER_MWH_TH = np.linspace(0.0, 100.0, 51)

# Duluth, MN
SITE_LAT = 46.7867
SITE_LON = -92.1005

# Cooling case 5: midsize DC water-cooled chiller. Recoverable fraction 0.20,
# waste-heat supply 40 C, return 30 C (Wahlroos 2018).
COOLING_CASE = 5

# Heat pump: upgrade recovered heat from 40 C source to 140 F (60 C) delivery.
HP_DELIVERY_TEMP_C = 60.0  # 140 F
HP_CARNOT_EFFICIENCY = 0.5


# ---------------------------------------------------------------------------
# Reusable tech-config blocks
# ---------------------------------------------------------------------------


def water_feedstock_tech():
    return {
        "performance_model": {"model": "FeedstockPerformanceModel"},
        "cost_model": {"model": "FeedstockCostModel"},
        "model_inputs": {
            "shared_parameters": {
                "commodity": "water",
                "commodity_rate_units": "galUS/h",
            },
            "performance_parameters": {"rated_capacity": 1.0e6},
            "cost_parameters": {
                "cost_year": COST_YEAR,
                "price": WATER_PRICE_USD_PER_GAL,
                "annual_cost": 0.0,
                "start_up_cost": 0.0,
            },
        },
    }


def grid_buy_tech(interconnection_kw):
    return {
        "performance_model": {"model": "GridPerformanceModel"},
        "cost_model": {"model": "GridCostModel"},
        "model_inputs": {
            "shared_parameters": {"interconnection_size": float(interconnection_kw)},
            "cost_parameters": {
                "cost_year": COST_YEAR,
                "electricity_buy_price": ELECTRICITY_PRICE_USD_PER_KWH,
                "interconnection_capex_per_kw": 0.0,
                "interconnection_opex_per_kw": 0.0,
                "fixed_interconnection_cost": 0.0,
            },
        },
    }


def data_center_tech(size_mw):
    """Data center at ``size_mw`` IT workload.

    Electricity and water costs are attributed to ``grid_buy`` and
    ``water_feedstock`` respectively (rates on the DC cost model are zeroed
    to avoid double-counting).
    """
    return {
        "performance_model": {"model": "DataCenterPUEWUEPerformanceModel"},
        "cost_model": {"model": "DataCenterPUEWUECostModel"},
        "model_inputs": {
            "shared_parameters": {"system_capacity_mw": float(size_mw)},
            "performance_parameters": {
                "compute_it_workload_profile": float(size_mw),
                "pue": 1.4,
                "wue": 1.0,
                "cooling_configuration": COOLING_CASE,
            },
            "cost_parameters": {
                "cost_year": COST_YEAR,
                "capex_per_mw": 4.0625e6,
                "fixed_opex_per_mw_per_year": 1.5e5,
                "electricity_rate": 0.0,
                "water_rate": 0.0,
                "waste_heat_sale_price_usd_per_mwh": 0.0,
            },
        },
    }


def heat_pump_tech(size_mw):
    """Carnot heat pump sized to comfortably handle DC waste heat.

    Case 5 recoverable waste heat = size_mw * 1.4 (PUE) * 0.20 ~= 0.28 * size_mw.
    Delivered heat = source + electric input; sizing HP at 0.5 * size_mw gives
    plenty of headroom.
    """
    return {
        "performance_model": {"model": "HeatPumpPerformanceModel"},
        "cost_model": {"model": "HeatPumpCostModel"},
        "model_inputs": {
            "shared_parameters": {"system_capacity_mw_th": float(size_mw) * 0.5},
            "performance_parameters": {
                "hp_mode": "carnot",
                "delivery_temp_C": HP_DELIVERY_TEMP_C,
                "carnot_efficiency": HP_CARNOT_EFFICIENCY,
                "min_source_temp_C": 5.0,
            },
            "cost_parameters": {
                "cost_year": COST_YEAR,
                "capex_per_mw_th": 8.0e5,
                "fixed_opex_per_mw_th_per_year": 2.0e4,
                "variable_opex_per_mwh_th": 1.0,
            },
        },
    }


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def build_plant_config(with_hp):
    interconnections = [
        ["data_center", "grid_buy", ["unmet_electricity_demand", "electricity_set_point"]],
        ["water_feedstock", "data_center", "water", "pipe"],
    ]
    if with_hp:
        interconnections += [
            ["data_center", "heat_pump", ["waste_heat_out", "heat_in"]],
            [
                "data_center",
                "heat_pump",
                ["waste_heat_supply_temp_C", "heat_supply_temp_C_in"],
            ],
            ["grid_buy_hp", "heat_pump", "electricity", "cable"],
            [
                "heat_pump",
                "grid_buy_hp",
                ["unmet_electricity_demand", "electricity_set_point"],
            ],
        ]
    return {
        "name": "plant_config",
        "description": "Duluth, MN data center size sweep",
        "sites": {"site": {"latitude": SITE_LAT, "longitude": SITE_LON}},
        "technology_interconnections": interconnections,
        "plant": {
            "plant_life": PLANT_LIFE,
            "simulation": {"n_timesteps": N_TIMESTEPS},
        },
    }


def build_full_config(size_mw, with_hp):
    technologies = {
        "water_feedstock": water_feedstock_tech(),
        "data_center": data_center_tech(size_mw),
        "grid_buy": grid_buy_tech(interconnection_kw=1.0e6),
    }
    if with_hp:
        technologies["heat_pump"] = heat_pump_tech(size_mw)
        technologies["grid_buy_hp"] = grid_buy_tech(interconnection_kw=1.0e5)

    return {
        "name": "datacenter_size_sweep",
        "system_summary": f"{size_mw} MW DC, {'with' if with_hp else 'without'} heat pump",
        "driver_config": {
            "name": "driver_config",
            "description": "Data center size sweep for Duluth, MN",
            "general": {"folder_output": "outputs", "create_om_reports": False},
        },
        "technology_config": {
            "name": "technology_config",
            "description": f"{size_mw} MW data center " + ("with heat pump" if with_hp else "only"),
            "technologies": deepcopy(technologies),
        },
        "plant_config": build_plant_config(with_hp),
    }


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def _sum_get(prob, name, units):
    return float(prob.get_val(name, units=units).sum())


def _scalar_get(prob, name, units):
    return float(prob.get_val(name, units=units)[0])


def collect_metrics(prob, with_hp):
    dc_capex = _scalar_get(prob, "data_center.CapEx", "USD")
    dc_opex = _scalar_get(prob, "data_center.OpEx", "USD/year")

    # Grid buy VarOpEx has shape = plant_life; sum for lifetime cost.
    grid_var_life = _sum_get(prob, "grid_buy.VarOpEx", "USD/year")
    grid_opex = _scalar_get(prob, "grid_buy.OpEx", "USD/year")
    grid_capex = _scalar_get(prob, "grid_buy.CapEx", "USD")

    wf_capex = _scalar_get(prob, "water_feedstock.CapEx", "USD")
    wf_opex = _scalar_get(prob, "water_feedstock.OpEx", "USD/year")
    wf_var_life = _sum_get(prob, "water_feedstock.VarOpEx", "USD/year")

    hp_capex = 0.0
    hp_opex = 0.0
    hp_grid_capex = 0.0
    hp_grid_opex = 0.0
    hp_grid_var_life = 0.0
    hp_elec_mwh_per_yr = 0.0
    hp_heat_delivered_mwh_per_yr = 0.0
    if with_hp:
        hp_capex = _scalar_get(prob, "heat_pump.CapEx", "USD")
        hp_opex = _scalar_get(prob, "heat_pump.OpEx", "USD/year")
        hp_grid_capex = _scalar_get(prob, "grid_buy_hp.CapEx", "USD")
        hp_grid_opex = _scalar_get(prob, "grid_buy_hp.OpEx", "USD/year")
        hp_grid_var_life = _sum_get(prob, "grid_buy_hp.VarOpEx", "USD/year")
        # 1 h dt so sum(MW) = MWh.
        hp_elec_mwh_per_yr = _sum_get(prob, "heat_pump.electricity_used", "MW")
        hp_heat_delivered_mwh_per_yr = _sum_get(prob, "heat_pump.heat_out", "MW")

    # Facility electricity (MW * 1 h = MWh) and water consumption (galUS/h * 1 h = galUS)
    facility_elec_mwh_per_yr = _sum_get(prob, "data_center.total_facility_power", "MW")
    water_gal_per_yr = _sum_get(prob, "data_center.water_consumed", "galUS/h")

    total_capex = dc_capex + wf_capex + grid_capex + hp_capex + hp_grid_capex
    total_annual_opex = dc_opex + wf_opex + grid_opex + hp_opex + hp_grid_opex
    total_var_lifetime = grid_var_life + wf_var_life + hp_grid_var_life

    lifetime_cost = total_capex + total_annual_opex * PLANT_LIFE + total_var_lifetime

    return {
        "total_capex_musd": total_capex / 1e6,
        "annual_fixed_opex_musd_per_yr": total_annual_opex / 1e6,
        "lifetime_variable_opex_musd": total_var_lifetime / 1e6,
        "lifetime_cost_musd": lifetime_cost / 1e6,
        "annual_electricity_gwh": (facility_elec_mwh_per_yr + hp_elec_mwh_per_yr) / 1e3,
        "annual_facility_electricity_gwh": facility_elec_mwh_per_yr / 1e3,
        "annual_hp_electricity_gwh": hp_elec_mwh_per_yr / 1e3,
        "annual_water_kgal": water_gal_per_yr / 1e3,
        "annual_hp_heat_delivered_gwh_th": hp_heat_delivered_mwh_per_yr / 1e3,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_sweep():
    rows = []
    for size in DC_SIZES_MW:
        for with_hp in (False, True):
            case_name = f"{size}MW_{'dc_hp' if with_hp else 'dc_only'}"
            print(f"Running {case_name} ...")

            model = H2IntegrateModel(build_full_config(size, with_hp))
            model.setup()
            model.run()

            metrics = collect_metrics(model.prob, with_hp)
            metrics.update({"case": case_name, "dc_size_mw": size, "with_hp": with_hp})
            rows.append(metrics)

    df = pd.DataFrame(rows).set_index("case")
    # Column ordering
    cols = [
        "dc_size_mw",
        "with_hp",
        "total_capex_musd",
        "annual_fixed_opex_musd_per_yr",
        "lifetime_variable_opex_musd",
        "lifetime_cost_musd",
        "annual_facility_electricity_gwh",
        "annual_hp_electricity_gwh",
        "annual_electricity_gwh",
        "annual_water_kgal",
        "annual_hp_heat_delivered_gwh_th",
    ]
    return df[cols]


def make_plots(df, out_dir):
    """Bar charts: cost, electricity, water, heat delivered — grouped by DC size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = sorted(df["dc_size_mw"].unique())
    x = np.arange(len(sizes))
    width = 0.38

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    def _bars(ax, column, ylabel, title):
        dc_only = df[df["with_hp"] == False].sort_values("dc_size_mw")[column].values  # noqa: E712
        dc_hp = df[df["with_hp"] == True].sort_values("dc_size_mw")[column].values  # noqa: E712
        ax.bar(x - width / 2, dc_only, width, label="DC only", color="#3366cc")
        ax.bar(x + width / 2, dc_hp, width, label="DC + heat pump", color="#2e7d32")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s} MW" for s in sizes])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    _bars(axes[0], "lifetime_cost_musd", "Lifetime cost (M USD)", "Total lifetime cost")
    _bars(
        axes[1],
        "annual_electricity_gwh",
        "Electricity (GWh/yr)",
        "Annual electricity consumption",
    )
    _bars(axes[2], "annual_water_kgal", "Water (kgal/yr)", "Annual water consumption")
    _bars(
        axes[3],
        "annual_hp_heat_delivered_gwh_th",
        f"Heat delivered @ {HP_DELIVERY_TEMP_C:.0f} C (GWh$_{{th}}$/yr)",
        "Annual heat pump heat delivered",
    )

    fig.suptitle(
        f"Duluth, MN data center size sweep ({PLANT_LIFE}-year plant life, "
        f"cooling case {COOLING_CASE}, HP delivery {HP_DELIVERY_TEMP_C:.0f} C / 140 F)"
    )
    fig.tight_layout()

    out_path = out_dir / "size_sweep_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Heat-pump economic value analysis
# ---------------------------------------------------------------------------


def compute_hp_economics(df):
    """Compare the lifetime cost of adding the heat pump to the lifetime value
    of the heat it delivers.

    Incremental HP-related cost (relative to the DC-only case) = added CapEx,
    added fixed OpEx (× plant_life), and added variable OpEx (the electricity
    the HP consumes, over the full plant life).

    Value = user-chosen thermal-energy price × lifetime heat delivered.

    Returns a DataFrame indexed by DC size with per-size incremental cost,
    lifetime heat, breakeven thermal-energy price, and net lifetime benefit
    evaluated at each ``REFERENCE_HEAT_VALUES_USD_PER_MWH_TH``.
    """
    dc_only = df[~df["with_hp"]].set_index("dc_size_mw").sort_index()
    dc_hp = df[df["with_hp"]].set_index("dc_size_mw").sort_index()

    rows = []
    for size in dc_hp.index:
        incremental_capex_musd = dc_hp.at[size, "total_capex_musd"] - dc_only.at[
            size, "total_capex_musd"
        ]
        incremental_annual_opex_musd = (
            dc_hp.at[size, "annual_fixed_opex_musd_per_yr"]
            - dc_only.at[size, "annual_fixed_opex_musd_per_yr"]
        )
        incremental_lifetime_varopex_musd = (
            dc_hp.at[size, "lifetime_variable_opex_musd"]
            - dc_only.at[size, "lifetime_variable_opex_musd"]
        )
        incremental_lifetime_cost_musd = (
            incremental_capex_musd
            + incremental_annual_opex_musd * PLANT_LIFE
            + incremental_lifetime_varopex_musd
        )

        annual_heat_gwh_th = dc_hp.at[size, "annual_hp_heat_delivered_gwh_th"]
        # 1 GWh = 1e3 MWh
        lifetime_heat_mwh_th = annual_heat_gwh_th * 1e3 * PLANT_LIFE

        breakeven_usd_per_mwh_th = (
            incremental_lifetime_cost_musd * 1e6 / lifetime_heat_mwh_th
            if lifetime_heat_mwh_th > 0
            else np.nan
        )

        row = {
            "dc_size_mw": size,
            "incremental_capex_musd": incremental_capex_musd,
            "incremental_annual_opex_musd_per_yr": incremental_annual_opex_musd,
            "incremental_lifetime_varopex_musd": incremental_lifetime_varopex_musd,
            "incremental_lifetime_cost_musd": incremental_lifetime_cost_musd,
            "lifetime_heat_delivered_gwh_th": lifetime_heat_mwh_th / 1e3,
            "breakeven_heat_price_usd_per_mwh_th": breakeven_usd_per_mwh_th,
        }
        for label, price in REFERENCE_HEAT_VALUES_USD_PER_MWH_TH.items():
            row[f"net_benefit_at_{price:.0f}_usd_per_mwh_musd"] = (
                price * lifetime_heat_mwh_th / 1e6 - incremental_lifetime_cost_musd
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("dc_size_mw")


def make_economics_plots(econ_df, out_dir):
    """Two-panel economic plot: breakeven price per size + net benefit vs price."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = econ_df.index.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: breakeven thermal-energy price per DC size, with reference lines
    ax = axes[0]
    breakevens = econ_df["breakeven_heat_price_usd_per_mwh_th"].to_numpy()
    x = np.arange(len(sizes))
    ax.bar(x, breakevens, color="#8e44ad", label="Breakeven price")
    for label, price in REFERENCE_HEAT_VALUES_USD_PER_MWH_TH.items():
        ax.axhline(price, linestyle="--", alpha=0.7, label=rf"{label} (\${price:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} MW" for s in sizes])
    ax.set_ylabel("Heat price to break even (USD/MWh$_{th}$)")
    ax.set_title("Heat-pump breakeven heat price")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: net lifetime benefit vs assumed heat value, one line per size
    ax = axes[1]
    prices = HEAT_VALUE_SWEEP_USD_PER_MWH_TH
    for size in sizes:
        lifetime_heat_mwh_th = econ_df.at[size, "lifetime_heat_delivered_gwh_th"] * 1e3
        incr_cost_musd = econ_df.at[size, "incremental_lifetime_cost_musd"]
        net_benefit_musd = prices * lifetime_heat_mwh_th / 1e6 - incr_cost_musd
        ax.plot(prices, net_benefit_musd, label=f"{size} MW DC", linewidth=2)
    ax.axhline(0.0, color="k", linewidth=0.8)
    for label, price in REFERENCE_HEAT_VALUES_USD_PER_MWH_TH.items():
        ax.axvline(price, linestyle="--", alpha=0.5, color="gray")
    ax.set_xlabel("Assumed thermal-energy value (USD/MWh$_{th}$)")
    ax.set_ylabel("Net lifetime benefit of HP (M USD)")
    ax.set_title(f"Net {PLANT_LIFE}-yr benefit vs. heat value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        f"Heat-pump value analysis (Duluth, MN, {PLANT_LIFE}-yr life, "
        f"delivery {HP_DELIVERY_TEMP_C:.0f} C / 140 F)"
    )
    fig.tight_layout()

    out_path = out_dir / "heat_pump_economics.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    df = run_sweep()
    print()
    print("=" * 100)
    print(df.to_string(float_format=lambda x: f"{x:,.3f}"))
    print("=" * 100)

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "size_sweep_results.csv"
    df.to_csv(csv_path)
    print(f"Saved results table to {csv_path}")

    make_plots(df.reset_index(), out_dir)

    # -- Heat-pump economic value analysis -----------------------------------
    econ_df = compute_hp_economics(df.reset_index())
    print()
    print("Heat-pump economics (incremental cost vs. lifetime heat delivered)")
    print("=" * 100)
    print(econ_df.to_string(float_format=lambda x: f"{x:,.3f}"))
    print("=" * 100)

    econ_csv_path = out_dir / "heat_pump_economics.csv"
    econ_df.to_csv(econ_csv_path)
    print(f"Saved economics table to {econ_csv_path}")

    make_economics_plots(econ_df, out_dir)


if __name__ == "__main__":
    main()

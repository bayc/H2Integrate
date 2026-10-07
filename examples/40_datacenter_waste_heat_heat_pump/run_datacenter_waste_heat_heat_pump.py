"""Run the data-center + heat-pump + district-heating example.

The data center (PUE/WUE model, air-cooled cooling case 9) exports low-grade
waste heat (~28 C) which is upgraded by an electric heat pump (Carnot model,
delivery at 75 C) before being consumed by a district-heating loop with a
70 C minimum supply temperature.
"""

from pathlib import Path

from h2integrate.core.h2integrate_model import H2IntegrateModel


def main():
    config = Path(__file__).parent / "datacenter_waste_heat_heat_pump.yaml"
    model = H2IntegrateModel(str(config))
    model.setup()
    model.run()
    model.post_process()

    dc_waste = model.prob.get_val("data_center.total_waste_heat_recovered", units="MW*h")[0]
    hp_heat_out = model.prob.get_val("heat_pump.heat_out", units="MW").sum()
    hp_elec = model.prob.get_val("heat_pump.electricity_used", units="MW").sum()
    hp_cop = model.prob.get_val("heat_pump.cop_actual", units="unitless").mean()
    dh_delivered = model.prob.get_val("district_heating.heat_out", units="MW").sum()
    dh_unmet = model.prob.get_val("district_heating.unmet_heat_demand_out", units="MW").sum()

    print(f"Data center waste heat (source): {dc_waste:,.0f} MWh/yr")
    print(f"Heat pump heat delivered:        {hp_heat_out:,.0f} MW-hours")
    print(f"Heat pump electricity used:      {hp_elec:,.0f} MW-hours")
    print(f"Mean heat pump COP:              {hp_cop:.2f}")
    print(f"DH heat delivered:               {dh_delivered:,.0f} MW-hours")
    print(f"DH unmet demand:                 {dh_unmet:,.0f} MW-hours")


if __name__ == "__main__":
    main()

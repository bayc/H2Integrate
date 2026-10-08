"""Run the data-center waste-heat-to-district-heating example.

The data center (PUE/WUE model, cooling case 5) exports its recovered waste
heat directly to a district-heating loop whose minimum supply temperature is
compatible with the data center's cooling return-water temperature.
"""

from pathlib import Path

from h2integrate.core.h2integrate_model import H2IntegrateModel


def main():
    config = Path(__file__).parent / "datacenter_waste_heat_direct.yaml"
    model = H2IntegrateModel(str(config))
    model.setup()
    model.run()
    model.post_process()

    total_waste_heat = model.prob.get_val("data_center.total_waste_heat_recovered", units="MW*h")[0]
    heat_delivered = model.prob.get_val("district_heating.heat_out", units="MW").sum()
    unmet = model.prob.get_val("district_heating.unmet_heat_demand_out", units="MW").sum()

    print(f"Total waste heat recovered: {total_waste_heat:,.0f} MWh/yr")
    print(f"Heat delivered to DH loop:  {heat_delivered:,.0f} MW-hours")
    print(f"Unmet DH demand:            {unmet:,.0f} MW-hours")


if __name__ == "__main__":
    main()

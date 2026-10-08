import numpy as np
import pytest
import openmdao.api as om
from pytest import approx, fixture

from h2integrate.demand.district_heating import (
    DistrictHeatingDemand,
    DistrictHeatingDemandCostModel,
)


@fixture
def plant_config():
    return {
        "plant": {
            "plant_life": 30,
            "simulation": {"n_timesteps": 24, "dt": 3600},
        },
    }


def _make_perf_config(**overrides):
    params = {
        "commodity": "heat",
        "commodity_rate_units": "MW",
        "demand_profile": 1.0,
        "min_supply_temp_C": 60.0,
        "system_capacity_mw_th": 2.0,
    }
    params.update(overrides)
    return {"model_inputs": {"performance_parameters": params}}


def _make_cost_config(**overrides):
    params = {
        "cost_year": 2022,
        "system_capacity_mw_th": 2.0,
        "capex_per_mw_th": 250_000.0,
        "fixed_opex_per_mw_th_per_year": 5_000.0,
        "variable_opex_per_mwh_th": 1.0,
    }
    params.update(overrides)
    return {"model_inputs": {"cost_parameters": params}}


@pytest.mark.unit
class TestDistrictHeatingDemand:
    def _build(self, plant_config, tech_config):
        prob = om.Problem()
        prob.model.add_subsystem(
            "dh",
            DistrictHeatingDemand(plant_config=plant_config, tech_config=tech_config),
            promotes=["*"],
        )
        prob.setup()
        return prob

    def test_demand_met_when_supply_matches(self, plant_config):
        prob = self._build(plant_config, _make_perf_config())
        prob.set_val("heat_in", np.full(24, 1.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 70.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("unmet_heat_demand_out", units="MW") == approx(np.zeros(24))
        assert prob.get_val("unused_heat_out", units="MW") == approx(np.zeros(24))
        assert prob.get_val("temperature_shortfall_flag")[0] == approx(0.0)

    def test_unmet_demand_tracked(self, plant_config):
        prob = self._build(plant_config, _make_perf_config(demand_profile=2.0))
        prob.set_val("heat_in", np.full(24, 1.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 70.0, units="degC")
        prob.run_model()

        assert prob.get_val("unmet_heat_demand_out", units="MW") == approx(np.full(24, 1.0))

    def test_supply_temperature_below_min_zeros_delivery(self, plant_config):
        prob = self._build(plant_config, _make_perf_config())
        prob.set_val("heat_in", np.full(24, 1.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 50.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.zeros(24))
        assert prob.get_val("unmet_heat_demand_out", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("temperature_shortfall_flag")[0] == approx(1.0)
        # Full simulation length (24 h) reported as temperature shortfall.
        assert prob.get_val("temperature_shortfall_hours", units="h")[0] == approx(24.0)

    def test_non_strict_mode_allows_low_temp_delivery(self, plant_config):
        prob = self._build(
            plant_config,
            _make_perf_config(strict_temperature=False),
        )
        prob.set_val("heat_in", np.full(24, 1.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 50.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("temperature_shortfall_flag")[0] == approx(1.0)


@pytest.mark.unit
class TestDistrictHeatingDemandCostModel:
    def test_capex_and_opex(self, plant_config):
        prob = om.Problem()
        prob.model.add_subsystem(
            "dh_cost",
            DistrictHeatingDemandCostModel(
                plant_config=plant_config, tech_config=_make_cost_config()
            ),
            promotes=["*"],
        )
        prob.setup()
        prob.set_val("heat_out", np.full(24, 1.0), units="MW")
        prob.run_model()

        # capex = 250_000 * 2 = 500_000; fixed_om = 5_000 * 2 = 10_000;
        # variable_om = 1 * 24 = 24; opex = 10_024
        assert prob.get_val("CapEx", units="USD")[0] == approx(500_000.0)
        assert prob.get_val("OpEx", units="USD/year")[0] == approx(10_024.0)

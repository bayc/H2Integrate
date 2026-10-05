import numpy as np
import pytest
import openmdao.api as om
from pytest import approx, fixture

from h2integrate.converters.heat.heat_pump import HeatPumpCostModel, HeatPumpPerformanceModel


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
        "hp_mode": "fixed_cop",
        "system_capacity_mw_th": 2.0,
        "delivery_temp_C": 70.0,
        "cop": 4.0,
    }
    params.update(overrides)
    return {"model_inputs": {"performance_parameters": params}}


def _make_cost_config(**overrides):
    params = {
        "cost_year": 2022,
        "system_capacity_mw_th": 2.0,
        "capex_per_mw_th": 500_000.0,
        "fixed_opex_per_mw_th_per_year": 15_000.0,
        "variable_opex_per_mwh_th": 2.0,
    }
    params.update(overrides)
    return {"model_inputs": {"cost_parameters": params}}


@pytest.mark.unit
class TestHeatPumpPerformanceModel:
    def _build(self, plant_config, tech_config):
        prob = om.Problem()
        prob.model.add_subsystem(
            "hp",
            HeatPumpPerformanceModel(plant_config=plant_config, tech_config=tech_config),
            promotes=["*"],
        )
        prob.setup()
        return prob

    def test_fixed_cop_capacity_limited(self, plant_config):
        # Ample heat_in and electricity_in -> capacity-limited delivery at 2 MW,
        # elec use = 2 / cop = 0.5 MW.
        prob = self._build(plant_config, _make_perf_config())
        prob.set_val("heat_in", np.full(24, 5.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 2.0))
        assert prob.get_val("electricity_used", units="MW") == approx(np.full(24, 0.5))
        assert prob.get_val("cop_actual") == approx(np.full(24, 4.0))
        assert prob.get_val("heat_supply_temp_C", units="degC") == approx(70.0)

    def test_fixed_cop_source_limited(self, plant_config):
        # heat_in = 1.5 MW -> delivered = heat_in / (1 - 1/cop) = 1.5 / 0.75 = 2.0
        prob = self._build(plant_config, _make_perf_config())
        prob.set_val("heat_in", np.full(24, 1.5), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 2.0))

    def test_fixed_cop_electricity_limited(self, plant_config):
        # electricity_in = 0.25 MW, cop = 4 -> delivered = 0.25 * 4 = 1.0 MW
        prob = self._build(plant_config, _make_perf_config())
        prob.set_val("heat_in", np.full(24, 10.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 0.25), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("electricity_used", units="MW") == approx(np.full(24, 0.25))

    def test_carnot_cop_matches_theory(self, plant_config):
        # COP = eta * T_sink / (T_sink - T_source); sink=70C, source=30C, eta=0.5
        cop_expected = 0.5 * (70 + 273.15) / ((70 + 273.15) - (30 + 273.15))
        prob = self._build(
            plant_config,
            _make_perf_config(hp_mode="carnot", cop=None, carnot_efficiency=0.5),
        )
        prob.set_val("heat_in", np.full(24, 5.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("cop_actual") == approx(np.full(24, cop_expected), rel=1e-6)
        # Delivered is capacity-limited at 2 MW; electricity used = 2 / cop.
        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 2.0))
        assert prob.get_val("electricity_used", units="MW") == approx(
            np.full(24, 2.0 / cop_expected), rel=1e-6
        )

    def test_carnot_infeasible_when_source_above_sink(self, plant_config):
        prob = self._build(
            plant_config,
            _make_perf_config(hp_mode="carnot", cop=None, carnot_efficiency=0.5),
        )
        prob.set_val("heat_in", np.full(24, 5.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 90.0, units="degC")  # above delivery
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.zeros(24))
        assert prob.get_val("cop_actual") == approx(np.zeros(24))

    def test_fixed_cop_requires_cop_value(self, plant_config):
        with pytest.raises(ValueError, match="cop"):
            HeatPumpPerformanceModel(
                plant_config=plant_config,
                tech_config=_make_perf_config(cop=None),
            ).setup()

    def test_throttle_to_demand_true_scales_electricity(self, plant_config):
        # cop=4, capacity=2 MW, demand ceiling=1 MW everywhere. Under the
        # default throttling policy the HP produces only what the off-taker
        # can take: heat_out = heat_delivered = 1 MW, heat_curtailed = 1 MW
        # (foregone potential), electricity_used = 1/4 = 0.25 MW.
        cfg = _make_perf_config(heat_demand_profile=[1.0] * 24)
        prob = self._build(plant_config, cfg)
        prob.set_val("heat_in", np.full(24, 5.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("heat_delivered", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("heat_curtailed", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("electricity_used", units="MW") == approx(np.full(24, 0.25))
        # Utilization = delivered / potential = 1 / 2.
        assert prob.get_val("dh_utilization_fraction") == approx(0.5)

    def test_throttle_to_demand_false_runs_at_potential(self, plant_config):
        # Same setup but legacy dispatch: HP runs to supply-limited potential
        # (2 MW), delivers 1 MW, dumps the other 1 MW as heat_curtailed, and
        # pays electricity for the full 2 MW production (2/4 = 0.5 MW).
        cfg = _make_perf_config(
            heat_demand_profile=[1.0] * 24, throttle_to_demand=False
        )
        prob = self._build(plant_config, cfg)
        prob.set_val("heat_in", np.full(24, 5.0), units="MW")
        prob.set_val("electricity_in", np.full(24, 5.0), units="MW")
        prob.set_val("heat_supply_temp_C_in", 30.0, units="degC")
        prob.run_model()

        assert prob.get_val("heat_out", units="MW") == approx(np.full(24, 2.0))
        assert prob.get_val("heat_delivered", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("heat_curtailed", units="MW") == approx(np.full(24, 1.0))
        assert prob.get_val("electricity_used", units="MW") == approx(np.full(24, 0.5))
        assert prob.get_val("dh_utilization_fraction") == approx(0.5)


@pytest.mark.unit
class TestHeatPumpCostModel:
    def _build(self, plant_config, tech_config):
        prob = om.Problem()
        prob.model.add_subsystem(
            "hp_cost",
            HeatPumpCostModel(plant_config=plant_config, tech_config=tech_config),
            promotes=["*"],
        )
        prob.setup()
        return prob

    def test_capex_and_opex(self, plant_config):
        prob = self._build(plant_config, _make_cost_config())
        # Deliver 2 MW for 24 h -> 48 MWh over the simulated year fraction.
        prob.set_val("heat_delivered", np.full(24, 2.0), units="MW")
        prob.run_model()

        # Annualization: 24 h simulated out of 8760 h/yr -> annual = 48 * 365
        # = 17520 MWh/yr. variable_om = 2 USD/MWh * 17520 = 35_040 USD/yr,
        # constant over all plant_life=30 years.
        # capex = 500_000 * 2 = 1_000_000; fixed_om = 15_000 * 2 = 30_000/yr.
        assert prob.get_val("CapEx", units="USD")[0] == approx(1_000_000.0)
        assert prob.get_val("OpEx", units="USD/year")[0] == approx(30_000.0)
        var = prob.get_val("VarOpEx", units="USD/year")
        assert var.shape == (30,)
        assert var == approx(np.full(30, 35_040.0))

    def test_bulk_trench_cost_overrides_diameter_model(self, plant_config):
        # 100 m of trench at 10 USD/ft -> 100 * 3.28084 * 10 = 3280.84 USD,
        # regardless of pipe diameter. Also add a nonzero diameter-based
        # cost to make sure it is ignored when the bulk override is set.
        cost_cfg = _make_cost_config(
            dh_connection_distance_m=100.0,
            dh_pipe_fixed_cost_usd_per_m=999.0,
            dh_pipe_cost_per_mm_diameter_usd_per_m=999.0,
            dh_trench_bulk_cost_usd_per_ft=10.0,
            dh_pipe_annual_om_fraction=0.02,
        )
        prob = self._build(plant_config, cost_cfg)
        prob.set_val("heat_delivered", np.zeros(24), units="MW")
        prob.run_model()

        expected_dh_capex = 10.0 * 100.0 * 3.28084
        assert prob.get_val("dh_connection_capex", units="USD")[0] == approx(
            expected_dh_capex
        )
        assert prob.get_val("dh_connection_opex", units="USD/year")[0] == approx(
            expected_dh_capex * 0.02
        )
        # Pipe diameter still reported (sized off HP capacity=2 MW, coeff=71).
        assert prob.get_val("dh_pipe_diameter_mm", units="mm")[0] == approx(
            71.0 * np.sqrt(2.0)
        )
        # HP CapEx (500k * 2 MW) + DH trench CapEx sum into CapEx.
        assert prob.get_val("CapEx", units="USD")[0] == approx(
            1_000_000.0 + expected_dh_capex
        )

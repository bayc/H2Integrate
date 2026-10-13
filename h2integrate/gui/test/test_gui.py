"""Tests for the Streamlit-independent GUI helpers and a smoke test of the app."""

import shutil
from pathlib import Path

import pytest

from h2integrate import EXAMPLE_DIR
from h2integrate.gui import backend as be


EXAMPLE_08 = EXAMPLE_DIR / "08_wind_electrolyzer" / "wind_plant_electrolyzer.yaml"


@pytest.mark.unit
def test_model_roles():
    roles = be.models_by_role()
    assert "PYSAMWindPlantPerformanceModel" in roles["performance_model"]
    assert "ATBWindPlantCostModel" in roles["cost_model"]
    assert "DemandOpenLoopStorageController" in roles["control_strategy"]
    assert "PyomoRuleStorageBaseclass" in roles["dispatch_rule_set"]
    assert "ProFastLCO" in roles["finance"]


@pytest.mark.unit
def test_find_example_configs():
    examples = be.find_example_configs()
    assert examples["08_wind_electrolyzer/wind_plant_electrolyzer.yaml"] == EXAMPLE_08


@pytest.mark.unit
def test_load_project_round_trips_tables():
    project = be.load_project(EXAMPLE_08)
    technologies = project["technology_config"]["technologies"]
    assert set(technologies) == {"wind", "electrolyzer"}
    # !include files are expanded
    wind_inputs = technologies["wind"]["model_inputs"]["performance_parameters"]
    assert isinstance(wind_inputs["pysam_options"], dict)

    rows = be.technologies_to_rows(project["technology_config"])
    new_technologies, changed = be.rows_to_technologies(rows, technologies)
    assert new_technologies == technologies
    assert changed == []

    for tech, info in technologies.items():
        rows = be.model_inputs_to_rows(tech, info)
        assert be.rows_to_model_inputs(rows) == info["model_inputs"]

    connections = project["plant_config"]["technology_interconnections"]
    rows = be.interconnections_to_rows(project["plant_config"])
    assert be.rows_to_interconnections(rows) == connections
    assert be.validate_project(project) == []


@pytest.mark.unit
def test_rows_to_technologies_detects_changes_and_errors():
    old = {"wind": {"performance_model": {"model": "A"}, "model_inputs": {"x": {"a": 1}}}}
    rows = [
        {"technology": "wind", "performance_model": "B", "cost_model": None},
        {"technology": "solar", "performance_model": "C", "cost_model": "D"},
        {"technology": None, "performance_model": None},
    ]
    technologies, changed = be.rows_to_technologies(rows, old)
    assert changed == ["wind", "solar"]
    assert technologies["wind"] == {
        "performance_model": {"model": "B"},
        "model_inputs": {"x": {"a": 1}},
    }
    assert technologies["solar"]["cost_model"] == {"model": "D"}

    with pytest.raises(ValueError, match="more than once"):
        be.rows_to_technologies([rows[0], rows[0]], old)
    with pytest.raises(ValueError, match="needs a performance model"):
        be.rows_to_technologies([{"technology": "wind", "performance_model": None}], old)


@pytest.mark.unit
def test_yaml_cells():
    for value in [1.5, 3, True, "text", "01/01 00:30:00", [1, 2], {"a": {"b": [1, "x"]}}]:
        assert be.from_yaml_str(be.to_yaml_str(value)) == value
    assert be.to_yaml_str(None) == ""
    assert be.from_yaml_str("") is None
    assert be.from_yaml_str(float("nan")) is None


@pytest.mark.unit
def test_rows_to_model_inputs_errors():
    with pytest.raises(ValueError, match="not valid YAML"):
        be.rows_to_model_inputs([{"section": "cost_parameters", "parameter": "a", "value": "[1"}])
    with pytest.raises(ValueError, match="more than once"):
        be.rows_to_model_inputs([{"section": "s", "parameter": "a", "value": "1"}] * 2)


@pytest.mark.unit
def test_rows_to_interconnections():
    rows = [
        {"source": "a", "destination": "b", "connection": "[heat_out, heat_in]", "transport": None},
        {"source": "a", "destination": "c", "connection": "electricity", "transport": "cable"},
    ]
    assert be.rows_to_interconnections(rows) == [
        ["a", "b", ["heat_out", "heat_in"]],
        ["a", "c", "electricity", "cable"],
    ]
    with pytest.raises(ValueError, match="source and a destination"):
        be.rows_to_interconnections([{"source": "a", "destination": None}])
    with pytest.raises(ValueError, match="needs a commodity"):
        be.rows_to_interconnections([{"source": "a", "destination": "b", "connection": ""}])


@pytest.mark.unit
def test_fill_default_inputs_keeps_existing_values():
    tech_info = {
        "performance_model": {"model": "StoragePerformanceModel"},
        "model_inputs": {"shared_parameters": {"commodity": "hydrogen"}},
    }
    added = be.fill_default_inputs(tech_info)
    model_inputs = tech_info["model_inputs"]
    assert "max_capacity" in added
    assert "commodity" not in added
    assert model_inputs["shared_parameters"]["commodity"] == "hydrogen"
    assert be.fill_default_inputs(tech_info) == []


@pytest.mark.unit
def test_export_project(tmp_path):
    project = be.load_project(EXAMPLE_08)
    top_level_path = be.export_project(project, tmp_path)
    assert top_level_path.exists()
    reloaded = be.load_project(top_level_path)
    for key in be.CONFIG_KEYS:
        assert reloaded[key] == project[key]


@pytest.mark.integration
def test_run_example_through_runner(tmp_path):
    example_dir = shutil.copytree(EXAMPLE_DIR / "16_natural_gas", tmp_path / "16_natural_gas")
    project = be.load_project(Path(example_dir) / "natgas.yaml")
    process, _ = be.start_run(project, example_dir)
    output, _ = process.communicate(timeout=600)
    assert process.returncode == 0, output

    results = be.load_results(be.results_dir(example_dir))
    assert results["n_timesteps"] == 8760
    assert "finance_subgroup_electricity.LCOE" in results["scalars"]
    assert len(results["arrays"]["battery.SOC"]) == 8760


@pytest.mark.unit
def test_app_smoke():
    testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(be.__file__).parent / "app.py"
    at = testing.AppTest.from_file(str(app_path), default_timeout=120).run()
    assert not at.exception

    # Load the first example from the sidebar
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.session_state.project["technology_config"]["technologies"]

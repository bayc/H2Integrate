"""Streamlit-independent helpers used by the H2Integrate GUI.

Keeping this logic out of ``app.py`` allows it to be unit tested without Streamlit.
"""

import os
import re
import sys
import copy
import json
import subprocess
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from h2integrate import EXAMPLE_DIR
from h2integrate.core.file_utils import find_file, load_yaml
from h2integrate.core.supported_models import supported_models
from h2integrate.preprocess.populate_tech_yaml import (
    MODEL_SECTION_NAMES,
    _validator_comments,
    organize_model_parameters,
)


CONFIG_KEYS = ("driver_config", "technology_config", "plant_config")
MODEL_SLOTS = tuple(MODEL_SECTION_NAMES)
PARAM_SECTIONS = (
    "shared_parameters",
    "performance_parameters",
    "cost_parameters",
    "control_parameters",
    "dispatch_parameters",
    "financial_parameters",
)
RESULTS_DIRNAME = "h2i_gui_results"
SUMMARY_FILE = "summary.json"
ARRAYS_FILE = "arrays.npz"
LOG_FILE = "run.log"
KEY_METRIC_PATTERN = re.compile(
    r"(LCO|NPV|CapEx$|OpEx$|total_\w+_produced$|annual_\w+_produced$|capacity_factor$)"
)


# ---------------------------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------------------------


def model_module(name: str) -> str:
    """Return the module path of a registered model without importing it."""
    value = dict.__getitem__(supported_models, name)
    return value.split(":")[0] if isinstance(value, str) else value.__module__


def model_role(name: str) -> str:
    """Classify a registered model by the configuration slot it is normally used in.

    The classification is based on the module location and model name, so that the model
    modules do not need to be imported.
    """
    module = model_module(name).removeprefix("h2integrate.")
    if module.startswith("control.control_rules"):
        return "dispatch_rule_set"
    if module.startswith("control.control_strategies.system_level"):
        return "system_level_control"
    if module.startswith("control."):
        return "control_strategy"
    if module.startswith("resource."):
        return "resource"
    if module.startswith("finances."):
        return "finance"
    if "Cost" in name or "Finance" in name:
        return "cost_model"
    return "performance_model"


def models_by_role() -> dict[str, list[str]]:
    """Group the registered model names by :func:`model_role`."""
    roles = {}
    for name in sorted(supported_models):
        roles.setdefault(model_role(name), []).append(name)
    return roles


# ---------------------------------------------------------------------------------------------
# Loading and saving projects
# ---------------------------------------------------------------------------------------------


def slugify(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_") or "h2i_project"


def new_project(name: str = "my_plant") -> dict:
    """Create a minimal, empty project (a top-level config with embedded sub-configs)."""
    return {
        "name": name,
        "system_summary": "Created with the H2Integrate GUI",
        "driver_config": {
            "name": "driver_config",
            "description": "Single model evaluation",
            "general": {"folder_output": "outputs"},
        },
        "technology_config": {
            "name": "technology_config",
            "description": "Technologies in the plant",
            "technologies": {},
        },
        "plant_config": {
            "name": "plant_config",
            "description": "Plant and site description",
            "sites": {"site": {"latitude": 39.74, "longitude": -105.17}},
            "plant": {"plant_life": 30, "simulation": {"n_timesteps": 8760, "dt": 3600}},
            "technology_interconnections": [],
        },
    }


def find_example_configs(root: Path = EXAMPLE_DIR) -> dict[str, Path]:
    """Find top-level H2Integrate configuration files in the examples folder."""
    configs = {}
    if not Path(root).is_dir():
        return configs
    for path in sorted(Path(root).glob("*/*.yaml")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "technology_config:" in text and "plant_config:" in text:
            configs[f"{path.parent.name}/{path.name}"] = path
    return configs


def load_project(top_level_path: str | Path) -> dict:
    """Load a top-level config and embed the driver, technology, and plant configs as dicts."""
    top_level_path = Path(top_level_path).expanduser().resolve()
    project = load_yaml(top_level_path)
    if not isinstance(project, dict):
        raise ValueError(f"{top_level_path} does not contain a YAML mapping.")
    missing = [key for key in CONFIG_KEYS if key not in project]
    if missing:
        raise ValueError(
            f"{top_level_path} is not a top-level H2Integrate config; missing {missing}."
        )
    for key in CONFIG_KEYS:
        if not isinstance(project[key], dict):
            project[key] = load_yaml(find_file(project[key], top_level_path.parent))
    project.setdefault("name", top_level_path.stem)
    return project


def dump_yaml(data) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=100)


def export_project(project: dict, out_dir: str | Path) -> Path:
    """Write the project as a top-level config plus driver, technology, and plant config files.

    Returns:
        Path: Path to the top-level config file.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    file_names = {
        "driver_config": "driver_config.yaml",
        "technology_config": "tech_config.yaml",
        "plant_config": "plant_config.yaml",
    }
    top_level = {k: v for k, v in project.items() if k not in CONFIG_KEYS}
    for key, file_name in file_names.items():
        (out_dir / file_name).write_text(dump_yaml(project[key]), encoding="utf-8")
        top_level[key] = file_name
    top_level_path = out_dir / f"{slugify(str(project.get('name', 'h2i_project')))}.yaml"
    top_level_path.write_text(dump_yaml(top_level), encoding="utf-8")
    return top_level_path


def validate_project(project: dict) -> list[str]:
    """Validate the embedded configs against the H2Integrate schemas.

    Returns:
        list[str]: Error messages; empty if all configs are valid.
    """
    from h2integrate.core.inputs.validation import load_tech_yaml, load_plant_yaml, load_driver_yaml

    validators = {
        "driver_config": load_driver_yaml,
        "technology_config": load_tech_yaml,
        "plant_config": load_plant_yaml,
    }
    errors = []
    for key, validator in validators.items():
        try:
            validator(copy.deepcopy(project[key]))
        except Exception as e:  # noqa: BLE001 - report any validation failure to the user
            errors.append(f"{key}: {e}")
    return errors


# ---------------------------------------------------------------------------------------------
# Table conversions used by the editors
# ---------------------------------------------------------------------------------------------


def _clean(value) -> str | None:
    """Normalize a table cell to a stripped string, or ``None`` if it is empty."""
    if value is None or (not isinstance(value, list | dict) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


def to_yaml_str(value) -> str:
    """Format a value as single-line YAML for display in a table cell."""
    if value is None:
        return ""
    text = yaml.safe_dump(value, default_flow_style=True, width=10**9, sort_keys=False).strip()
    return text.removesuffix("...").strip()


def from_yaml_str(text) -> object:
    """Parse a table cell written as YAML; empty cells become ``None``."""
    text = _clean(text)
    return None if text is None else yaml.safe_load(text)


def technologies_to_rows(tech_config: dict) -> list[dict]:
    rows = []
    for tech, info in (tech_config.get("technologies") or {}).items():
        row = {"technology": tech}
        for slot in MODEL_SLOTS:
            row[slot] = ((info or {}).get(slot) or {}).get("model")
        rows.append(row)
    return rows


def rows_to_technologies(rows: list[dict], old_technologies: dict) -> tuple[dict, list[str]]:
    """Rebuild the ``technologies`` mapping from the composition table.

    Existing ``model_inputs`` (and other keys) are kept for technologies whose name is unchanged.

    Returns:
        tuple[dict, list[str]]: The new technologies mapping, and the names of technologies
        that are new or have a changed model selection.
    """
    technologies = {}
    changed = []
    for row in rows:
        name = _clean(row.get("technology"))
        if name is None:
            continue
        if name in technologies:
            raise ValueError(f"Technology '{name}' is listed more than once.")
        old_info = old_technologies.get(name) or {}
        info = copy.deepcopy(old_info)
        for slot in MODEL_SLOTS:
            model = _clean(row.get(slot))
            if model is None:
                info.pop(slot, None)
                continue
            block = info.get(slot) or {}
            if block.get("model") != model:
                block = {"model": model}
            info[slot] = block
        if "performance_model" not in info:
            raise ValueError(f"Technology '{name}' needs a performance model.")
        if info.get("model_inputs") is None:
            info["model_inputs"] = {}
        if any(
            (info.get(slot) or {}).get("model") != (old_info.get(slot) or {}).get("model")
            for slot in MODEL_SLOTS
        ):
            changed.append(name)
        technologies[name] = info
    return technologies, changed


def fill_default_inputs(tech_info: dict) -> list[str]:
    """Add model input parameters (with defaults) that are missing from ``model_inputs``.

    Parameters already present in any section are left untouched.

    Returns:
        list[str]: Names of the parameters that were added.
    """
    template = organize_model_parameters(tech_info)
    model_inputs = tech_info.get("model_inputs") or {}
    tech_info["model_inputs"] = model_inputs
    existing = {
        key for section in model_inputs.values() if isinstance(section, dict) for key in section
    }
    added = []
    for section, params in template.items():
        for key, value in params.items():
            if key not in existing:
                if model_inputs.get(section) is None:
                    model_inputs[section] = {}
                model_inputs[section][key] = value
                added.append(key)
    return added


def model_inputs_to_rows(tech_name: str, tech_info: dict) -> list[dict]:
    comments = _validator_comments({"technologies": {tech_name: tech_info}})
    rows = []
    for section, params in (tech_info.get("model_inputs") or {}).items():
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            rows.append(
                {
                    "section": section,
                    "parameter": key,
                    "value": to_yaml_str(value),
                    "constraint": comments.get((tech_name, section, key), ""),
                }
            )
    return rows


def rows_to_model_inputs(rows: list[dict]) -> dict:
    model_inputs = {}
    for i, row in enumerate(rows, start=1):
        key = _clean(row.get("parameter"))
        if key is None:
            continue
        section = _clean(row.get("section")) or "performance_parameters"
        try:
            value = from_yaml_str(row.get("value"))
        except yaml.YAMLError as e:
            raise ValueError(f"Row {i} ('{key}'): value is not valid YAML.\n{e}") from e
        params = model_inputs.setdefault(section, {})
        if key in params:
            raise ValueError(f"Row {i}: '{key}' appears more than once in {section}.")
        params[key] = value
    return model_inputs


def interconnections_to_rows(plant_config: dict) -> list[dict]:
    return [
        {
            "source": connection[0],
            "destination": connection[1] if len(connection) > 1 else None,
            "connection": to_yaml_str(connection[2]) if len(connection) > 2 else "",
            "transport": connection[3] if len(connection) > 3 else None,
        }
        for connection in plant_config.get("technology_interconnections") or []
    ]


def rows_to_interconnections(rows: list[dict]) -> list[list]:
    connections = []
    for i, row in enumerate(rows, start=1):
        source, destination = _clean(row.get("source")), _clean(row.get("destination"))
        if source is None and destination is None:
            continue
        if source is None or destination is None:
            raise ValueError(f"Connection {i} needs both a source and a destination.")
        try:
            connected = from_yaml_str(row.get("connection"))
        except yaml.YAMLError as e:
            raise ValueError(f"Connection {i}: connection is not valid YAML.\n{e}") from e
        if connected is None:
            raise ValueError(f"Connection {i} needs a commodity or [source_output, dest_input].")
        connection = [source, destination, connected]
        transport = _clean(row.get("transport"))
        if transport is not None:
            connection.append(transport)
        connections.append(connection)
    return connections


def _dot_id(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def system_diagram(project: dict) -> str:
    """Build a Graphviz DOT description of the technologies and their connections."""
    plant = project["plant_config"]
    lines = [
        "digraph {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fillcolor="#e8f0fe", fontname="Helvetica"];',
        'edge [fontname="Helvetica", fontsize=10];',
    ]
    lines.extend(
        f"{_dot_id(tech)};" for tech in project["technology_config"].get("technologies") or {}
    )
    for source, dest, *_ in plant.get("site_to_tech_connections") or []:
        lines.append(f'{_dot_id(source)} [shape=note, fillcolor="#fef7e0"];')
        lines.append(f'{_dot_id(source)} -> {_dot_id(dest)} [style=dashed, color="#999999"];')
    for connection in plant.get("technology_interconnections") or []:
        source, dest = connection[0], connection[1]
        label = connection[2] if len(connection) > 2 else ""
        if isinstance(label, list):
            label = " → ".join(str(item) for item in label)
        if len(connection) > 3:
            label = f"{label} ({connection[3]})"
        lines.append(f"{_dot_id(source)} -> {_dot_id(dest)} [label={_dot_id(label)}];")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Running and results
# ---------------------------------------------------------------------------------------------


def results_dir(workdir: str | Path) -> Path:
    return Path(workdir).expanduser() / RESULTS_DIRNAME


def start_run(project: dict, workdir: str | Path) -> tuple[subprocess.Popen, Path]:
    """Write the project config and launch the model in a separate Python process.

    The model runs with ``workdir`` as its working directory so that relative paths in the
    configuration (e.g. custom models or resource files) resolve as they would from the
    original config folder. Outputs are written to ``<workdir>/h2i_gui_results``.

    Returns:
        tuple[subprocess.Popen, Path]: The running process (stdout and stderr combined), and the
        path of the log file to write its output to.
    """
    workdir = Path(workdir).expanduser().resolve()
    out_dir = results_dir(workdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (SUMMARY_FILE, ARRAYS_FILE):
        (out_dir / file_name).unlink(missing_ok=True)
    config_path = out_dir / "config.yaml"
    config_path.write_text(dump_yaml(project), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "h2integrate.gui.runner",
        str(config_path),
        "--workdir",
        str(workdir),
        "--results-dir",
        str(out_dir),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return process, out_dir / LOG_FILE


def collect_outputs(prob) -> tuple[dict, dict, dict]:
    """Collect the numeric outputs of a run OpenMDAO problem.

    Returns:
        tuple[dict, dict, dict]: Scalars as ``{name: {"value", "units"}}``, 1-D arrays as
        ``{name: ndarray}``, and array units as ``{name: units}``.
    """
    scalars, arrays, array_units = {}, {}, {}
    outputs = prob.model.list_outputs(
        val=True, units=True, prom_name=True, out_stream=None, excludes=["*resource_data*"]
    )
    for abs_name, meta in outputs:
        try:
            value = np.asarray(meta["val"])
        except (TypeError, ValueError):
            continue
        if value.dtype.kind not in "biuf" or value.size == 0:
            continue
        name = meta.get("prom_name") or abs_name
        units = meta.get("units")
        if value.size == 1:
            scalars[name] = {"value": float(value.ravel()[0]), "units": units}
        elif value.squeeze().ndim == 1:
            arrays[name] = value.squeeze().astype(float)
            array_units[name] = units
    return scalars, arrays, array_units


def save_results(out_dir: Path, scalars: dict, arrays: dict, array_units: dict, n_timesteps):
    out_dir = Path(out_dir)
    np.savez_compressed(out_dir / ARRAYS_FILE, **arrays)
    summary = {"n_timesteps": n_timesteps, "scalars": scalars, "array_units": array_units}
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")


def load_results(out_dir: str | Path) -> dict | None:
    """Load results saved by :func:`save_results`, or return ``None`` if there are none."""
    out_dir = Path(out_dir)
    summary_path = out_dir / SUMMARY_FILE
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    arrays = {}
    arrays_path = out_dir / ARRAYS_FILE
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    summary["arrays"] = arrays
    return summary


def scalars_table(scalars: dict, key_metrics_only: bool = False) -> pd.DataFrame:
    rows = [
        {"output": name, "value": item["value"], "units": item["units"] or ""}
        for name, item in scalars.items()
        if not key_metrics_only or KEY_METRIC_PATTERN.search(name.split(".")[-1])
    ]
    return pd.DataFrame(rows, columns=["output", "value", "units"])

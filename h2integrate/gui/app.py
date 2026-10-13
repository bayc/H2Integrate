"""H2Integrate GUI (Streamlit app). Launch with ``python -m h2integrate.gui``."""

import time
from pathlib import Path

import yaml
import pandas as pd
import streamlit as st

from h2integrate import __version__
from h2integrate.gui import backend as be


st.set_page_config(page_title="H2Integrate", layout="wide")
ss = st.session_state


# ---------------------------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------------------------


def default_workdir(project):
    return str(Path.cwd() / "h2i_gui_runs" / be.slugify(str(project.get("name", ""))))


if "project" not in ss:
    ss.project = be.new_project()
    ss.source = None
    ss.workdir = default_workdir(ss.project)
    ss.version = 0
    ss.results = None
    ss.run_log = ""
    ss.flash = None


def commit(message=None, kind="success"):
    """Record a project change, refresh all editors, and optionally show a message."""
    ss.version += 1
    if message:
        ss.flash = (kind, message)
    st.rerun()


def set_project(project, source_dir=None):
    ss.project = project
    ss.source = str(source_dir) if source_dir else None
    ss.workdir = str(source_dir) if source_dir else default_workdir(project)
    ss.results = be.load_results(be.results_dir(ss.workdir))
    ss.run_log = ""
    commit(f"Loaded '{project.get('name')}'.")


def load_from_path(path):
    path = Path(path).expanduser().resolve()
    try:
        project = be.load_project(path)
    except Exception as e:  # noqa: BLE001 - show any load failure in the UI
        st.error(f"Could not load {path}: {e}")
        return
    set_project(project, path.parent)


@st.cache_data(show_spinner=False)
def example_configs():
    return {name: str(path) for name, path in be.find_example_configs().items()}


@st.cache_resource(show_spinner=False)
def model_roles():
    return be.models_by_role()


def model_options(role, current_values):
    options = set(model_roles().get(role, []))
    options.update(value for value in current_values if value)
    return sorted(options)


def technologies():
    tech_config = ss.project["technology_config"]
    if tech_config.get("technologies") is None:
        tech_config["technologies"] = {}
    return tech_config["technologies"]


# ---------------------------------------------------------------------------------------------
# Sidebar: project selection
# ---------------------------------------------------------------------------------------------

with st.sidebar:
    st.title("H2Integrate")
    st.caption(f"Version {__version__}")
    start_from = st.radio("Start from", ["Example", "Config file", "New"], horizontal=True)
    if start_from == "Example":
        examples = example_configs()
        choice = st.selectbox("Example", list(examples))
        if st.button("Load example", disabled=not examples):
            load_from_path(examples[choice])
    elif start_from == "Config file":
        path = st.text_input("Top-level config YAML", placeholder="/path/to/my_plant.yaml")
        if st.button("Load file", disabled=not path):
            load_from_path(path)
    else:
        name = st.text_input("Project name", "my_plant")
        if st.button("Create project"):
            set_project(be.new_project(be.slugify(name)))

    st.divider()
    st.markdown(f"**Project:** {ss.project.get('name')}")
    st.caption(f"Source folder: {ss.source or 'none (new project)'}")
    st.caption(f"Technologies: {', '.join(technologies()) or 'none'}")

st.title(ss.project.get("name") or "H2Integrate")
if ss.project.get("system_summary"):
    st.caption(ss.project["system_summary"])

if ss.flash:
    kind, message = ss.flash
    getattr(st, kind)(message)
    ss.flash = None

tab_compose, tab_inputs, tab_run, tab_results, tab_yaml = st.tabs(
    ["Compose", "Inputs", "Run", "Results", "YAML"]
)
version = ss.version
plant = ss.project["plant_config"]


# ---------------------------------------------------------------------------------------------
# Compose: technologies, connections, and plant settings
# ---------------------------------------------------------------------------------------------

with tab_compose:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Technologies")
        rows = be.technologies_to_rows(ss.project["technology_config"])
        current = {slot: [row[slot] for row in rows] for slot in be.MODEL_SLOTS}
        with st.form(f"technologies_{version}"):
            edited = st.data_editor(
                pd.DataFrame(rows, columns=["technology", *be.MODEL_SLOTS]),
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "technology": st.column_config.TextColumn(
                        "Technology", required=True, help="Unique name, e.g. wind"
                    ),
                    **{
                        slot: st.column_config.SelectboxColumn(
                            slot.replace("_", " ").capitalize(),
                            options=model_options(slot, current[slot]),
                            required=slot == "performance_model",
                        )
                        for slot in be.MODEL_SLOTS
                    },
                },
            )
            fill_defaults = st.checkbox("Fill default inputs for new or changed models", True)
            submitted = st.form_submit_button("Apply technologies", type="primary")
        if submitted:
            try:
                new_techs, changed = be.rows_to_technologies(
                    edited.to_dict("records"), technologies()
                )
            except ValueError as e:
                st.error(str(e))
            else:
                problems = []
                if fill_defaults:
                    for tech in changed:
                        try:
                            be.fill_default_inputs(new_techs[tech])
                        except Exception as e:  # noqa: BLE001 - some models lack templates
                            problems.append(f"{tech}: {e}")
                ss.project["technology_config"]["technologies"] = new_techs
                if problems:
                    commit("Could not read default inputs for " + "; ".join(problems), "warning")
                commit("Technologies updated.")

        st.subheader("Connections")
        connection_rows = be.interconnections_to_rows(plant)
        endpoints = set(technologies())
        endpoints.update(row[key] for row in connection_rows for key in ("source", "destination"))
        endpoints = sorted(endpoint for endpoint in endpoints if endpoint)
        with st.form(f"connections_{version}"):
            edited = st.data_editor(
                pd.DataFrame(
                    connection_rows, columns=["source", "destination", "connection", "transport"]
                ),
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "source": st.column_config.SelectboxColumn(
                        "From", options=endpoints, required=True
                    ),
                    "destination": st.column_config.SelectboxColumn(
                        "To", options=endpoints, required=True
                    ),
                    "connection": st.column_config.TextColumn(
                        "Commodity or [output, input]",
                        help="A commodity such as `electricity`, or `[heat_out, heat_in]` "
                        "to connect specific variables.",
                    ),
                    "transport": st.column_config.TextColumn(
                        "Transport", help="Optional transport model, e.g. `cable` or `pipe`."
                    ),
                },
            )
            if st.form_submit_button("Apply connections", type="primary"):
                try:
                    plant["technology_interconnections"] = be.rows_to_interconnections(
                        edited.to_dict("records")
                    )
                except ValueError as e:
                    st.error(str(e))
                else:
                    commit("Connections updated.")

        st.subheader("Plant and site")
        plant_settings = plant.setdefault("plant", {})
        simulation = plant_settings.setdefault("simulation", {})
        with st.form(f"plant_{version}"):
            col1, col2, col3 = st.columns(3)
            plant_life = col1.number_input(
                "Plant life (years)", min_value=1, value=int(plant_settings.get("plant_life", 30))
            )
            n_timesteps = col2.number_input(
                "Timesteps", min_value=1, value=int(simulation.get("n_timesteps", 8760))
            )
            dt = col3.number_input(
                "Timestep length (s)", min_value=1, value=int(simulation.get("dt", 3600))
            )
            locations = {}
            for site_name, site in (plant.get("sites") or {}).items():
                col1, col2 = st.columns(2)
                locations[site_name] = (
                    col1.number_input(
                        f"{site_name}: latitude",
                        min_value=-90.0,
                        max_value=90.0,
                        value=float(site.get("latitude", 0.0)),
                        format="%.5f",
                    ),
                    col2.number_input(
                        f"{site_name}: longitude",
                        min_value=-180.0,
                        max_value=180.0,
                        value=float(site.get("longitude", 0.0)),
                        format="%.5f",
                    ),
                )
            st.caption("Resources and finance settings can be edited in the YAML tab.")
            if st.form_submit_button("Apply plant settings", type="primary"):
                plant_settings["plant_life"] = int(plant_life)
                simulation["n_timesteps"] = int(n_timesteps)
                simulation["dt"] = int(dt)
                for site_name, (latitude, longitude) in locations.items():
                    plant["sites"][site_name]["latitude"] = latitude
                    plant["sites"][site_name]["longitude"] = longitude
                commit("Plant settings updated.")

    with right:
        st.subheader("System diagram")
        if technologies():
            st.graphviz_chart(be.system_diagram(ss.project), width="stretch")
        else:
            st.info("Add technologies to see the system diagram.")


# ---------------------------------------------------------------------------------------------
# Inputs: model parameters for each technology
# ---------------------------------------------------------------------------------------------

with tab_inputs:
    if not technologies():
        st.info("Add technologies in the Compose tab first.")
    else:
        tech = st.selectbox("Technology", list(technologies()))
        tech_info = technologies()[tech]
        st.caption(
            "  |  ".join(
                f"{slot.replace('_', ' ')}: `{tech_info[slot]['model']}`"
                for slot in be.MODEL_SLOTS
                if slot in tech_info
            )
        )
        if st.button("Fill missing parameters from model defaults"):
            try:
                added = be.fill_default_inputs(tech_info)
            except Exception as e:  # noqa: BLE001 - some models lack templates
                st.error(f"Could not read model defaults: {e}")
            else:
                if added:
                    commit(f"Added {len(added)} parameters to '{tech}': {', '.join(added)}.")
                commit("No missing parameters found.", "info")

        rows = be.model_inputs_to_rows(tech, tech_info)
        sections = sorted(set(be.PARAM_SECTIONS) | {row["section"] for row in rows})
        with st.form(f"inputs_{tech}_{version}"):
            edited = st.data_editor(
                pd.DataFrame(rows, columns=["section", "parameter", "value", "constraint"]),
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "section": st.column_config.SelectboxColumn(
                        "Section",
                        options=sections,
                        required=True,
                        default="performance_parameters",
                    ),
                    "parameter": st.column_config.TextColumn("Parameter", required=True),
                    "value": st.column_config.TextColumn("Value (YAML)", width="large"),
                    "constraint": st.column_config.TextColumn("Constraint", disabled=True),
                },
            )
            st.caption(
                "Values are YAML: `1.5`, `true`, `some text`, `[1, 2]`, `{key: value}`. "
                "Empty values are null; required inputs without defaults are left empty."
            )
            if st.form_submit_button("Apply inputs", type="primary"):
                try:
                    tech_info["model_inputs"] = be.rows_to_model_inputs(edited.to_dict("records"))
                except ValueError as e:
                    st.error(str(e))
                else:
                    commit(f"Inputs for '{tech}' updated.")


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------


def run_model(workdir, log_box):
    try:
        process, log_path = be.start_run(ss.project, workdir)
    except OSError as e:
        st.error(f"Could not start the run: {e}")
        return
    lines = []
    last_update = 0.0
    try:
        with st.spinner("Running H2Integrate..."), log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                lines.append(line)
                if time.monotonic() - last_update > 0.5:
                    log_box.code("".join(lines[-200:]), language="text")
                    last_update = time.monotonic()
            process.wait()
    finally:
        # Interacting with the app interrupts this script; don't leave the run orphaned.
        if process.poll() is None:
            process.terminate()
    ss.run_log = "".join(lines)
    log_box.code("".join(lines[-500:]), language="text")
    if process.returncode == 0:
        ss.results = be.load_results(be.results_dir(workdir))
        st.success("Run finished. See the Results tab.")
    else:
        ss.results = None
        st.error(f"Run failed (exit code {process.returncode}). Full log: {log_path}")


with tab_run:
    ss.workdir = st.text_input(
        "Working directory",
        value=ss.workdir,
        key=f"workdir_{version}",
        help="The model runs from this folder, so relative paths in the configuration resolve "
        f"here. Results are written to `<folder>/{be.RESULTS_DIRNAME}`.",
    )
    col1, col2, _ = st.columns([1, 1, 3])
    check = col1.button("Check configuration", width="stretch")
    run = col2.button(
        "Run simulation", type="primary", disabled=not technologies(), width="stretch"
    )
    st.caption("Interacting with the app while a simulation is running cancels the run.")
    if check:
        errors = be.validate_project(ss.project)
        for error in errors:
            st.error(error)
        if not errors:
            st.success("The configuration passes schema validation.")
    log_box = st.empty()
    if run:
        run_model(ss.workdir, log_box)
    elif ss.run_log:
        log_box.code(ss.run_log[-20000:], language="text")

    st.subheader("Export")
    export_dir = st.text_input(
        "Export folder", value=str(Path(ss.workdir) / "gui_config"), key=f"export_{version}"
    )
    if st.button("Export YAML files"):
        try:
            top_level_path = be.export_project(ss.project, export_dir)
        except OSError as e:
            st.error(f"Could not export: {e}")
        else:
            st.success(f"Wrote {top_level_path} and its driver, tech, and plant configs.")


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

with tab_results:
    results = ss.results
    if not results:
        st.info("No results yet. Run the simulation from the Run tab.")
    else:
        scalars = results["scalars"]
        key_metrics = be.scalars_table(scalars, key_metrics_only=True)
        headline = key_metrics[key_metrics["output"].str.contains(r"(?:^|\.)(?:LCO|NPV)")]
        if not headline.empty:
            columns = st.columns(4)
            for i, row in enumerate(headline.head(8).itertuples()):
                columns[i % 4].metric(
                    row.output.removeprefix("finance_subgroup_").replace(".", " "),
                    f"{row.value:.4g} {row.units}",
                    help=row.output,
                )
        st.subheader("Key metrics")
        st.dataframe(key_metrics, hide_index=True)
        with st.expander(f"All scalar outputs ({len(scalars)})"):
            query = st.text_input("Filter by name")
            all_scalars = be.scalars_table(scalars)
            if query:
                all_scalars = all_scalars[
                    all_scalars["output"].str.contains(query, case=False, regex=False)
                ]
            st.dataframe(all_scalars, hide_index=True)

        arrays, units = results["arrays"], results["array_units"]
        n_timesteps = int(results["n_timesteps"])
        timeseries = sorted(name for name, value in arrays.items() if len(value) == n_timesteps)
        st.subheader("Time series")
        if timeseries:
            selected = st.multiselect(
                "Outputs",
                timeseries,
                default=[name for name in timeseries if name.endswith("_out")][:3],
            )
            if selected:
                start, end = st.slider("Timestep range", 0, n_timesteps, (0, min(n_timesteps, 168)))
                labels = {name: f"{name} [{units.get(name) or '-'}]" for name in selected}
                data = pd.DataFrame({labels[name]: arrays[name] for name in selected})
                data.index.name = "timestep"
                st.line_chart(data.iloc[start:end])
                st.download_button(
                    "Download selected time series (CSV)",
                    data.to_csv(),
                    file_name="timeseries.csv",
                    mime="text/csv",
                )
        else:
            st.info("No time series outputs found.")

        other_arrays = sorted(name for name, value in arrays.items() if len(value) != n_timesteps)
        if other_arrays:
            st.subheader("Other array outputs")
            name = st.selectbox("Output", other_arrays)
            st.bar_chart(pd.Series(arrays[name], name=f"{name} [{units.get(name) or '-'}]"))


# ---------------------------------------------------------------------------------------------
# YAML: direct editing of the full configuration
# ---------------------------------------------------------------------------------------------

with tab_yaml:
    st.caption(
        "Edit the full configuration directly, e.g. for resources, finance, or driver settings. "
        "`!include` files from loaded configs are shown expanded."
    )
    with st.form(f"yaml_{version}"):
        texts = {
            key: st.text_area(
                key.replace("_", " ").capitalize(), be.dump_yaml(ss.project[key]), height=400
            )
            for key in be.CONFIG_KEYS
        }
        if st.form_submit_button("Apply YAML", type="primary"):
            try:
                parsed = {key: yaml.safe_load(text) for key, text in texts.items()}
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")
            else:
                invalid = [key for key, value in parsed.items() if not isinstance(value, dict)]
                if invalid:
                    st.error(f"These configs must be YAML mappings: {', '.join(invalid)}")
                else:
                    ss.project.update(parsed)
                    commit("Configuration updated from YAML.")

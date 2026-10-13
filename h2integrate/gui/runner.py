"""Run an H2Integrate model for the GUI and save its numeric outputs.

Usage:
    python -m h2integrate.gui.runner <config.yaml> --workdir <dir> --results-dir <dir>

``config.yaml`` is a top-level config with the driver, technology, and plant configs embedded.
"""

import os
import argparse
from pathlib import Path

import yaml

from h2integrate.gui.backend import save_results, collect_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    os.chdir(args.workdir)

    from h2integrate.core.h2integrate_model import H2IntegrateModel

    print(f"Building model '{config.get('name')}' in {args.workdir}", flush=True)
    model = H2IntegrateModel(config)
    print("Running model...", flush=True)
    model.run()

    scalars, arrays, array_units = collect_outputs(model.prob)
    n_timesteps = model.plant_config["plant"]["simulation"]["n_timesteps"]
    save_results(args.results_dir, scalars, arrays, array_units, n_timesteps)
    print(
        f"Done. Saved {len(scalars)} scalar and {len(arrays)} array outputs to {args.results_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

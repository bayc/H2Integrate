from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "10mw_80per_compute_load_training.csv"
INFERENCE_PATH = BASE_DIR / "1mw_80per_compute_load_inference.csv"


def load_weekly_profile(csv_path: Path, scale_factor: float = 1.0) -> pd.Series:
    """Return the first full week of data in megawatts, optionally scaled."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep the first seven days only for a comparable weekly view.
    week_df = df.iloc[: 7 * 24 * 60].copy()
    power_mw = (week_df["power_W"] * scale_factor) / 1e6
    power_mw.index = week_df["timestamp"]
    return power_mw


def plot_raw_series(ax: plt.Axes, series: pd.Series, color: str, label: str) -> None:
    ax.plot(series.index, series.values, color=color, linewidth=1.5, alpha=0.9, label=label)


if __name__ == "__main__":
    train_week = load_weekly_profile(TRAIN_PATH, scale_factor=10.0)
    inference_week = load_weekly_profile(INFERENCE_PATH, scale_factor=100.0)

    shared_max = max(train_week.max(), inference_week.max()) * 1.1
    shared_min = 0.0

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_ylim(shared_min, shared_max)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.3)
    ax.set_title("Weekly variance in data-center load profiles")

    plot_raw_series(ax, train_week, "tab:blue", "Training")
    plot_raw_series(ax, inference_week, "tab:orange", "Inference")

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "weekly_variance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

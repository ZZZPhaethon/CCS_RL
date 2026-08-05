"""Plot the three baseline annual emitter capture profiles.

The plotted rates come directly from the calibrated 8,760-hour source CSV.
No sampled availability noise, outages, high-output windows, or utilization
changes are applied.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "capture_rates"
    / "phase1plus_emitters_capture_rate_profile_hourly.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output"
    / "emitter_profiles"
    / "three_emitters_annual_capture_no_disturbance"
)

SERIES = (
    ("brevik_capture_tph", "Brevik", "#0F4D92"),
    ("celsio_capture_tph", "Celsio", "#42949E"),
    ("yara_sluiskil_capture_tph", "Yara Sluiskil", "#9A4D8E"),
)


def load_profiles(path: Path) -> tuple[list[datetime], dict[str, list[float]]]:
    timestamps: list[datetime] = []
    values = {column: [] for column, _label, _color in SERIES}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamps.append(datetime.fromisoformat(row["timestamp"]))
            for column in values:
                values[column].append(float(row[column]))

    if len(timestamps) != 8760:
        raise ValueError(f"Expected 8,760 hourly rows, found {len(timestamps):,}.")
    if any(len(series) != len(timestamps) for series in values.values()):
        raise ValueError("Emitter series lengths do not match the timestamp series.")
    return timestamps, values


def plot_profiles(
    timestamps: list[datetime],
    values: dict[str, list[float]],
    output_base: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )

    fig, ax = plt.subplots(figsize=(183 / 25.4, 92 / 25.4))
    for column, label, color in SERIES:
        annual_total = sum(values[column])
        ax.plot(
            timestamps,
            values[column],
            color=color,
            linewidth=1.35,
            label=f"{label} ({annual_total / 1e6:.2f} Mt y$^{{-1}}$)",
        )

    ax.set_title(
        "Annual emitter CO$_2$ capture profiles without additional disturbances",
        loc="left",
        pad=8,
        fontweight="bold",
    )
    ax.set_ylabel("Capture rate (t CO$_2$ h$^{-1}$)")
    ax.set_xlabel("Month of reference year (2026)")
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", handlelength=2.5)
    ax.text(
        0.0,
        -0.27,
        "Baseline hourly profiles only: availability = 1, utilization = 1; "
        "no sampled noise, outages, or high-output multipliers.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#4D4D4D",
        fontsize=6.5,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.88, bottom=0.27)
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    timestamps, values = load_profiles(args.input)
    plot_profiles(timestamps, values, args.output)
    for column, label, _color in SERIES:
        print(
            f"{label}: total={sum(values[column]):.6f} t, "
            f"min={min(values[column]):.6f} t/h, "
            f"max={max(values[column]):.6f} t/h"
        )


if __name__ == "__main__":
    main()

from pathlib import Path

from sim.visualization import write_e1_cinematic_dashboard


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_CSV = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "figures"
    / "source_data"
    / "figure_4_hourly_trace.csv"
)
OUTPUT_HTML = (
    REPO_ROOT
    / "docs"
    / "physical_layer_dashboardV2.html"
)


if __name__ == "__main__":
    output = write_e1_cinematic_dashboard(
        OUTPUT_HTML,
        TRACE_CSV,
    )
    print(output)

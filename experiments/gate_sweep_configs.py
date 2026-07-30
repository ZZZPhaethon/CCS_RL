"""Locked validation-only gate configurations for the E2 follow-up."""

from __future__ import annotations


VALIDATION_SEEDS = tuple(range(8100001, 8100021))
FORMAL_TEST_SEEDS = tuple(range(9000031, 9000061))


def partition_windows(width_h: int) -> tuple[tuple[int, int], ...]:
    if width_h <= 0:
        raise ValueError("window width must be positive")
    windows = []
    start = 108
    while start <= 680:
        end = min(start + int(width_h) - 1, 680)
        windows.append((start, end))
        start = end + 1
    return tuple(windows)


WINDOW_SCHEMES = {
    "global": None,
    "w12": partition_windows(48),
    "w24": partition_windows(24),
    "w48": partition_windows(12),
}


def _margin_tag(margin: float) -> str:
    return f"{int(round(float(margin) * 100)):03d}"


def gate_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    signatures: set[tuple[int, float, int, str]] = set()

    def add(
        required_heads: int,
        margin: float,
        max_overrides: int,
        window_scheme: str,
    ) -> None:
        signature = (
            int(required_heads),
            float(margin),
            int(max_overrides),
            str(window_scheme),
        )
        if signature in signatures:
            return
        signatures.add(signature)
        records.append(
            {
                "name": (
                    f"h{required_heads}_m{_margin_tag(margin)}_"
                    f"{window_scheme}_c{max_overrides}"
                ),
                "required_heads": int(required_heads),
                "margin": float(margin),
                "max_overrides": int(max_overrides),
                "window_scheme": str(window_scheme),
                "windows": WINDOW_SCHEMES[window_scheme],
            }
        )

    # Confidence sweep under the frozen 12-window/12-intervention structure.
    for required_heads in (2, 3, 4, 5):
        for margin in (0.0, 0.1, 0.2, 0.4):
            add(required_heads, margin, 12, "w12")

    # Capacity sweep at a deliberately relaxed confidence gate.
    for max_overrides in (6, 12, 24, 48):
        add(3, 0.1, max_overrides, "global")
    for max_overrides in (6, 12):
        add(3, 0.1, max_overrides, "w12")
    for max_overrides in (12, 24):
        add(3, 0.1, max_overrides, "w24")
    for max_overrides in (24, 48):
        add(3, 0.1, max_overrides, "w48")

    # Check whether expanded capacity interacts with aggressive/conservative
    # confidence settings without evaluating the full Cartesian product.
    for required_heads, margin in ((2, 0.0), (4, 0.4)):
        add(required_heads, margin, 48, "global")
        add(required_heads, margin, 24, "w24")
        add(required_heads, margin, 48, "w48")

    return records


def gate_cli_values() -> list[str]:
    values = []
    for gate in gate_records():
        prefix = (
            f"{gate['name']}:{gate['required_heads']}:{gate['margin']}:"
            f"{gate['max_overrides']}"
        )
        windows = gate["windows"]
        if windows is None:
            values.append(prefix)
        else:
            window_text = ",".join(
                f"{start}-{end}" for start, end in windows
            )
            values.append(f"{prefix}:{window_text}")
    return values

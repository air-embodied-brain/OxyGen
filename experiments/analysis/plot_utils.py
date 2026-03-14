"""Shared plotting utilities for experiment analysis scripts."""

from pathlib import Path

import matplotlib.pyplot as plt

ACTION_HORIZON = 10


class PlotColors:
    """Unified color palette matching design figures."""
    # Ours (CB) - Blue family
    OURS_PRIMARY = "#4472C4"
    OURS_LIGHT = "#A8C4E0"
    OURS_DARK = "#2B4C7E"

    # Baseline - Red family
    BASELINE_PRIMARY = "#C44E52"
    BASELINE_LIGHT = "#E8A5A8"
    BASELINE_DARK = "#8B3236"

    # MPS (Parallel) - Similar to baseline
    MPS_PRIMARY = "#DD8452"
    MPS_LIGHT = "#F2C9A3"
    MPS_DARK = "#C44E00"

    # Variants
    VARIANT_GOLD = "#F0C75E"  # yellow/gold - between ours and baseline
    VARIANT_GREEN = "#70AD47"  # design green

    # Utility
    NEUTRAL_DARK = "#404040"
    BAR_EDGE = "#2C2C2C"


class BarStyle:
    """Bar styling constants."""
    EDGE_COLOR = PlotColors.BAR_EDGE
    EDGE_WIDTH = 0.8
    # Hatch patterns for print-friendly plots
    HATCH_OURS = ""
    HATCH_BASELINE = "///"
    HATCH_MPS = "\\\\\\"
    HATCH_VARIANT = "xxx"


def round_dual_axis_ranges(y1_data_max: float, y2_data_max: float,
                          max_ratio: float = 1.5,
                          buffer: float = 1.05) -> tuple[float, float, int]:
    """Co-optimize y-axis ranges for dual-axis plots with shared tick count.

    Args:
        y1_data_max: Actual max data value for primary axis (e.g., latency)
        y2_data_max: Actual max data value for secondary axis (e.g., throughput or batch size)
        max_ratio: Maximum allowed ratio between rounded range and data max (default: 1.5)
        buffer: Initial buffer multiplier for data max (default: 1.05)

    Returns:
        (y1_nice, y2_nice, n_ticks): Rounded ranges and shared tick count
    """
    import numpy as np

    if y1_data_max <= 0 or y2_data_max <= 0:
        return max(y1_data_max, 10), max(y2_data_max, 10), 6

    # Apply buffer to get initial target ranges
    y1_max = y1_data_max * buffer
    y2_max = y2_data_max * buffer

    # Determine if each axis is small range (needs integer ticks)
    y1_is_small = y1_data_max < 30
    y2_is_small = y2_data_max < 30

    # Try different max_ratio values if needed
    for ratio in [max_ratio, max_ratio * 1.2, max_ratio * 1.5, max_ratio * 2.0]:
        # Try different tick counts from 5 down to 3
        for n_ticks in [6, 5, 4, 3]:
            y1_nice = _round_to_nice_value(y1_max, y1_data_max, small_range_threshold=30)
            y2_nice = _round_to_nice_value(y2_max, y2_data_max, small_range_threshold=30)

            # Ensure ranges are larger than input max values
            if y1_nice < y1_max:
                y1_nice = _round_to_nice_value(y1_max * 1.01, y1_data_max, small_range_threshold=30)
            if y2_nice < y2_max:
                y2_nice = _round_to_nice_value(y2_max * 1.01, y2_data_max, small_range_threshold=30)

            # For small ranges, ensure ticks are integers by adjusting range to be divisible by (n_ticks - 1)
            if y1_is_small:
                tick_interval = y1_nice / (n_ticks - 1)
                if tick_interval != int(tick_interval):
                    # Round up to make interval an integer
                    y1_nice = float(np.ceil(tick_interval) * (n_ticks - 1))

            if y2_is_small:
                tick_interval = y2_nice / (n_ticks - 1)
                if tick_interval != int(tick_interval):
                    # Round up to make interval an integer
                    y2_nice = float(np.ceil(tick_interval) * (n_ticks - 1))

            # Check if both ranges are reasonable
            y1_ok = y1_nice <= y1_data_max * ratio and y1_nice >= y1_max
            y2_ok = y2_nice <= y2_data_max * ratio and y2_nice >= y2_max

            if y1_ok and y2_ok:
                return y1_nice, y2_nice, n_ticks

    # Fallback: use last computed values with 3 ticks
    return y1_nice, y2_nice, 3


def _round_to_nice_value(val: float, data_max: float, small_range_threshold: float = 30) -> float:
    """Round a single value to a nice tick value.

    Args:
        val: Value to round
        data_max: Actual max data value
        small_range_threshold: If data_max is below this, use integer rounding instead of magnitude-based

    Returns:
        Rounded nice value
    """
    import numpy as np

    if val <= 0:
        return 10

    # For small ranges (e.g., batch size < 30), just round to nearest integer
    # Note: This only ensures the range is integer; tick spacing is handled in round_dual_axis_ranges
    if data_max < small_range_threshold:
        return float(np.ceil(val))

    # For larger ranges, use magnitude-based rounding with more options for easier matching
    # Customize the rounding options below as needed:
    # More options = easier to find matching ranges for dual axes
    # Fewer options = "nicer" looking tick values
    magnitude = 10 ** np.floor(np.log10(val))
    normalized = val / magnitude

    # Extended options: 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10
    if normalized <= 1:
        nice = 1
    elif normalized <= 1.5:
        nice = 1.5
    elif normalized <= 2:
        nice = 2
    elif normalized <= 2.5:
        nice = 2.5
    elif normalized <= 3:
        nice = 3
    elif normalized <= 4:
        nice = 4
    elif normalized <= 5:
        nice = 5
    elif normalized <= 6:
        nice = 6
    elif normalized <= 7:
        nice = 7
    elif normalized <= 8:
        nice = 8
    elif normalized <= 9:
        nice = 9
    else:
        nice = 10

    return nice * magnitude


def setup_style():
    """Set matplotlib rcParams for ECCV academic style."""
    # Register user-local fonts (e.g. ~/.local/share/fonts)
    import matplotlib.font_manager as fm
    user_font_dir = Path.home() / ".local" / "share" / "fonts"
    if user_font_dir.is_dir():
        for f in user_font_dir.glob("*.ttf"):
            fm.fontManager.addfont(str(f))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif"],
        "font.sans-serif": ["Arial", "Liberation Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

"""Host defaults for optional training report plots."""

from pathlib import Path


def is_jetson() -> bool:
    """Identify Jetson without treating every ARM workstation as a Jetson."""
    try:
        if Path("/etc/nv_tegra_release").is_file():
            return True
        return "jetson" in Path("/proc/device-tree/model").read_text().lower()
    except OSError:
        return False


def training_plots_enabled(plots: bool | None = None) -> bool:
    """None follows the executing host; explicit booleans override it."""
    if plots is None:
        return not is_jetson()
    if not isinstance(plots, bool):
        raise ValueError("plots must be true, false, or null (automatic)")
    return plots

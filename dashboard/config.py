from pathlib import Path

APP_TITLE = "CAAT Visualizer"
APP_VERSION = "1.0.0"
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".svg"})
DEFAULT_PERCENTILE = 90
DEFAULT_TOP_N = 10
DEFAULT_COLS = 2
MAX_HEADS_PER_ROW = 4

# Colour scales
HEATMAP_SCALE = "Blues"
DIVERGING_SCALE = "RdBu_r"
GRADIENT_CMAP = "YlOrRd"

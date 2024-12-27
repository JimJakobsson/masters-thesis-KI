from dataclasses import dataclass, field
from typing import Dict, Tuple
@dataclass
class PlotConfig:
    """Configuration for plot styling"""
    FIGURE_SIZES: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'feature': (10, 8),
        'shap': (10, 12),
        'waterfall': (20, 15),
        'learning': (10, 6),
        'roc': (8, 8),
    })
    DPI: int = 300
    FONT_SIZES: Dict[str, int] = field(default_factory=lambda: {
        'title': 14,
        'label': 12,
        'tick': 10,
        'legend': 10
    })
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'primary': 'royalblue',
        'secondary': 'orangered',
        'tertiary': 'forestgreen',
        'background': '#f0f0f0'
    })
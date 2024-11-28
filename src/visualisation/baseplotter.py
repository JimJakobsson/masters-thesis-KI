from pathlib import Path
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from config.plot_config import PlotConfig

class BasePlotter(ABC):
    """Base class for all plotters"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.config = PlotConfig()
        
    def save_plot(self, filename: str, suffix: str = '') -> None:
        """Save plot with optional suffix"""
        if suffix:
            filename = f"{Path(filename).stem}{suffix}{Path(filename).suffix}"
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close()
    
    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        """Abstract method for plotting"""
        pass

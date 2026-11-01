

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORMATS = ("pdf",)

def configure(extension="pdf"):
    
    global FORMATS
    FORMATS = ("pdf", "png", "svg") if extension == "all" else (extension,)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def save(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        fig.savefig(
            output / f"{name}.{extension}",
            bbox_inches="tight", pad_inches=0.06, dpi=300,
        )
    plt.close(fig)

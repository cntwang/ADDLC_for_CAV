"""Consistent typography and tight figure exports."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORMATS = ("pdf",)

def configure(extension="pdf"):
    """Keep plot captions in LaTeX and choose the requested output format."""
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
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def save(fig, output, name, pad_inches=0.06):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        fig.savefig(
            output / f"{name}.{extension}",
            bbox_inches="tight", pad_inches=pad_inches, dpi=300,
        )
    plt.close(fig)

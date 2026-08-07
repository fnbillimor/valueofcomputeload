from __future__ import annotations

import shutil

import matplotlib as mpl


def configure_tex_fonts() -> None:
    """Prefer TeX-style fonts without requiring a TeX install."""
    has_tex_rendering = shutil.which("latex") is not None and shutil.which("dvipng") is not None

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "Latin Modern Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "text.usetex": has_tex_rendering,
        }
    )

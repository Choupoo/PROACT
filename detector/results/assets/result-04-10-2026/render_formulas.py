"""Render the report's equations as portable SVGs using Matplotlib mathtext.

Requires Matplotlib. Run this file to regenerate the five adjacent SVG files.
No external LaTeX installation or Markdown math extension is needed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORMULAS = {
    "predicted-loss": [
        r"$\hat{y}(x)=\mathrm{arg\,max}_{c}\,p_c(x),"
        r"\qquad L(x)=-\log p_{\hat{y}(x)}(x)$",
    ],
    "tensor-gradient": [
        r"$G_j(x)=\left\Vert\nabla_{\theta_j}L(x)\right\Vert_2$",
    ],
    "mad-threshold": [
        r"$t_{\mathrm{MAD}}=\sigma\left(\mathrm{median}(u)"
        r"+3\times1.4826\,\mathrm{median}\left|u-\mathrm{median}(u)\right|\right)$",
        r"$u=\mathrm{logit}(s)$",
    ],
    "dataset-count": [
        r"$K=\sum_{i=1}^{150}\mathbf{1}\{s(x_i)\geq t\}$",
    ],
    "kendall-profile": [
        r"$\hat{\tau}_{ab}(D)=\frac{2}{n(n-1)}"
        r"\sum_{i<j}\mathrm{sign}(z_{ia}-z_{ja})"
        r"\,\mathrm{sign}(z_{ib}-z_{jb})$",
    ],
}


def main():
    output = Path(__file__).resolve().parent
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "report-04-10-2026"})
    for name, lines in FORMULAS.items():
        fig = plt.figure(figsize=(10, 0.65 * len(lines)), facecolor="white")
        for i, line in enumerate(lines):
            fig.text(0.025, 0.75 - i * 0.48, line, fontsize=18, color="black")
        fig.savefig(
            output / (name + ".svg"),
            format="svg",
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor="white",
            metadata={"Date": None, "Title": name.replace("-", " ")},
        )
        plt.close(fig)


if __name__ == "__main__":
    main()

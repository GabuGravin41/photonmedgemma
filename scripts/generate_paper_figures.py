#!/usr/bin/env python3
"""
Generate all figures for the PhotoMedGemma papers.
Outputs publication-quality matplotlib figures as PNG and PDF.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(PROJECT, "output", "simulations", "paper_results")
FIG_DIR    = os.path.join(PROJECT, "papers", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "rank_sweep.json")) as f:
    sweep_data = json.load(f)

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "q_proj": "#2196F3",  # blue
    "k_proj": "#FF9800",  # orange
    "v_proj": "#4CAF50",  # green
    "o_proj": "#E91E63",  # pink
}
MARKERS = {"q_proj": "o", "k_proj": "s", "v_proj": "^", "o_proj": "D"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf")


# ════════════════════════════════════════════════════════════════════
# Figure 1: SVD Approximation Error vs Rank (all 4 projections)
# ════════════════════════════════════════════════════════════════════
def fig1_svd_error_vs_rank():
    fig, ax = plt.subplots(figsize=(9, 6))

    for proj_data in sweep_data["results"]:
        name = proj_data["proj"]
        ranks = [r["rank"] for r in proj_data["ranks"]]
        errors = [r["mean_svd_error"] for r in proj_data["ranks"]]
        # Replace near-zero with small value for log scale
        errors = [max(e, 1e-16) for e in errors]
        ax.semilogy(ranks, errors, marker=MARKERS[name], color=COLORS[name],
                    linewidth=2.5, markersize=9, label=name.replace("_", " "),
                    markeredgecolor="white", markeredgewidth=1)

    ax.set_xlabel("SVD Truncation Rank $r$")
    ax.set_ylabel("SVD Approximation Error $\\|W_r x - Wx\\| / \\|Wx\\|$")
    ax.set_title("SVD Approximation Error vs. Rank\n(MedGemma Layer-0 Attention, 5 Breast Cancer Images)")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(1e-16, 1.0)
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotate machine precision zone
    ax.axhspan(1e-16, 1e-13, color="green", alpha=0.07)
    ax.text(200, 3e-15, "Machine precision", fontsize=10, color="green",
            fontstyle="italic", ha="center")

    save(fig, "fig1_svd_error_vs_rank")


# ════════════════════════════════════════════════════════════════════
# Figure 2: Energy Retained vs Rank
# ════════════════════════════════════════════════════════════════════
def fig2_energy_vs_rank():
    fig, ax = plt.subplots(figsize=(9, 6))

    for proj_data in sweep_data["results"]:
        name = proj_data["proj"]
        ranks = [r["rank"] for r in proj_data["ranks"]]
        energy = [r["energy_retained"] * 100 for r in proj_data["ranks"]]
        ax.plot(ranks, energy, marker=MARKERS[name], color=COLORS[name],
                linewidth=2.5, markersize=9, label=name.replace("_", " "),
                markeredgecolor="white", markeredgewidth=1)

    ax.set_xlabel("SVD Truncation Rank $r$")
    ax.set_ylabel("Frobenius Energy Retained (%)")
    ax.set_title("Weight Matrix Energy Retained vs. SVD Rank")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    save(fig, "fig2_energy_retained_vs_rank")


# ════════════════════════════════════════════════════════════════════
# Figure 3: MZI Count vs Rank (log-log)
# ════════════════════════════════════════════════════════════════════
def fig3_mzi_count():
    fig, ax = plt.subplots(figsize=(9, 6))

    ranks = [64, 128, 256, 512, 1024, 2048]
    mzis = [r * (r - 1) // 2 for r in ranks]

    ax.loglog(ranks, mzis, "o-", color="#3F51B5", linewidth=2.5, markersize=10,
              markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    for r, m in zip(ranks, mzis):
        label = f"{m:,}"
        ax.annotate(label, (r, m), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#3F51B5", alpha=0.8))

    # Reference lines
    ax.axhline(y=10_000, color="red", linestyle="--", alpha=0.4, linewidth=1.5)
    ax.text(70, 13_000, "Current single-chip limit (~10K MZIs)", fontsize=9,
            color="red", alpha=0.7)
    ax.axhline(y=100_000, color="orange", linestyle="--", alpha=0.4, linewidth=1.5)
    ax.text(70, 130_000, "Near-term photonics (~100K MZIs)", fontsize=9,
            color="orange", alpha=0.7)

    ax.set_xlabel("SVD Truncation Rank $r$")
    ax.set_ylabel("MZIs per Unitary Mesh  $[r(r-1)/2]$")
    ax.set_title("MZI Count Scaling with Rank\n(Clements Mesh, One Unitary)")
    ax.grid(True, which="both", alpha=0.2)
    save(fig, "fig3_mzi_count_vs_rank")


# ════════════════════════════════════════════════════════════════════
# Figure 4: Chip Area vs Rank (Standard + Compact Si)
# ════════════════════════════════════════════════════════════════════
def fig4_chip_area():
    fig, ax = plt.subplots(figsize=(9, 6))

    # Use q_proj data (all projections have same MZI counts per rank)
    q = sweep_data["results"][0]
    ranks = [r["rank"] for r in q["ranks"]]
    area_std = [r["area_mm2_per_mesh_standard_si"] for r in q["ranks"]]
    area_cmp = [r["area_mm2_per_mesh_compact_si"] for r in q["ranks"]]

    ax.semilogy(ranks, area_std, "s-", color="#E91E63", linewidth=2.5,
                markersize=9, label="Standard Si (150 x 8 um)",
                markeredgecolor="white", markeredgewidth=1)
    ax.semilogy(ranks, area_cmp, "^-", color="#009688", linewidth=2.5,
                markersize=9, label="Compact Si (80 x 5 um)",
                markeredgecolor="white", markeredgewidth=1)

    # Reference: single 10x10mm chip
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(100, 120, "Single 10 mm x 10 mm chip (100 mm$^2$)",
            fontsize=9, color="gray")

    # Reference: 300mm wafer
    ax.axhline(y=70_685, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(100, 85_000, "300 mm wafer area", fontsize=9, color="gray", alpha=0.7)

    ax.set_xlabel("SVD Truncation Rank $r$")
    ax.set_ylabel("Mesh Area (mm$^2$)")
    ax.set_title("Chip Area per Unitary Mesh vs. Rank")
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc="upper left", framealpha=0.9)
    save(fig, "fig4_chip_area_vs_rank")


# ════════════════════════════════════════════════════════════════════
# Figure 5: Three Error Metrics — Bar Chart
# ════════════════════════════════════════════════════════════════════
def fig5_error_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
    chip_fid = []
    svd_err_r64 = []
    quant_err = [0.0017, 0.0019, 0.0022, 0.0020]  # from comparison results

    for proj_data in sweep_data["results"]:
        chip_fid.append(proj_data["ranks"][0]["mean_chip_fidelity"])
        svd_err_r64.append(proj_data["ranks"][0]["mean_svd_error"])

    x = np.arange(len(projs))
    width = 0.25

    bars1 = ax.bar(x - width, svd_err_r64, width, label="SVD Error (rank 64)",
                   color="#F44336", alpha=0.85, edgecolor="white", linewidth=1)
    bars2 = ax.bar(x, quant_err, width, label="Quantisation Error (NF4)",
                   color="#FF9800", alpha=0.85, edgecolor="white", linewidth=1)
    bars3 = ax.bar(x + width, chip_fid, width, label="Chip Fidelity",
                   color="#4CAF50", alpha=0.85, edgecolor="white", linewidth=1)

    ax.set_yscale("log")
    ax.set_ylabel("Relative Error")
    ax.set_title("Three Independent Error Metrics\n(MedGemma Layer-0 Attention Projections)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in projs])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(1e-16, 2.0)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.3,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 2,
                        f"{h:.1e}", ha="center", va="bottom", fontsize=7,
                        rotation=45)

    save(fig, "fig5_error_comparison_bars")


# ════════════════════════════════════════════════════════════════════
# Figure 6: Power Comparison — GPU vs Photonic
# ════════════════════════════════════════════════════════════════════
def fig6_power_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Absolute power breakdown
    categories = ["A100 GPU", "PhotoMedGemma\n(rank-64)"]
    gpu_components = {"Compute": 200, "Memory": 60, "Cooling": 40}
    pmc_components = {"Laser": 0.8, "FPGA": 5.0, "Memory": 2.0, "Detectors": 0.05, "TEC": 0.15}

    gpu_vals = list(gpu_components.values())
    gpu_labels = list(gpu_components.keys())
    gpu_colors = ["#F44336", "#FF5722", "#FF7043"]

    pmc_vals = list(pmc_components.values())
    pmc_labels = list(pmc_components.keys())
    pmc_colors = ["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB"]

    # GPU stacked bar
    bottom = 0
    for val, label, color in zip(gpu_vals, gpu_labels, gpu_colors):
        ax1.bar(0, val, bottom=bottom, color=color, width=0.5, edgecolor="white",
                linewidth=1, label=f"GPU: {label}")
        ax1.text(0, bottom + val / 2, f"{val}W", ha="center", va="center",
                 fontsize=10, color="white", fontweight="bold")
        bottom += val

    # PMC stacked bar
    bottom = 0
    for val, label, color in zip(pmc_vals, pmc_labels, pmc_colors):
        ax1.bar(1, val, bottom=bottom, color=color, width=0.5, edgecolor="white",
                linewidth=1, label=f"Photonic: {label}")
        bottom += val

    ax1.text(1, bottom + 5, f"{sum(pmc_vals):.1f}W", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#2196F3")
    ax1.text(0, sum(gpu_vals) + 5, f"{sum(gpu_vals)}W", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#F44336")

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(categories, fontsize=12)
    ax1.set_ylabel("Power (W)")
    ax1.set_title("System Power Comparison")
    ax1.set_ylim(0, 350)
    ax1.legend(loc="upper right", fontsize=9, ncol=2)

    # Right: Energy per token comparison
    platforms = ["A100 GPU\n(300W)", "T4 GPU\n(70W)", "Jetson AGX\n(30W)",
                 "PhotoMedGemma\nRank-64 (8W)", "PhotoMedGemma\nRank-64 + WDM"]
    energy_per_token = [300, 70, 30, 8, 1]  # mJ/token (WDM: ~1mJ with parallelism)
    bar_colors = ["#F44336", "#FF5722", "#FF9800", "#2196F3", "#00BCD4"]

    bars = ax2.barh(range(len(platforms)), energy_per_token, color=bar_colors,
                    edgecolor="white", linewidth=1.5, height=0.6)
    ax2.set_yticks(range(len(platforms)))
    ax2.set_yticklabels(platforms, fontsize=10)
    ax2.set_xlabel("Energy per Token (mJ)")
    ax2.set_title("Energy Efficiency Comparison")
    ax2.set_xscale("log")
    ax2.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, energy_per_token)):
        ax2.text(val * 1.3, i, f"{val} mJ", va="center", fontsize=11, fontweight="bold")

    # Annotation: 37x improvement
    ax2.annotate("37x", xy=(8, 3), xytext=(150, 1.5),
                 fontsize=16, fontweight="bold", color="#2196F3",
                 arrowprops=dict(arrowstyle="->", color="#2196F3", lw=2),
                 ha="center")

    fig.suptitle("Power and Energy Efficiency: GPU vs. Photonic Chip", fontsize=16, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_power_comparison")


# ════════════════════════════════════════════════════════════════════
# Figure 7: Compilation Pipeline Diagram
# ════════════════════════════════════════════════════════════════════
def fig7_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.5,  "MedGemma\nWeights\n(HuggingFace)", "#BBDEFB"),
        (2.5,  "SVD\nDecomposition\n$W = U\\Sigma V^\\dagger$", "#C8E6C9"),
        (4.5,  "Clements\nDecomposition\n$U = D \\cdot \\prod T_k$", "#FFF9C4"),
        (6.5,  "Phase\nQuantisation\n12-bit DAC", "#FFCCBC"),
        (8.5,  "Photonic\nNetlist\n(.pntl)", "#E1BEE7"),
        (10.5, "GDS Layout\n(.gds)", "#B2DFDB"),
        (12.5, "Photonic\nChip", "#F8BBD0"),
    ]

    for x, text, color in boxes:
        rect = FancyBboxPatch((x, 1.3), 1.6, 2.4, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="gray", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.8, 2.5, text, ha="center", va="center", fontsize=9,
                fontweight="bold", wrap=True)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.6
        x2 = boxes[i + 1][0]
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2))

    # Stage numbers
    for i, (x, _, _) in enumerate(boxes):
        ax.text(x + 0.8, 0.8, f"Stage {i + 1}", ha="center", fontsize=9,
                color="gray", fontstyle="italic")

    ax.set_title("PhotoMedGemma Compilation Pipeline", fontsize=16, pad=20)
    save(fig, "fig7_compilation_pipeline")


# ════════════════════════════════════════════════════════════════════
# Figure 8: SVD Error vs MZI Count (Pareto front / trade-off)
# ════════════════════════════════════════════════════════════════════
def fig8_error_vs_mzis():
    fig, ax = plt.subplots(figsize=(9, 6))

    for proj_data in sweep_data["results"]:
        name = proj_data["proj"]
        mzis = [r["n_mzis_per_mesh"] for r in proj_data["ranks"]]
        errors = [max(r["mean_svd_error"], 1e-16) for r in proj_data["ranks"]]
        ax.loglog(mzis, errors, marker=MARKERS[name], color=COLORS[name],
                  linewidth=2.5, markersize=9, label=name.replace("_", " "),
                  markeredgecolor="white", markeredgewidth=1)

    ax.set_xlabel("MZIs per Unitary Mesh")
    ax.set_ylabel("SVD Approximation Error")
    ax.set_title("Accuracy vs. Hardware Cost Trade-off\n(Lower-Left = Better)")
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotate corners
    ax.annotate("Low cost\nLow accuracy", xy=(2016, 0.5), fontsize=9,
                color="gray", ha="center", fontstyle="italic")
    ax.annotate("High cost\nExact", xy=(2e6, 5e-15), fontsize=9,
                color="gray", ha="center", fontstyle="italic")

    save(fig, "fig8_error_vs_mzis_tradeoff")


# ════════════════════════════════════════════════════════════════════
# Figure 9: Compilation Fidelity (bar chart, log scale)
# ════════════════════════════════════════════════════════════════════
def fig9_compilation_fidelity():
    fig, ax = plt.subplots(figsize=(9, 5))

    projs = []
    err_U = []
    err_Vh = []
    for proj_data in sweep_data["results"]:
        projs.append(proj_data["proj"].replace("_", " "))
        err_U.append(proj_data["compilation_error_U"])
        err_Vh.append(proj_data["compilation_error_Vh"])

    x = np.arange(len(projs))
    width = 0.35

    ax.bar(x - width / 2, err_U, width, label="$\\varepsilon_U$ (left mesh)",
           color="#2196F3", alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, err_Vh, width, label="$\\varepsilon_{V^\\dagger}$ (right mesh)",
           color="#FF9800", alpha=0.85, edgecolor="white")

    ax.set_yscale("log")
    ax.set_ylabel("Compilation Error (Frobenius norm)")
    ax.set_title("Clements Compilation Fidelity\n(Machine Precision for All Projections)")
    ax.set_xticks(x)
    ax.set_xticklabels(projs)
    ax.legend(framealpha=0.9)
    ax.set_ylim(1e-16, 1e-13)

    # Machine precision line
    ax.axhline(y=np.finfo(np.float64).eps, color="red", linestyle="--",
               alpha=0.5, linewidth=1.5)
    ax.text(0.5, np.finfo(np.float64).eps * 1.5, "float64 $\\varepsilon_{\\rm mach}$",
            fontsize=9, color="red", alpha=0.7)

    save(fig, "fig9_compilation_fidelity")


# ════════════════════════════════════════════════════════════════════
# Figure 10: MZI Mesh Schematic (Clements 8x8 illustration)
# ════════════════════════════════════════════════════════════════════
def fig10_clements_schematic():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    N = 8  # 8-mode illustration
    cols = N - 1  # 7 columns
    pitch = 1.0

    # Draw horizontal waveguides
    for i in range(N):
        ax.plot([-0.3, cols + 0.5], [i * pitch, i * pitch], "-", color="#90CAF9",
                linewidth=1.5, alpha=0.5, zorder=1)
        ax.text(-0.5, i * pitch, f"$a_{{{i+1}}}$", ha="right", va="center",
                fontsize=10, color="#1565C0")
        ax.text(cols + 0.7, i * pitch, f"$b_{{{i+1}}}$", ha="left", va="center",
                fontsize=10, color="#1565C0")

    # Draw MZIs (Clements pattern: alternating even/odd mode pairs)
    mzi_count = 0
    for col in range(cols):
        start = col % 2
        for row in range(start, N - 1, 2):
            y_center = (row + row + 1) * pitch / 2
            # MZI box
            rect = FancyBboxPatch((col - 0.25, y_center - 0.35), 0.5, 0.7,
                                  boxstyle="round,pad=0.05",
                                  facecolor="#FFF3E0", edgecolor="#E65100",
                                  linewidth=1.5, zorder=3)
            ax.add_patch(rect)
            ax.text(col, y_center, f"$T$", ha="center", va="center",
                    fontsize=8, color="#E65100", fontweight="bold")
            mzi_count += 1

    # Phase screen at the end
    for i in range(N):
        circle = plt.Circle((cols + 0.2, i * pitch), 0.15, facecolor="#E8F5E9",
                             edgecolor="#2E7D32", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(cols + 0.2, i * pitch, "$\\phi$", ha="center", va="center",
                fontsize=7, color="#2E7D32")

    ax.set_title(f"Clements Mesh: {N} Modes, {mzi_count} MZIs\n"
                 f"(Illustrative — rank-64 chip has 2,016 MZIs)", fontsize=14)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#FFF3E0", edgecolor="#E65100", label="MZI $T(\\theta, \\varphi)$"),
        mpatches.Patch(facecolor="#E8F5E9", edgecolor="#2E7D32", label="Phase screen $D$"),
        plt.Line2D([0], [0], color="#90CAF9", linewidth=2, label="Waveguide"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    save(fig, "fig10_clements_mesh_schematic")


# ════════════════════════════════════════════════════════════════════
# Figure 11: SVD Photonic Circuit Architecture
# ════════════════════════════════════════════════════════════════════
def fig11_svd_circuit():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Three main blocks
    blocks = [
        (1.0, "$V^\\dagger$ Mesh\n(Clements)", "#BBDEFB", "n inputs\n$x \\in \\mathbb{R}^n$"),
        (5.5, "$\\Sigma$ Stage\n(VOA Bank)", "#FFF9C4", "Singular\nvalues"),
        (10.0, "$U$ Mesh\n(Clements)", "#C8E6C9", "m outputs\n$y = Wx$"),
    ]

    for x, text, color, subtext in blocks:
        rect = FancyBboxPatch((x, 1.0), 3.0, 3.0, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="gray", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.5, 2.8, text, ha="center", va="center", fontsize=13,
                fontweight="bold")
        ax.text(x + 1.5, 1.5, subtext, ha="center", va="center", fontsize=10,
                color="gray")

    # Arrows between blocks
    for x1, x2 in [(4.0, 5.5), (8.5, 10.0)]:
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2.5))

    # Input arrow
    ax.annotate("", xy=(1.0, 2.5), xytext=(-0.2, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=2.5))
    ax.text(-0.3, 3.2, "Light in\n$\\mathbf{x}$", ha="center", fontsize=11,
            color="#1565C0", fontweight="bold")

    # Output arrow
    ax.annotate("", xy=(14.2, 2.5), xytext=(13.0, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2.5))
    ax.text(14.3, 3.2, "Light out\n$\\mathbf{y} = W\\mathbf{x}$", ha="center",
            fontsize=11, color="#2E7D32", fontweight="bold")

    # Equation
    ax.text(7.0, 4.6, "$W = U \\Sigma V^\\dagger$  (SVD)",
            ha="center", va="center", fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    ax.set_title("Photonic SVD Circuit: One Linear Layer", fontsize=16, pad=15)
    save(fig, "fig11_svd_circuit_architecture")


# ════════════════════════════════════════════════════════════════════
# Figure 12: MZI Transfer Function
# ════════════════════════════════════════════════════════════════════
def fig12_mzi_transfer():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    theta = np.linspace(0, 2 * np.pi, 500)

    # Panel 1: Power splitting vs theta (phi=0)
    ax = axes[0]
    P_bar = np.cos(theta / 2) ** 2
    P_cross = np.sin(theta / 2) ** 2
    ax.plot(np.degrees(theta), P_bar, "-", color="#2196F3", linewidth=2.5,
            label="Bar port $|\\cos(\\theta/2)|^2$")
    ax.plot(np.degrees(theta), P_cross, "-", color="#F44336", linewidth=2.5,
            label="Cross port $|\\sin(\\theta/2)|^2$")
    ax.set_xlabel("$\\theta$ (degrees)")
    ax.set_ylabel("Normalised Power")
    ax.set_title("MZI Power Splitting\n($\\varphi = 0$)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 360)

    # Panel 2: Phase response vs phi (theta = pi/2)
    ax = axes[1]
    phi = np.linspace(0, 2 * np.pi, 500)
    theta_fixed = np.pi / 2
    # bar port output phase for input [1, 0]
    out_bar = np.cos(theta_fixed / 2)
    out_cross = np.exp(1j * phi) * np.sin(theta_fixed / 2)
    ax.plot(np.degrees(phi), np.angle(out_cross), "-", color="#4CAF50", linewidth=2.5)
    ax.set_xlabel("$\\varphi$ (degrees)")
    ax.set_ylabel("Output Phase (rad)")
    ax.set_title("Phase Control\n($\\theta = \\pi/2$)")
    ax.set_xlim(0, 360)

    # Panel 3: Extinction ratio vs coupler splitting error
    ax = axes[2]
    delta_eta = np.linspace(0, 0.1, 200)  # splitting ratio error
    eta = 0.5 + delta_eta
    ER = 10 * np.log10((1 + 2 * np.sqrt(eta * (1 - eta))) /
                        (1 - 2 * np.sqrt(eta * (1 - eta)) + 1e-30))
    ax.plot(delta_eta * 100, ER, "-", color="#FF9800", linewidth=2.5)
    ax.set_xlabel("Coupler Splitting Error (%)")
    ax.set_ylabel("Extinction Ratio (dB)")
    ax.set_title("Fabrication Tolerance")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 50)

    fig.suptitle("Mach-Zehnder Interferometer Characteristics", fontsize=16, y=1.03)
    fig.tight_layout()
    save(fig, "fig12_mzi_transfer_function")


# ════════════════════════════════════════════════════════════════════
# Figure 13: Chip Hierarchy / System Architecture
# ════════════════════════════════════════════════════════════════════
def fig13_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Level 0: Single MZI
    rect = FancyBboxPatch((0.5, 6.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(1.75, 7.1, "Single MZI\n$T(\\theta, \\varphi)$", ha="center",
            va="center", fontsize=10, fontweight="bold")
    ax.text(1.75, 6.3, "50 x 200 um", ha="center", fontsize=8, color="gray")

    # Level 1: Clements Mesh
    rect = FancyBboxPatch((4.0, 6.5), 3.0, 1.2, boxstyle="round,pad=0.1",
                          facecolor="#BBDEFB", edgecolor="#1565C0", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5.5, 7.1, "Clements Mesh\n2,016 MZIs (r=64)", ha="center",
            va="center", fontsize=10, fontweight="bold")
    ax.text(5.5, 6.3, "4.9 mm$^2$", ha="center", fontsize=8, color="gray")

    # Level 2: Single Chip
    rect = FancyBboxPatch((8.0, 6.5), 3.0, 1.2, boxstyle="round,pad=0.1",
                          facecolor="#C8E6C9", edgecolor="#2E7D32", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.5, 7.1, "Single Chip\n10 x 10 mm", ha="center",
            va="center", fontsize=10, fontweight="bold")
    ax.text(9.5, 6.3, "1 unitary mesh", ha="center", fontsize=8, color="gray")

    # Level 3: Layer-0 Attention Module
    rect = FancyBboxPatch((1.0, 4.0), 5.0, 1.8, boxstyle="round,pad=0.1",
                          facecolor="#E1BEE7", edgecolor="#7B1FA2", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3.5, 5.1, "Layer-0 Attention (rank-64)", ha="center",
            va="center", fontsize=11, fontweight="bold")

    # 4 projection sub-boxes
    proj_names = ["Q", "K", "V", "O"]
    proj_colors = ["#BBDEFB", "#FFCCBC", "#C8E6C9", "#FFF9C4"]
    for i, (name, col) in enumerate(zip(proj_names, proj_colors)):
        small = FancyBboxPatch((1.3 + i * 1.15, 4.2), 1.0, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=col, edgecolor="gray", linewidth=1)
        ax.add_patch(small)
        ax.text(1.8 + i * 1.15, 4.5, f"{name}\n2 chips", ha="center",
                va="center", fontsize=8)

    ax.text(3.5, 3.8, "8 chips total (4 proj x 2 meshes)", ha="center",
            fontsize=9, color="#7B1FA2", fontstyle="italic")

    # Level 4: Full MedGemma System
    rect = FancyBboxPatch((7.5, 4.0), 5.5, 1.8, boxstyle="round,pad=0.1",
                          facecolor="#F8BBD0", edgecolor="#C2185B", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(10.25, 5.1, "Full MedGemma-4B (rank-64)", ha="center",
            va="center", fontsize=11, fontweight="bold")
    ax.text(10.25, 4.6, "46 layers x 7 matrices x 2 meshes", ha="center",
            fontsize=9, color="gray")
    ax.text(10.25, 4.2, "806 chips (no WDM) / 101 chips (with WDM)",
            ha="center", fontsize=9, fontweight="bold", color="#C2185B")

    # Arrows
    ax.annotate("", xy=(4.0, 7.1), xytext=(3.0, 7.1),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
    ax.annotate("", xy=(8.0, 7.1), xytext=(7.0, 7.1),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
    ax.annotate("", xy=(3.5, 5.8), xytext=(3.5, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
    ax.annotate("", xy=(10.25, 5.8), xytext=(10.25, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    # Power budget box
    rect = FancyBboxPatch((3.0, 0.5), 8.0, 2.8, boxstyle="round,pad=0.15",
                          facecolor="#FAFAFA", edgecolor="#616161", linewidth=1.5,
                          linestyle="--")
    ax.add_patch(rect)
    ax.text(7.0, 3.0, "Power Budget Comparison", ha="center", fontsize=12,
            fontweight="bold")
    power_text = (
        "A100 GPU:          300 W   |   300 mJ/token\n"
        "PhotoMedGemma:       8 W   |     8 mJ/token\n"
        "Improvement:                     37x"
    )
    ax.text(7.0, 1.8, power_text, ha="center", va="center", fontsize=11,
            fontfamily="monospace",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_title("PhotoMedGemma System Hierarchy", fontsize=16, pad=15)
    save(fig, "fig13_system_architecture")


# ════════════════════════════════════════════════════════════════════
# Figure 14: Heatmap — SVD Error across Projections and Ranks
# ════════════════════════════════════════════════════════════════════
def fig14_error_heatmap():
    fig, ax = plt.subplots(figsize=(10, 5))

    projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
    all_ranks = [64, 128, 256, 512, 1024, 2048]

    # Build matrix (fill missing with NaN)
    matrix = np.full((len(projs), len(all_ranks)), np.nan)
    for i, proj_data in enumerate(sweep_data["results"]):
        for r_data in proj_data["ranks"]:
            r = r_data["rank"]
            if r in all_ranks:
                j = all_ranks.index(r)
                val = r_data["mean_svd_error"]
                matrix[i, j] = val if val > 1e-10 else 0.0

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.7)
    ax.set_xticks(range(len(all_ranks)))
    ax.set_xticklabels(all_ranks)
    ax.set_yticks(range(len(projs)))
    ax.set_yticklabels([p.replace("_", " ") for p in projs])
    ax.set_xlabel("SVD Truncation Rank")
    ax.set_ylabel("Attention Projection")
    ax.set_title("SVD Approximation Error Heatmap\n(Green = Low Error, Red = High Error)")

    # Annotate cells
    for i in range(len(projs)):
        for j in range(len(all_ranks)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9,
                        color="gray")
            elif val < 1e-10:
                ax.text(j, i, "0.000", ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold")
            else:
                color = "white" if val > 0.35 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("SVD Approximation Error")
    fig.tight_layout()
    save(fig, "fig14_svd_error_heatmap")


# ════════════════════════════════════════════════════════════════════
# Figure 15: Rank vs Area with highlighted feasibility zones
# ════════════════════════════════════════════════════════════════════
def fig15_feasibility_zones():
    fig, ax = plt.subplots(figsize=(10, 7))

    q = sweep_data["results"][0]
    ranks = [r["rank"] for r in q["ranks"]]
    area_cmp = [r["area_mm2_per_mesh_compact_si"] for r in q["ranks"]]
    svd_err = [r["mean_svd_error"] for r in q["ranks"]]

    # Scatter with size proportional to MZI count
    mzis = [r["n_mzis_per_mesh"] for r in q["ranks"]]
    sizes = [np.sqrt(m) * 0.3 for m in mzis]

    scatter = ax.scatter(area_cmp, svd_err, s=sizes, c=ranks,
                         cmap="viridis", edgecolors="white", linewidth=2,
                         zorder=5, alpha=0.85)
    ax.plot(area_cmp, svd_err, "--", color="gray", alpha=0.4, zorder=1)

    # Label each point
    for r, a, e in zip(ranks, area_cmp, svd_err):
        offset = (10, 10) if r != 2048 else (10, -15)
        ax.annotate(f"r={r}", (a, max(e, 1e-16)),
                    textcoords="offset points", xytext=offset,
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="gray", alpha=0.8))

    # Feasibility zones
    ax.axvspan(0, 100, color="green", alpha=0.06)
    ax.axvspan(100, 500, color="yellow", alpha=0.06)
    ax.axvspan(500, 2000, color="orange", alpha=0.06)

    ax.text(30, 0.19, "Single chip", fontsize=10, color="green",
            fontstyle="italic", fontweight="bold")
    ax.text(200, 0.19, "Multi-chip\nmodule", fontsize=10, color="#F57F17",
            fontstyle="italic", fontweight="bold")
    ax.text(800, 0.19, "Wafer-scale", fontsize=10, color="#E65100",
            fontstyle="italic", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Mesh Area, Compact Si (mm$^2$)")
    ax.set_ylabel("SVD Approximation Error (q\\_proj)")
    ax.set_title("Accuracy vs. Manufacturing Feasibility\n(q\\_proj, Compact Si Technology)")
    ax.set_ylim(-0.01, 0.22)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("Rank")

    save(fig, "fig15_feasibility_zones")


# ════════════════════════════════════════════════════════════════════
# Run all figures
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating PhotoMedGemma paper figures...")
    print(f"Output directory: {FIG_DIR}\n")

    fig1_svd_error_vs_rank()
    fig2_energy_vs_rank()
    fig3_mzi_count()
    fig4_chip_area()
    fig5_error_comparison()
    fig6_power_comparison()
    fig7_pipeline()
    fig8_error_vs_mzis()
    fig9_compilation_fidelity()
    fig10_clements_schematic()
    fig11_svd_circuit()
    fig12_mzi_transfer()
    fig13_system_architecture()
    fig14_error_heatmap()
    fig15_feasibility_zones()

    print(f"\nDone! {len(os.listdir(FIG_DIR))} files generated in {FIG_DIR}")

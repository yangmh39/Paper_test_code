# -*- coding: utf-8 -*-
"""
实验结果可视化：读取 results.csv 生成 4 张论文用图。
用法: python -I plot_results.py [--csv results.csv] [--src cleaned_data.csv]
      [--products 60] [--sims 4000] [--seed 0]
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex

# ---- dataviz 默认调色板（light mode） ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CAT = {"U": "#2a78d6", "L": "#eb6834", "C": "#1baf7a", "P": "#eda100"}
METHOD_CAT = {"post": "#2a78d6", "ro": "#eb6834", "sp": "#1baf7a", "mc": "#eda100"}
LAYOUT_NAMES = {"U": "U型", "L": "直线", "C": "环型", "P": "并行"}
METHOD_NAMES = {"post": "后评估法", "ro": "鲁棒优化RO", "sp": "随机规划SP", "mc": "基础MC"}
SCEN_NAMES = {"none": "无扰动", "fatigue": "仅疲劳", "failure": "仅故障",
              "both_low": "疲劳+故障(低)", "both_high": "疲劳+故障(高)"}
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#184f95"])
RED_CMAP = LinearSegmentedColormap.from_list("red_seq", ["#f7d4d0", "#e88680", "#a3261f"])

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": SEC,
    "text.color": INK,
    "xtick.color": SEC,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig1_layout_metrics(df):
    """H1：各布局在后评估法高扰动下的 5 项鲁棒性指标。"""
    sub = df[(df.method == "post") & (df.scenario == "both_high")]
    order = ["U", "L", "C", "P"]
    metrics = [
        ("exp_profit", "期望利润 E[f1]", "元", lambda v: f"{v:,.0f}"),
        ("exp_fatigue", "期望疲劳 E[f2]", "", lambda v: f"{v:,.0f}"),
        ("success_rate", "可行率 R_feas", "", lambda v: f"{v:.2f}"),
        ("profit_std", "利润标准差 √Var[f1]", "元", lambda v: f"{v:,.0f}"),
        ("avg_loss", "失效场景平均损失", "元", lambda v: f"{v:,.0f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    for ax, (col, title, unit, fmt) in zip(axes.ravel(), metrics):
        vals = [sub[sub.layout == l][col].iloc[0] for l in order]
        colors = [CAT[l] for l in order]
        bars = ax.bar([LAYOUT_NAMES[l] for l in order], vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt(v), ha="center", va="bottom", fontsize=8, color=SEC)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.tick_params(axis="x", labelsize=8)
    axes.ravel()[-1].axis("off")  # 隐藏第 6 个空面板
    handles = [Patch(color=CAT[l], label=LAYOUT_NAMES[l]) for l in order]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=9)
    fig.suptitle("图1  各布局在疲劳+故障高强度扰动下的鲁棒性指标（后评估法）",
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig("fig1_layout_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _heat(ax, mat, rows, cols, cmap, fmt, title):
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=np.max(mat) if np.max(mat) > 0 else 1,
                   aspect="auto")
    ax.set_xticks(range(len(cols)), [LAYOUT_NAMES[c] for c in cols], fontsize=9)
    ax.set_yticks(range(len(rows)), [SCEN_NAMES[s] for s in rows], fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            txt = "white" if v > np.max(mat) * 0.6 else INK
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=9, color=txt, fontweight="bold")
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    ax.grid(False)
    return im


def fig2_sensitivity_heatmap(df):
    """H3：布局 × 扰动源 的敏感度（可行率 与 失效平均损失 双热力图）。"""
    sub = df[df.method == "post"]
    rows = ["none", "fatigue", "failure", "both_low", "both_high"]
    cols = ["U", "L", "C", "P"]
    feas_mat = np.array([[sub[(sub.scenario == s) & (sub.layout == l)]["success_rate"].iloc[0]
                          for l in cols] for s in rows])
    loss_mat = np.array([[sub[(sub.scenario == s) & (sub.layout == l)]["avg_loss"].iloc[0]
                          for l in cols] for s in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im1 = _heat(axes[0], feas_mat, rows, cols, BLUE_CMAP, lambda v: f"{v:.2f}",
                "可行率 R_feas（越高越稳健）")
    im2 = _heat(axes[1], loss_mat, rows, cols, RED_CMAP, lambda v: f"{v:,.0f}",
                "失效场景平均损失（越低越稳健）")
    cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.suptitle("图2  各布局对不同扰动源的敏感度（后评估法）", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig("fig2_sensitivity_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3_method_tradeoff(df):
    """H2：方法权衡散点（x=可行率, y=期望利润），点按方法着色、按布局标注。"""
    sub = df[df.scenario == "both_high"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ["post", "ro", "sp", "mc"]:
        msub = sub[sub.method == m]
        ax.scatter(msub.success_rate, msub.exp_profit, s=70,
                   color=METHOD_CAT[m], label=METHOD_NAMES[m], zorder=3,
                   edgecolor="white", linewidth=1.5)
        for _, r in msub.iterrows():
            ax.annotate(LAYOUT_NAMES[r.layout],
                        (r.success_rate, r.exp_profit),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=SEC)
    ax.set_xlabel("可行率（鲁棒性）", fontsize=10)
    ax.set_ylabel("期望利润 E[f1]", fontsize=10)
    ax.set_title("图3  后评估法 vs 鲁棒优化/随机规划 的性能权衡（高扰动）",
                 fontsize=11, color=INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig3_method_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4_profit_distribution(src, products_n, sims, seed):
    """H1：各布局在后评估法高扰动下的利润分布（KDE，可复现重采样）。"""
    products = ex.load_products(src, products_n, seed)
    products = products.sort_values("price", ascending=False).reset_index(drop=True)
    cands = ex.build_candidates(products)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for l in ["U", "L", "C", "P"]:
        plan = ex.solve_stage1(cands, l, "post")
        prof, _, _, _ = ex.simulate_batch(plan, l, sigma_t=0.30, sigma_e=0.3,
                                          p_f=0.15, n_sim=sims, seed=seed)
        ax.hist(prof, bins=60, density=True, histtype="step", linewidth=1.8,
                color=CAT[l], label=LAYOUT_NAMES[l], alpha=0.9)
    ax.set_xlabel("利润", fontsize=10)
    ax.set_ylabel("密度", fontsize=10)
    ax.set_title("图4  利润分布与稳定性（后评估法，高扰动）",
                 fontsize=11, color=INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig4_profit_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--src", default="cleaned_data.csv")
    ap.add_argument("--products", type=int, default=60)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fig1_layout_metrics(df)
    fig2_sensitivity_heatmap(df)
    fig3_method_tradeoff(df)
    fig4_profit_distribution(args.src, args.products, args.sims, args.seed)
    print("已生成: fig1_layout_metrics.png, fig2_sensitivity_heatmap.png, "
          "fig3_method_tradeoff.png, fig4_profit_distribution.png")


if __name__ == "__main__":
    main()

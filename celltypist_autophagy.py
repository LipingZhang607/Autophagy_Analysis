#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE189050: CellTypist精细注释 + 自噬基因跨亚群分析
重点: DNT细胞及其他免疫亚群与SLE / 自噬的关系
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import scipy.stats as stats
from scipy.sparse import issparse
import warnings
warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
H5AD      = "/home/h3033/statics/WQ/output/exploration/GSE189050.h5ad"
ATG_CSV   = "/home/h3033/statics/WQ/Autophagy.csv"
OUT_DIR   = "/home/h3033/statics/WQ/output/celltypist_autophagy"
os.makedirs(OUT_DIR, exist_ok=True)

sc.settings.set_figure_params(dpi=150, fontsize=10, facecolor="white")
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

# ── 自噬基因集 ────────────────────────────────────────────────────────────────
ATG_ALL   = pd.read_csv(ATG_CSV)["Symbol"].dropna().str.strip().tolist()

# 差异自噬基因（SLE相关，来自你的分析）
UP_GENES  = ["FOS", "CCL2", "EIF2AK2", "DDIT3", "TNFSF10", "PPP1R15A",
             "NAMPT", "FAS", "MAPK3", "CASP1"]
DOWN_GENES = ["ST13", "BAG3", "ITGA6", "FOXO3", "BNIP3L", "BCL2L1", "BAG1"]
SIG_GENES = UP_GENES + DOWN_GENES  # 17个差异基因

# ─────────────────────────────────────────────────────────────────────────────
# 1. 加载数据
# ─────────────────────────────────────────────────────────────────────────────
print("[1/6] 加载 h5ad...")
adata = sc.read_h5ad(H5AD)
print(f"  {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"  分组: {adata.obs['classification'].value_counts().to_dict()}")

# 确保用 log1p-normalized 数据（CellTypist要求）
# h5ad 里存的是 counts，需要先normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("  完成 normalize_total + log1p")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CellTypist 注释（全局：Immune_All_High）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] CellTypist 注释 (Immune_All_High)...")
models.download_models(model=["Immune_All_High.pkl", "Developing_Human_Thymus.pkl"],
                       force_update=False)

model_high = models.Model.load(model="Immune_All_High.pkl")
pred_high  = celltypist.annotate(adata, model=model_high,
                                  majority_voting=True, over_clustering="wsnn.res.0.8")
adata = pred_high.to_adata()
adata.obs["celltypist_high"]   = adata.obs["majority_voting"].copy()
adata.obs["celltypist_high_p"] = adata.obs["predicted_labels"].copy()
print(f"  细胞类型数: {adata.obs['celltypist_high'].nunique()}")
print(adata.obs["celltypist_high"].value_counts().head(20).to_string())

# ───────────────────────────────────────────────────────────────────────────���─
# 3. T细胞子集：Developing_Human_Thymus 精细注释 + DNT鉴定
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] T细胞精细注释 + DNT鉴定...")

# 用 CellTypist high 标签筛选T细胞相关亚群
t_keywords = ["T cell", "T-cell", "NK", "NKT", "ILC", "Treg",
              "CD4", "CD8", "Innate", "gamma", "delta"]
t_mask = adata.obs["celltypist_high"].str.contains(
    "|".join(t_keywords), case=False, na=False)
# 也纳入原始注释的T/NK细胞
orig_t_mask = adata.obs["coarse_cell_type"].isin(["T cells", "NK cells"])
t_mask = t_mask | orig_t_mask

adata_t = adata[t_mask].copy()
print(f"  T/NK 亚群细胞数: {adata_t.shape[0]}")

# Developing_Human_Thymus 注释
model_thy = models.Model.load(model="Developing_Human_Thymus.pkl")
pred_thy   = celltypist.annotate(adata_t, model=model_thy,
                                  majority_voting=True)
adata_t    = pred_thy.to_adata()
adata_t.obs["celltypist_thymus"] = adata_t.obs["majority_voting"].copy()

# DNT 标识：CD3E/D/G 高 + CD4 低 + CD8A/B 低
def is_dnt(adata_sub, cd3_thresh=0.5, cd4_thresh=0.3, cd8_thresh=0.3):
    """基于表达量鉴定 DNT 细胞"""
    results = {}
    for gene in ["CD3E", "CD3D", "CD3G", "CD4", "CD8A", "CD8B"]:
        if gene in adata_sub.var_names:
            idx = list(adata_sub.var_names).index(gene)
            x = adata_sub.X[:, idx]
            results[gene] = np.asarray(x.todense()).ravel() if issparse(x) else np.asarray(x).ravel()
        else:
            results[gene] = np.zeros(adata_sub.shape[0])

    cd3_pos = (results["CD3E"] > cd3_thresh) | (results["CD3D"] > cd3_thresh)
    cd4_neg = results["CD4"] < cd4_thresh
    cd8_neg = (results["CD8A"] < cd8_thresh) & (results["CD8B"] < cd8_thresh)
    return cd3_pos & cd4_neg & cd8_neg

dnt_flag = is_dnt(adata_t)
adata_t.obs["is_DNT"] = dnt_flag
print(f"  DNT细胞数 (CD3+CD4-CD8-): {dnt_flag.sum()}")

# 将 DNT 标签写回主 adata
adata.obs["celltypist_thymus"] = "Non-T/NK"
adata.obs.loc[adata_t.obs.index, "celltypist_thymus"] = adata_t.obs["celltypist_thymus"]
adata.obs["is_DNT"] = False
adata.obs.loc[adata_t.obs.index, "is_DNT"] = adata_t.obs["is_DNT"]

# 构建最终细胞类型标签（DNT单独标出）
adata.obs["cell_label"] = adata.obs["celltypist_high"].astype(str)
dnt_idx = adata.obs["is_DNT"]
adata.obs.loc[dnt_idx, "cell_label"] = "Double Negative T (DNT)"

# ─────────────────────────────────────────────────────────────────────────────
# 4. 自噬基因可用性检查
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] 自噬基因检查...")
avail_atg = [g for g in ATG_ALL if g in adata.var_names]
avail_sig = [g for g in SIG_GENES if g in adata.var_names]
avail_up  = [g for g in UP_GENES  if g in adata.var_names]
avail_dn  = [g for g in DOWN_GENES if g in adata.var_names]
print(f"  全自噬基因: {len(avail_atg)}/{len(ATG_ALL)} 可用")
print(f"  差异基因:   {len(avail_sig)}/{len(SIG_GENES)} 可用")
print(f"  上调基因:   {avail_up}")
print(f"  下调基因:   {avail_dn}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 辅助函数
# ─────────────────────────────────────────────────────────────────────────────
def get_expr(adata_sub, gene):
    if gene not in adata_sub.var_names:
        return None
    idx = list(adata_sub.var_names).index(gene)
    x = adata_sub.X[:, idx]
    return np.asarray(x.todense()).ravel() if issparse(x) else np.asarray(x).ravel()

def mean_expr_table(adata_sub, genes, groupby):
    """返回 DataFrame: rows=groups, cols=genes, values=mean expr"""
    records = []
    for grp, sub_idx in adata_sub.obs.groupby(groupby).groups.items():
        sub = adata_sub[sub_idx]
        row = {"group": grp}
        for g in genes:
            e = get_expr(sub, g)
            row[g] = e.mean() if e is not None else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("group")

def pct_expr_table(adata_sub, genes, groupby):
    records = []
    for grp, sub_idx in adata_sub.obs.groupby(groupby).groups.items():
        sub = adata_sub[sub_idx]
        row = {"group": grp}
        for g in genes:
            e = get_expr(sub, g)
            row[g] = (e > 0).mean() * 100 if e is not None else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("group")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 绘图
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] 绘图...")

# 聚焦细胞类型（过滤太少的亚群）
min_cells = 30
ct_counts = adata.obs["cell_label"].value_counts()
valid_cts  = ct_counts[ct_counts >= min_cells].index.tolist()
adata_plot = adata[adata.obs["cell_label"].isin(valid_cts)].copy()

# ── Fig1: UMAP 双图（原始注释 vs CellTypist High + DNT）──────────────────────
print("  Fig1: UMAP...")
fig1, axes = plt.subplots(1, 3, figsize=(24, 7))

sc.pl.embedding(adata, basis="X_umap_wnn", color="fine_cell_type",
                title="Original annotation", ax=axes[0], show=False,
                legend_loc="right margin", frameon=False)
sc.pl.embedding(adata, basis="X_umap_wnn", color="celltypist_high",
                title="CellTypist (Immune_All_High)", ax=axes[1], show=False,
                legend_loc="right margin", frameon=False)
sc.pl.embedding(adata, basis="X_umap_wnn", color="cell_label",
                title="Final label (DNT highlighted)", ax=axes[2], show=False,
                legend_loc="right margin", frameon=False)

fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "Fig1_UMAP_annotation.pdf"), bbox_inches="tight")
fig1.savefig(os.path.join(OUT_DIR, "Fig1_UMAP_annotation.png"), bbox_inches="tight")
plt.close(fig1)

# ── Fig2: 差异自噬基因 × 细胞类型热图（mean expr，按SLE/Control分）─────────
print("  Fig2: 自噬差异基因热图...")
gene_order = avail_up + avail_dn  # 上调在上，下调在下

# 分组: condition × cell_type
adata_plot.obs["cond_ct"] = (adata_plot.obs["classification"].astype(str) + " | "
                              + adata_plot.obs["cell_label"].astype(str))
df_mean = mean_expr_table(adata_plot, gene_order, "cond_ct")

# 按condition和celltype排序
df_mean = df_mean.sort_index()

if not df_mean.empty and gene_order:
    fig2, ax = plt.subplots(figsize=(len(gene_order) * 0.9 + 2, len(df_mean) * 0.4 + 2))
    vmax = np.nanpercentile(df_mean.values, 95)
    im = ax.imshow(df_mean.values, aspect="auto", cmap="RdYlBu_r",
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(len(gene_order)))
    ax.set_xticklabels(gene_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(df_mean)))
    ax.set_yticklabels(df_mean.index, fontsize=7)
    # 标记上调/下调分界线
    ax.axvline(len(avail_up) - 0.5, color="black", lw=1.5, linestyle="--")
    ax.text(len(avail_up)/2 - 0.5, -1.2, "↑ in SLE", ha="center", fontsize=9, color="#D6604D")
    ax.text(len(avail_up) + len(avail_dn)/2 - 0.5, -1.2, "↓ in SLE", ha="center", fontsize=9, color="#4393C3")
    plt.colorbar(im, ax=ax, shrink=0.4, label="Mean log1p expr")
    ax.set_title("SLE-related autophagy genes across cell types & conditions", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "Fig2_autophagy_heatmap.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(OUT_DIR, "Fig2_autophagy_heatmap.png"), bbox_inches="tight")
    plt.close(fig2)

# ── Fig3: 各细胞类型 × 3组 差异自噬基因点图 ─────────────────────────────────
print("  Fig3: 点图（%expr × mean）...")
# 每个细胞类型内计算3组的表达
cell_types_focus = [ct for ct in valid_cts if adata_plot.obs["cell_label"].eq(ct).sum() >= min_cells]

dot_records = []
for ct in cell_types_focus:
    sub_ct = adata_plot[adata_plot.obs["cell_label"] == ct]
    for cond in ["Control", "SLE INACT", "SLE ACT"]:
        sub = sub_ct[sub_ct.obs["classification"] == cond]
        if len(sub) < 10:
            continue
        for g in gene_order:
            e = get_expr(sub, g)
            if e is None:
                continue
            dot_records.append({
                "cell_type": ct, "condition": cond, "gene": g,
                "mean_expr": e.mean(), "pct_expr": (e > 0).mean() * 100,
                "n_cells": len(sub)
            })

df_dot = pd.DataFrame(dot_records)
if not df_dot.empty:
    df_dot.to_csv(os.path.join(OUT_DIR, "autophagy_dot_stats.csv"), index=False)

    # 绘制点图（只展示DNT和主要T细胞亚群 + 代表性其他亚群）
    focus_cts = [ct for ct in cell_types_focus
                 if any(k in ct for k in ["T", "NK", "B cell", "Mono", "DNT", "Plasma", "pDC"])]
    focus_cts = focus_cts[:12]  # 最多12个

    n_ct = len(focus_cts)
    n_g  = len(gene_order)
    if n_ct > 0:
        fig3, axes3 = plt.subplots(1, n_ct, figsize=(n_ct * 3.5 + 1, n_g * 0.45 + 2),
                                    sharey=True)
        if n_ct == 1:
            axes3 = [axes3]

        cond_colors = {"Control": "#4393C3", "SLE INACT": "#FDB863", "SLE ACT": "#D6604D"}
        y_pos = np.arange(len(gene_order))

        for ax_i, ct in enumerate(focus_cts):
            ax3 = axes3[ax_i]
            sub_df = df_dot[df_dot["cell_type"] == ct]

            for ci, cond in enumerate(["Control", "SLE INACT", "SLE ACT"]):
                cond_df = sub_df[sub_df["condition"] == cond].set_index("gene")
                x_offset = (ci - 1) * 0.3
                for gi, g in enumerate(gene_order):
                    if g not in cond_df.index:
                        continue
                    pct  = cond_df.loc[g, "pct_expr"]
                    mean = cond_df.loc[g, "mean_expr"]
                    ax3.scatter(x_offset, gi, s=max(pct * 2, 5),
                                c=cond_colors[cond], alpha=0.8,
                                edgecolors="gray", linewidths=0.3)

            ax3.set_xlim(-0.6, 0.6)
            ax3.set_xticks([-0.3, 0, 0.3])
            ax3.set_xticklabels(["Ctrl", "INACT", "ACT"], fontsize=6, rotation=45)
            ax3.axhline(len(avail_up) - 0.5, color="gray", lw=0.8, linestyle="--")
            ct_label = ct[:20] + ".." if len(ct) > 22 else ct
            ax3.set_title(ct_label, fontsize=7, fontweight="bold")

        axes3[0].set_yticks(y_pos)
        axes3[0].set_yticklabels(gene_order, fontsize=8)
        # 在最后一个轴上画图例
        for cond, color in cond_colors.items():
            axes3[-1].scatter([], [], c=color, s=60, label=cond, alpha=0.8)
        axes3[-1].legend(title="Condition", fontsize=7, frameon=False,
                         bbox_to_anchor=(1.05, 1), loc="upper left")

        fig3.suptitle("Autophagy gene expression: % expressing cells × mean\n(circle size = % cells, color = condition)",
                      fontsize=11, fontweight="bold")
        fig3.tight_layout()
        fig3.savefig(os.path.join(OUT_DIR, "Fig3_dotplot_by_celltype.pdf"), bbox_inches="tight")
        fig3.savefig(os.path.join(OUT_DIR, "Fig3_dotplot_by_celltype.png"), bbox_inches="tight")
        plt.close(fig3)

# ── Fig4: DNT 细胞专项分析 ────────────────────────────────────────────────────
print("  Fig4: DNT细胞专项...")
adata_dnt = adata[adata.obs["is_DNT"]].copy()
print(f"  DNT cells: {len(adata_dnt)}")

if len(adata_dnt) >= 20:
    fig4 = plt.figure(figsize=(20, 12))
    gs4  = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.5, wspace=0.4)

    # DNT 在 UMAP 上的位置
    ax_u = fig4.add_subplot(gs4[0, 0])
    adata.obs["is_DNT_str"] = adata.obs["is_DNT"].map({True: "DNT", False: "Other"})
    sc.pl.embedding(adata, basis="X_umap_wnn", color="is_DNT_str",
                    palette={"DNT": "#E31A1C", "Other": "#CCCCCC88"},
                    title=f"DNT cells on UMAP (n={len(adata_dnt)})",
                    ax=ax_u, show=False, frameon=False, size=3)

    # DNT 内 Thymus 亚型分布
    ax_t = fig4.add_subplot(gs4[0, 1])
    thy_counts = adata_dnt.obs["celltypist_thymus"].value_counts()
    ax_t.barh(range(len(thy_counts)), thy_counts.values, color=plt.cm.Set2(np.linspace(0, 1, len(thy_counts))))
    ax_t.set_yticks(range(len(thy_counts)))
    ax_t.set_yticklabels(thy_counts.index, fontsize=8)
    ax_t.set_xlabel("Cell count")
    ax_t.set_title("DNT sub-types\n(Developing_Human_Thymus model)", fontweight="bold")

    # DNT 在3组中的比例
    ax_p = fig4.add_subplot(gs4[0, 2])
    dnt_prop = adata.obs.groupby("classification")["is_DNT"].mean() * 100
    bars = ax_p.bar(dnt_prop.index, dnt_prop.values,
                    color=["#4393C3", "#FDB863", "#D6604D"])
    ax_p.set_ylabel("% DNT cells")
    ax_p.set_title("DNT proportion by condition", fontweight="bold")
    for bar, val in zip(bars, dnt_prop.values):
        ax_p.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                  f"{val:.2f}%", ha="center", fontsize=9)

    # DNT 自噬差异基因 3组比较（violin）
    ax_v = fig4.add_subplot(gs4[1, :])
    viol_data = []
    viol_labels = []
    viol_colors = []
    cond_c = {"Control": "#4393C3", "SLE INACT": "#FDB863", "SLE ACT": "#D6604D"}
    for g in gene_order[:10]:  # 最多10个基因
        for cond in ["Control", "SLE INACT", "SLE ACT"]:
            sub = adata_dnt[adata_dnt.obs["classification"] == cond]
            if len(sub) < 5:
                continue
            e = get_expr(sub, g)
            if e is None:
                continue
            viol_data.append(e)
            viol_labels.append(f"{g}\n{cond[:4]}")
            viol_colors.append(cond_c[cond])

    if viol_data:
        parts = ax_v.violinplot(viol_data, showmedians=True, showextrema=False)
        for i, (pc, col) in enumerate(zip(parts["bodies"], viol_colors)):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax_v.set_xticks(range(1, len(viol_labels) + 1))
        ax_v.set_xticklabels(viol_labels, fontsize=6, rotation=45, ha="right")
        ax_v.set_ylabel("log1p expression")
        ax_v.set_title("DNT cells: autophagy gene expression by condition", fontweight="bold")

    fig4.suptitle("Double Negative T (DNT) Cell Analysis", fontsize=14, fontweight="bold")
    fig4.savefig(os.path.join(OUT_DIR, "Fig4_DNT_analysis.pdf"), bbox_inches="tight")
    fig4.savefig(os.path.join(OUT_DIR, "Fig4_DNT_analysis.png"), bbox_inches="tight")
    plt.close(fig4)
else:
    print(f"  [跳过] DNT细胞数不足（{len(adata_dnt)}），阈值调整中...")
    # 放宽阈值重试
    dnt_flag2 = is_dnt(adata_t, cd3_thresh=0.3, cd4_thresh=0.5, cd8_thresh=0.5)
    print(f"  放宽后DNT: {dnt_flag2.sum()}")

# ── Fig5: 统计检验：各亚群自噬评分（SLE ACT vs Control）─────────────────────
print("  Fig5: 自噬评分统计...")

# 计算每个细胞的自噬评分（上调基因均值 - 下调基因均值）
def autophagy_score(adata_sub, up_genes, dn_genes):
    up_exprs = [get_expr(adata_sub, g) for g in up_genes if g in adata_sub.var_names]
    dn_exprs = [get_expr(adata_sub, g) for g in dn_genes if g in adata_sub.var_names]
    up_mean = np.mean(up_exprs, axis=0) if up_exprs else np.zeros(adata_sub.shape[0])
    dn_mean = np.mean(dn_exprs, axis=0) if dn_exprs else np.zeros(adata_sub.shape[0])
    return up_mean - dn_mean  # 正值 = 偏SLE自噬模式

adata.obs["autophagy_score"] = 0.0
for ct in valid_cts:
    idx = adata.obs["cell_label"] == ct
    adata.obs.loc[idx, "autophagy_score"] = autophagy_score(adata[idx], avail_up, avail_dn)

# 各亚群 Control vs SLE ACT 的自噬评分比较
stat_records = []
for ct in valid_cts:
    sub_ct = adata[adata.obs["cell_label"] == ct]
    ctrl_s  = sub_ct[sub_ct.obs["classification"] == "Control"].obs["autophagy_score"].values
    act_s   = sub_ct[sub_ct.obs["classification"] == "SLE ACT"].obs["autophagy_score"].values
    inact_s = sub_ct[sub_ct.obs["classification"] == "SLE INACT"].obs["autophagy_score"].values
    if len(ctrl_s) < 10 or len(act_s) < 10:
        continue
    _, p_act   = stats.mannwhitneyu(act_s,   ctrl_s, alternative="two-sided")
    _, p_inact = stats.mannwhitneyu(inact_s, ctrl_s, alternative="two-sided")
    stat_records.append({
        "cell_type": ct,
        "n_ctrl": len(ctrl_s), "n_act": len(act_s), "n_inact": len(inact_s),
        "mean_ctrl": ctrl_s.mean(), "mean_act": act_s.mean(), "mean_inact": inact_s.mean(),
        "delta_act": act_s.mean() - ctrl_s.mean(),
        "delta_inact": inact_s.mean() - ctrl_s.mean(),
        "p_act_vs_ctrl": p_act, "p_inact_vs_ctrl": p_inact
    })

stat_df = pd.DataFrame(stat_records).sort_values("delta_act", ascending=False)
stat_df.to_csv(os.path.join(OUT_DIR, "autophagy_score_stats.csv"), index=False)
print(stat_df[["cell_type","delta_act","p_act_vs_ctrl"]].to_string(index=False))

# 绘制自噬评分差异图
if not stat_df.empty:
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, max(len(stat_df) * 0.5 + 1, 6)))

    y = np.arange(len(stat_df))
    # SLE ACT vs Control
    colors5 = ["#D6604D" if d > 0 else "#4393C3" for d in stat_df["delta_act"]]
    ax5a.barh(y, stat_df["delta_act"], color=colors5, alpha=0.8, edgecolor="gray", lw=0.4)
    ax5a.axvline(0, color="black", lw=0.8)
    ax5a.set_yticks(y)
    ax5a.set_yticklabels(stat_df["cell_type"], fontsize=8)
    ax5a.set_xlabel("Δ Autophagy Score (SLE ACT − Control)")
    ax5a.set_title("SLE ACT vs Control\nautophagy score shift", fontweight="bold")
    # 标注显著性
    for yi, (_, row) in enumerate(stat_df.iterrows()):
        sig = "***" if row["p_act_vs_ctrl"] < 0.001 else "**" if row["p_act_vs_ctrl"] < 0.01 else "*" if row["p_act_vs_ctrl"] < 0.05 else ""
        if sig:
            x_pos = row["delta_act"] + (0.02 if row["delta_act"] >= 0 else -0.02)
            ax5a.text(x_pos, yi, sig, va="center", ha="left" if row["delta_act"] >= 0 else "right", fontsize=9)

    # SLE INACT vs Control
    colors5b = ["#FDB863" if d > 0 else "#92C5DE" for d in stat_df["delta_inact"]]
    ax5b.barh(y, stat_df["delta_inact"], color=colors5b, alpha=0.8, edgecolor="gray", lw=0.4)
    ax5b.axvline(0, color="black", lw=0.8)
    ax5b.set_yticks(y)
    ax5b.set_yticklabels(stat_df["cell_type"], fontsize=8)
    ax5b.set_xlabel("Δ Autophagy Score (SLE INACT − Control)")
    ax5b.set_title("SLE INACT vs Control\nautophagy score shift", fontweight="bold")

    fig5.suptitle("Autophagy Score by Cell Type\n(score = mean(UP genes) − mean(DOWN genes))",
                  fontsize=12, fontweight="bold")
    fig5.tight_layout()
    fig5.savefig(os.path.join(OUT_DIR, "Fig5_autophagy_score.pdf"), bbox_inches="tight")
    fig5.savefig(os.path.join(OUT_DIR, "Fig5_autophagy_score.png"), bbox_inches="tight")
    plt.close(fig5)

# ── Fig6: B细胞亚群专项（受体编辑相关）──────────────────────────────────────
print("  Fig6: B细胞亚群...")
b_mask = adata.obs["celltypist_high"].str.contains("B cell|B-cell|Plasma|ABC|Transitional",
                                                     case=False, na=False)
adata_b = adata[b_mask].copy()
print(f"  B细胞亚群: {adata_b.shape[0]} cells")
print(adata_b.obs["celltypist_high"].value_counts().to_string())

# RAG1/RAG2 (受体编辑标志) + 自噬基因
rag_genes = [g for g in ["RAG1", "RAG2", "AICDA", "TDT", "DNTT"] if g in adata.var_names]
b_sig_genes = avail_sig + rag_genes

if len(adata_b) >= 20 and b_sig_genes:
    b_celltypes = adata_b.obs["celltypist_high"].value_counts()
    b_celltypes = b_celltypes[b_celltypes >= 10].index.tolist()

    df_b_mean = mean_expr_table(adata_b[adata_b.obs["celltypist_high"].isin(b_celltypes)],
                                 b_sig_genes, "celltypist_high")

    fig6, ax6 = plt.subplots(figsize=(len(b_sig_genes) * 0.8 + 1, len(df_b_mean) * 0.5 + 1))
    if not df_b_mean.empty:
        vmax6 = np.nanpercentile(df_b_mean.values, 95)
        im6 = ax6.imshow(df_b_mean.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax6)
        ax6.set_xticks(range(len(b_sig_genes)))
        ax6.set_xticklabels(b_sig_genes, rotation=45, ha="right", fontsize=8)
        ax6.set_yticks(range(len(df_b_mean)))
        ax6.set_yticklabels(df_b_mean.index, fontsize=8)
        if rag_genes:
            ax6.axvline(len(avail_sig) - 0.5, color="navy", lw=1.5, linestyle="--")
            ax6.text(len(avail_sig) + len(rag_genes)/2 - 0.5, -1.2, "Receptor editing",
                     ha="center", fontsize=8, color="navy")
        plt.colorbar(im6, ax=ax6, shrink=0.5, label="Mean log1p expr")
        ax6.set_title("B cell subtypes: autophagy + receptor editing genes", fontweight="bold")
    fig6.tight_layout()
    fig6.savefig(os.path.join(OUT_DIR, "Fig6_Bcell_autophagy.pdf"), bbox_inches="tight")
    fig6.savefig(os.path.join(OUT_DIR, "Fig6_Bcell_autophagy.png"), bbox_inches="tight")
    plt.close(fig6)

# ── 保存注释后的 h5ad ────────────────────────────────────────────────────────
print("\n[6/6] 保存注释结果...")
adata.obs[["celltypist_high", "celltypist_thymus", "is_DNT",
           "cell_label", "autophagy_score"]].to_csv(
    os.path.join(OUT_DIR, "cell_annotations.csv"))

out_h5ad = os.path.join(OUT_DIR, "GSE189050_annotated.h5ad")
adata.write_h5ad(out_h5ad)
print(f"  h5ad: {out_h5ad}")

print("\n完成！输出文件:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    size  = os.path.getsize(fpath) / 1024**2
    print(f"  {f:<45} {size:.1f} MB")

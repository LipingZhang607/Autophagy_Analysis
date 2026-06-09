#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE189050 自噬-SLE深度分析 v2
任务：
  T1b.   17基因UMAP feature plot + violin（补充）
  T2.    精细T/B亚群鉴定（marker-based，最高分胜出）
  T2_dev. T细胞发育状态评估
  T5.    B细胞发育状态评估
  T6.    受体编辑基因 × 自噬关系
  T7.    SLE vs Control逐基因统计（BH校正 + 效应量）
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
ANNOTATED_H5AD = "/home/h3033/statics/WQ/output/celltypist_autophagy/GSE189050_annotated.h5ad"
OUT_DIR        = "/home/h3033/statics/WQ/output/autophagy_v2"
os.makedirs(OUT_DIR, exist_ok=True)

sc.settings.set_figure_params(dpi=150, fontsize=10, facecolor="white")
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

# ── 基因集定义 ─────────────────────────────────────────────────────────────────
UP_GENES   = ["FOS","CCL2","EIF2AK2","DDIT3","TNFSF10","PPP1R15A",
               "NAMPT","FAS","MAPK3","CASP1"]
DOWN_GENES = ["ST13","BAG3","ITGA6","FOXO3","BNIP3L","BCL2L1","BAG1"]
SIG_GENES  = UP_GENES + DOWN_GENES

# B细胞发育阶段标志基因集
BCELL_DEV = {
    "Pro/Pre-B":     ["CD34","MME","VPREB1","VPREB3","IGLL1","RAG1","RAG2","DNTT"],
    "Immature B":    ["IGHM","MME","CD24","CD38","RAG1","CD19"],
    "Transitional B":["IGHM","IGHD","CD24","CD38","TCL1A","MME"],
    "Naive B":       ["IGHM","IGHD","CD27","TCL1A","FCER2","SELL"],
    "Memory B":      ["CD27","CD38","IGHG1","IGHA1","CD80","CXCR3"],
    "ABC":           ["ITGAX","TBX21","CR2","FCRL4","FCRL5","CXCR3"],
    "Plasmablast":   ["CD38","MZB1","XBP1","PRDM1","JCHAIN","SDC1"],
    "Plasma cell":   ["SDC1","PRDM1","MZB1","XBP1","JCHAIN","CD38"],
}

# T细胞亚型标志基因（用于亚型分配）
TCELL_MARKERS = {
    "Th1":         ["TBX21","IFNG","CXCR3","IL12RB2"],
    "Th2":         ["GATA3","IL4","IL13","CCR4"],
    "Th17":        ["RORC","IL17A","IL17F","CCR6"],
    "Treg":        ["FOXP3","IL2RA","CTLA4","IKZF2"],
    "Tfh":         ["CXCR5","BCL6","PDCD1","IL21","ICOS"],
    "CD8 effector":["GZMB","PRF1","NKG7","GNLY"],
    "CD8 exhaust": ["PDCD1","TIGIT","LAG3","HAVCR2","TOX"],
    "γδ T":        ["TRDC","TRGC1","TRGC2"],
}

# T细胞发育 / 分化状态基因集
TCELL_DEV = {
    "Naive T":       ["CCR7","SELL","TCF7","LEF1","IL7R","S1PR1"],
    "Central Memory":["CCR7","CD27","IL7R","TCF7","SELL"],
    "Effector Memory":["GZMK","CX3CR1","KLRG1","S1PR5","ITGAM"],
    "TEMRA":         ["CX3CR1","GZMB","PRF1","KLRG1","FCGR3A","S1PR5"],
    "Exhausted":     ["PDCD1","TIGIT","LAG3","HAVCR2","TOX","ENTPD1","CTLA4"],
    "Early activation":["CD69","CD44","HLA-DRA","CD38","MKI67"],
}

# 受体编辑相关基因
EDIT_GENES = ["RAG1","RAG2","DNTT","AICDA","APEX1","UNG","XRCC6","XRCC5"]

# ── 工具函数 ───────────────────────────────────────────────────────────────────
def get_expr(adata_sub, gene):
    if gene not in adata_sub.var_names:
        return None
    idx = list(adata_sub.var_names).index(gene)
    x = adata_sub.X[:, idx]
    return np.asarray(x.todense()).ravel() if issparse(x) else np.asarray(x).ravel()

def module_score(adata_sub, genes):
    """基因集均值评分（仅使用可用基因）"""
    exprs = [get_expr(adata_sub, g) for g in genes if g in adata_sub.var_names]
    return np.mean(exprs, axis=0) if exprs else np.zeros(adata_sub.shape[0])

def autophagy_score(adata_sub):
    up = module_score(adata_sub, UP_GENES)
    dn = module_score(adata_sub, DOWN_GENES)
    return up - dn

def cohens_d(a, b):
    """Cohen's d 效应量"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

def pval_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def add_stat_brackets(ax, data_list, positions, y_start=None, h_ratio=0.05,
                      comparisons=None, fontsize=8):
    """
    在violin/bar图上标注显著性括号。
    data_list: list of arrays (每组数据)
    positions: x轴位置列表（对应data_list）
    comparisons: [(i, j), ...] 要比较的组对索引，默认所有相邻对
    """
    if comparisons is None:
        comparisons = [(i, i+1) for i in range(len(data_list)-1)]

    all_vals = np.concatenate([d for d in data_list if len(d)])
    y_max = np.nanmax(all_vals) if len(all_vals) else 1.0
    y_range = np.nanmax(all_vals) - np.nanmin(all_vals) if len(all_vals) else 1.0
    h = y_range * h_ratio
    current_y = y_max + h if y_start is None else y_start

    for (i, j) in comparisons:
        if i >= len(data_list) or j >= len(data_list):
            continue
        a, b = data_list[i], data_list[j]
        if len(a) < 3 or len(b) < 3:
            continue
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        stars = pval_to_stars(p)
        x1, x2 = positions[i], positions[j]
        ax.plot([x1, x1, x2, x2], [current_y, current_y+h, current_y+h, current_y],
                lw=0.8, color="black")
        ax.text((x1+x2)/2, current_y+h, stars, ha="center", va="bottom",
                fontsize=fontsize, color="black")
        current_y += h * 3.5
    ax.set_ylim(top=current_y + h)

# ═══════════════════════════════════════════════════════════════════════════════
# 加载已注释数据
# ═══════════════════════════════════════════════════════════════════════════════
print("[0] 加载已注释 h5ad...")
adata = sc.read_h5ad(ANNOTATED_H5AD)
# 数据已经 normalize_total + log1p（上次脚本处理过）
print(f"  {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"  celltypist_high: {adata.obs['celltypist_high'].nunique()} types")

avail_sig = [g for g in SIG_GENES  if g in adata.var_names]
avail_up  = [g for g in UP_GENES   if g in adata.var_names]
avail_dn  = [g for g in DOWN_GENES if g in adata.var_names]

COND_COLORS = {"Control": "#4393C3", "SLE INACT": "#FDB863", "SLE ACT": "#D6604D"}

# ═══════════════════════════════════════════════════════════════════════════════
# T1b. 17基因 UMAP feature plot + violin
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[T1b] 17基因可视化...")

# UMAP feature plots（4列排列）
avail_sig = [g for g in SIG_GENES if g in adata.var_names]
ncols = 4
nrows = int(np.ceil(len(avail_sig) / ncols))
fig_fp, axes_fp = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
axes_fp = axes_fp.ravel()
for i, gene in enumerate(avail_sig):
    sc.pl.embedding(adata, basis="X_umap_wnn", color=gene,
                    ax=axes_fp[i], show=False, frameon=False,
                    color_map="RdYlBu_r", vmin=0,
                    title=f"{'↑' if gene in UP_GENES else '↓'} {gene}")
    axes_fp[i].set_xlabel(""); axes_fp[i].set_ylabel("")
for j in range(i+1, len(axes_fp)):
    axes_fp[j].set_visible(False)
fig_fp.suptitle("SLE-associated autophagy genes — UMAP expression", fontsize=13, fontweight="bold")
fig_fp.tight_layout()
fig_fp.savefig(os.path.join(OUT_DIR, "T1b_feature_plots.pdf"), bbox_inches="tight")
fig_fp.savefig(os.path.join(OUT_DIR, "T1b_feature_plots.png"), bbox_inches="tight")
plt.close(fig_fp)

# Violin plots（按主要细胞类型）
print("  violin...")
# 用粗分类 + DNT合并标签
adata.obs["plot_label"] = adata.obs["celltypist_high"].astype(str)
adata.obs.loc[adata.obs["is_DNT"], "plot_label"] = "DNT"
ct_order = adata.obs["plot_label"].value_counts()
ct_order = ct_order[ct_order >= 50].index.tolist()

fig_vl, axes_vl = plt.subplots(len(avail_sig), 1, figsize=(14, len(avail_sig) * 1.8))
for i, gene in enumerate(avail_sig):
    sc.pl.violin(adata[adata.obs["plot_label"].isin(ct_order)],
                 keys=gene, groupby="plot_label", order=ct_order,
                 ax=axes_vl[i], show=False, stripplot=False,
                 rotation=45)
    axes_vl[i].set_title(f"{'↑' if gene in UP_GENES else '↓'} {gene}", fontsize=9, fontweight="bold")
    axes_vl[i].set_xlabel("")
    if i < len(avail_sig) - 1:
        axes_vl[i].set_xticklabels([])
fig_vl.suptitle("17 autophagy genes — violin by cell type", fontsize=12, fontweight="bold")
fig_vl.tight_layout()
fig_vl.savefig(os.path.join(OUT_DIR, "T1b_violin.pdf"), bbox_inches="tight")
fig_vl.savefig(os.path.join(OUT_DIR, "T1b_violin.png"), bbox_inches="tight")
plt.close(fig_vl)

# ═══════════════════════════════════════════════════════════════════════════════
# T2. 精细T/B亚群鉴定（marker评分法）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[T2] 精细T/B亚群鉴定...")

# ── T细胞子集 ─────────────────────────────────────────────────────────────────
t_mask = adata.obs["celltypist_high"].isin(["T cells","ILC"]) | adata.obs["is_DNT"]
adata_t = adata[t_mask].copy()
print(f"  T/ILC/DNT: {len(adata_t)} cells")

# 为每个T亚型计算marker评分
for subtype, markers in TCELL_MARKERS.items():
    avail_m = [g for g in markers if g in adata_t.var_names]
    if avail_m:
        adata_t.obs[f"score_{subtype}"] = module_score(adata_t, avail_m)

# 加入CD4/CD8/CD3表达
for gene in ["CD4","CD8A","CD8B","CD3E"]:
    e = get_expr(adata_t, gene)
    if e is not None:
        adata_t.obs[f"expr_{gene}"] = e

# 根据marker评分 + CD4/CD8 分配T亚型（最高分胜出，不设硬阈值）
def assign_t_subtype(row):
    if row.get("is_DNT", False):
        return "DNT"
    cd4 = row.get("expr_CD4", 0)
    cd8 = row.get("expr_CD8A", 0) + row.get("expr_CD8B", 0)
    # γδ T 优先判断
    if row.get("score_γδ T", 0) > 0.3:
        return "γδ T"
    if cd8 > cd4:
        exh = row.get("score_CD8 exhaust", 0)
        eff = row.get("score_CD8 effector", 0)
        return "CD8 exhausted" if exh > eff else "CD8 effector"
    if cd4 > 0.2:
        cd4_types = {s: row.get(f"score_{s}", 0) for s in ["Th1","Th2","Th17","Treg","Tfh"]}
        best, best_score = max(cd4_types.items(), key=lambda x: x[1])
        return best if best_score > 0.1 else "CD4 other"
    # 兜底：用全部类型最高分
    all_scores = {s: row.get(f"score_{s}", 0) for s in TCELL_MARKERS}
    best, best_score = max(all_scores.items(), key=lambda x: x[1])
    return best if best_score > 0.1 else "T other"

adata_t.obs["t_subtype"] = adata_t.obs.apply(assign_t_subtype, axis=1)
print("  T亚型分布:")
print(adata_t.obs["t_subtype"].value_counts().to_string())

# 写回主 adata
adata.obs["t_subtype"] = "Non-T"
adata.obs.loc[adata_t.obs.index, "t_subtype"] = adata_t.obs["t_subtype"]

# ── B细胞子集 ─────────────────────────────────────────────────────────────────
b_mask = adata.obs["celltypist_high"].isin(["B cells","Plasma cells"])
adata_b = adata[b_mask].copy()
print(f"\n  B cells: {len(adata_b)} cells")

# 使用已有 fine_cell_type 作为基础，加marker评分细化
for stage, markers in BCELL_DEV.items():
    avail_m = [g for g in markers if g in adata_b.var_names]
    if avail_m:
        adata_b.obs[f"dev_{stage}"] = module_score(adata_b, avail_m)

# 关键基因表达
for gene in ["CD27","CD38","IGHM","IGHD","IGHG1","IGHA1","SDC1",
             "TCL1A","ITGAX","PRDM1","MZB1"]:
    e = get_expr(adata_b, gene)
    if e is not None:
        adata_b.obs[f"expr_{gene}"] = e

def assign_b_subtype(row):
    prdm1 = row.get("expr_PRDM1", 0)
    sdc1  = row.get("expr_SDC1", 0)
    mzb1  = row.get("expr_MZB1", 0)
    cd27  = row.get("expr_CD27", 0)
    itgax = row.get("expr_ITGAX", 0)
    tbx21 = row.get("dev_ABC", 0)
    ighm  = row.get("expr_IGHM", 0)
    ighd  = row.get("expr_IGHD", 0)
    ighg  = row.get("expr_IGHG1", 0)
    igha  = row.get("expr_IGHA1", 0)
    tcl1a = row.get("expr_TCL1A", 0)

    if prdm1 > 0.5 and sdc1 > 0.3:
        return "Plasma cell"
    if mzb1 > 0.5 or (prdm1 > 0.3 and sdc1 < 0.3):
        return "Plasmablast"
    if itgax > 0.3 and cd27 < 0.3:
        return "ABC (age-associated)"
    if cd27 > 0.5 and (ighg > 0.3 or igha > 0.3):
        return "Switched Memory B"
    if cd27 > 0.5 and ighm > 0.3:
        return "IgM Memory B"
    if tcl1a > 0.5 and ighm > 0.3 and ighd > 0.3:
        return "Naive B"
    if ighm > 0.3 and row.get("dev_Transitional B", 0) > row.get("dev_Naive B", 0):
        return "Transitional B"
    return "B other"

adata_b.obs["b_subtype"] = adata_b.obs.apply(assign_b_subtype, axis=1)
print("  B亚型分布:")
print(adata_b.obs["b_subtype"].value_counts().to_string())

adata.obs["b_subtype"] = "Non-B"
adata.obs.loc[adata_b.obs.index, "b_subtype"] = adata_b.obs["b_subtype"]

# 合并为最终精细标签
adata.obs["fine_label"] = adata.obs["celltypist_high"].astype(str)
adata.obs.loc[adata_t.obs.index, "fine_label"] = adata_t.obs["t_subtype"].values
adata.obs.loc[adata_b.obs.index, "fine_label"] = adata_b.obs["b_subtype"].values

# UMAP 精细标签
fig_t2, axes_t2 = plt.subplots(1, 2, figsize=(22, 8))
sc.pl.embedding(adata, basis="X_umap_wnn", color="fine_label",
                ax=axes_t2[0], show=False, frameon=False,
                title="Fine cell type labels", legend_loc="right margin")
sc.pl.embedding(adata_t, basis="X_umap_wnn", color="t_subtype",
                ax=axes_t2[1], show=False, frameon=False,
                title="T cell subtypes", legend_loc="right margin")
fig_t2.tight_layout()
fig_t2.savefig(os.path.join(OUT_DIR, "T2_fine_subtypes_UMAP.pdf"), bbox_inches="tight")
fig_t2.savefig(os.path.join(OUT_DIR, "T2_fine_subtypes_UMAP.png"), bbox_inches="tight")
plt.close(fig_t2)

# T/B 亚型 marker heatmap
fig_mk, axes_mk = plt.subplots(1, 2, figsize=(20, 10))
for ax, (adata_sub, subtype_col, marker_dict, title) in zip(
    axes_mk,
    [(adata_t, "t_subtype", TCELL_MARKERS, "T cell subtypes"),
     (adata_b, "b_subtype", BCELL_DEV,    "B cell developmental stages")]
):
    all_m = [g for genes in marker_dict.values() for g in genes if g in adata_sub.var_names]
    all_m = list(dict.fromkeys(all_m))
    groups = adata_sub.obs[subtype_col].value_counts()
    groups = groups[groups >= 10].index.tolist()
    mat = pd.DataFrame(index=groups, columns=all_m, dtype=float)
    for g in groups:
        sub_g = adata_sub[adata_sub.obs[subtype_col] == g]
        for gene in all_m:
            e = get_expr(sub_g, gene)
            mat.loc[g, gene] = e.mean() if e is not None else np.nan
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=np.nanpercentile(mat.values.astype(float), 95))
    ax.set_xticks(range(len(all_m)))
    ax.set_xticklabels(all_m, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.5, label="Mean log1p expr")
    ax.set_title(title, fontweight="bold")
fig_mk.tight_layout()
fig_mk.savefig(os.path.join(OUT_DIR, "T2_marker_heatmap.pdf"), bbox_inches="tight")
fig_mk.savefig(os.path.join(OUT_DIR, "T2_marker_heatmap.png"), bbox_inches="tight")
plt.close(fig_mk)

# ═══════════════════════════════════════════════════════════════════════════════
# T2_dev. T细胞发育 / 分化状态评估
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[T2_dev] T细胞发育状态评估...")

# 为每个T细胞计算发育状态评分
for state, markers in TCELL_DEV.items():
    avail_m = [g for g in markers if g in adata_t.var_names]
    adata_t.obs[f"dev_{state}"] = module_score(adata_t, avail_m) if avail_m else 0.0

# 最高分分配发育状态
dev_states = list(TCELL_DEV.keys())
dev_score_cols = [f"dev_{s}" for s in dev_states]
adata_t.obs["t_dev_state"] = adata_t.obs[dev_score_cols].idxmax(axis=1).str.replace("dev_", "")

print("  T发育状态分布:")
print(adata_t.obs["t_dev_state"].value_counts().to_string())

# 写回主 adata
adata.obs["t_dev_state"] = "Non-T"
adata.obs.loc[adata_t.obs.index, "t_dev_state"] = adata_t.obs["t_dev_state"]

# ── 图1：T发育状态 UMAP + 3组比例 ────────────────────────────────────────────
fig_tdev, axes_tdev = plt.subplots(1, 3, figsize=(24, 7))

sc.pl.embedding(adata_t, basis="X_umap_wnn", color="t_dev_state",
                ax=axes_tdev[0], show=False, frameon=False,
                title="T cell differentiation states", legend_loc="right margin")

# 各发育状态在3组中的比例
t_dev_prop = adata_t.obs.groupby(["classification","t_dev_state"]).size().unstack(fill_value=0)
t_dev_prop = t_dev_prop.div(t_dev_prop.sum(axis=1), axis=0) * 100
t_dev_prop.T.plot(kind="bar", ax=axes_tdev[1],
                  color=[COND_COLORS.get(c,"gray") for c in t_dev_prop.index],
                  alpha=0.8, edgecolor="gray", lw=0.3)
axes_tdev[1].set_xlabel("")
axes_tdev[1].set_ylabel("% of T cells")
axes_tdev[1].set_title("T differentiation state proportions\nby condition", fontweight="bold")
axes_tdev[1].legend(title="Condition", frameon=False, fontsize=8)
axes_tdev[1].tick_params(axis="x", rotation=30)

# 各发育状态评分在3组中的均值（grouped bar）
dev_score_records = []
for state in dev_states:
    col = f"dev_{state}"
    for cond in ["Control","SLE INACT","SLE ACT"]:
        vals = adata_t[adata_t.obs["classification"]==cond].obs[col].values
        dev_score_records.append({"state": state, "condition": cond,
                                   "mean": vals.mean(), "sem": vals.std()/np.sqrt(len(vals))})
df_tdev = pd.DataFrame(dev_score_records)

x_td = np.arange(len(dev_states))
w_td = 0.25
for ci, (cond, color) in enumerate(COND_COLORS.items()):
    means = [df_tdev[(df_tdev["state"]==s) & (df_tdev["condition"]==cond)]["mean"].values[0]
             for s in dev_states]
    sems  = [df_tdev[(df_tdev["state"]==s) & (df_tdev["condition"]==cond)]["sem"].values[0]
             for s in dev_states]
    axes_tdev[2].bar(x_td + (ci-1)*w_td, means, w_td,
                     yerr=sems, capsize=2,
                     label=cond, color=color, alpha=0.8)
axes_tdev[2].set_xticks(x_td)
axes_tdev[2].set_xticklabels(dev_states, rotation=30, ha="right", fontsize=9)
axes_tdev[2].set_ylabel("Mean module score")
axes_tdev[2].set_title("T differentiation state scores\nby condition", fontweight="bold")
axes_tdev[2].legend(frameon=False, fontsize=8)

fig_tdev.tight_layout()
fig_tdev.savefig(os.path.join(OUT_DIR, "T2_dev_Tcell_states.pdf"), bbox_inches="tight")
fig_tdev.savefig(os.path.join(OUT_DIR, "T2_dev_Tcell_states.png"), bbox_inches="tight")
plt.close(fig_tdev)

# ── 图2：T发育状态 × T亚型交叉热图 ──────────────────────────────────────────
cross = pd.crosstab(adata_t.obs["t_subtype"], adata_t.obs["t_dev_state"])
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

fig_cross, ax_cross = plt.subplots(figsize=(len(dev_states)*1.2+1, len(cross_pct)*0.6+1))
im_c = ax_cross.imshow(cross_pct.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
ax_cross.set_xticks(range(len(cross_pct.columns)))
ax_cross.set_xticklabels(cross_pct.columns, rotation=30, ha="right", fontsize=9)
ax_cross.set_yticks(range(len(cross_pct)))
ax_cross.set_yticklabels(cross_pct.index, fontsize=9)
for ri in range(cross_pct.shape[0]):
    for ci in range(cross_pct.shape[1]):
        val = cross_pct.values[ri, ci]
        if val > 5:
            ax_cross.text(ci, ri, f"{val:.0f}%", ha="center", va="center", fontsize=7)
plt.colorbar(im_c, ax=ax_cross, shrink=0.5, label="% within subtype")
ax_cross.set_title("T subtype × differentiation state (%)", fontweight="bold")
fig_cross.tight_layout()
fig_cross.savefig(os.path.join(OUT_DIR, "T2_dev_subtype_cross.pdf"), bbox_inches="tight")
fig_cross.savefig(os.path.join(OUT_DIR, "T2_dev_subtype_cross.png"), bbox_inches="tight")
plt.close(fig_cross)

# ── 图3：各发育状态内的自噬评分 × 3组 ───────────────────────────────────────
adata_t.obs["autophagy_score"] = autophagy_score(adata_t)

fig_tatg, axes_tatg = plt.subplots(1, len(dev_states), figsize=(len(dev_states)*3, 5), sharey=True)
for ai, state in enumerate(dev_states):
    sub_state = adata_t[adata_t.obs["t_dev_state"] == state]
    if len(sub_state) < 10:
        axes_tatg[ai].set_title(state, fontsize=8)
        continue
    data_v = [sub_state[sub_state.obs["classification"]==c].obs["autophagy_score"].values
              for c in ["Control","SLE INACT","SLE ACT"]]
    data_v = [d for d in data_v if len(d) >= 3]
    if not data_v:
        continue
    parts = axes_tatg[ai].violinplot(data_v, showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], list(COND_COLORS.values())[:len(data_v)]):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    axes_tatg[ai].set_xticks(range(1, len(data_v)+1))
    axes_tatg[ai].set_xticklabels(["Ctrl","INACT","ACT"][:len(data_v)], fontsize=7, rotation=30)
    axes_tatg[ai].set_title(f"{state}\n(n={len(sub_state)})", fontsize=8, fontweight="bold")
    add_stat_brackets(axes_tatg[ai], data_v, list(range(1, len(data_v)+1)),
                      comparisons=[(0,2),(0,1),(1,2)], fontsize=7)
axes_tatg[0].set_ylabel("Autophagy score")
fig_tatg.suptitle("T cell differentiation states: autophagy score by condition",
                   fontsize=11, fontweight="bold")
fig_tatg.tight_layout()
fig_tatg.savefig(os.path.join(OUT_DIR, "T2_dev_autophagy_by_state.pdf"), bbox_inches="tight")
fig_tatg.savefig(os.path.join(OUT_DIR, "T2_dev_autophagy_by_state.png"), bbox_inches="tight")
plt.close(fig_tatg)

df_tdev.to_csv(os.path.join(OUT_DIR, "T2_dev_Tcell_state_scores.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# T5. B细胞发育状态评估
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[T5] B细胞发育状态评估...")

# 各发育阶段评分 × 3组条件
stage_cols = [f"dev_{s}" for s in BCELL_DEV.keys()]
for stage, markers in BCELL_DEV.items():
    avail_m = [g for g in markers if g in adata_b.var_names]
    adata_b.obs[f"dev_{stage}"] = module_score(adata_b, avail_m) if avail_m else 0.0

dev_records = []
for stage in BCELL_DEV:
    col = f"dev_{stage}"
    if col not in adata_b.obs.columns:
        continue
    for cond in ["Control","SLE INACT","SLE ACT"]:
        sub = adata_b[adata_b.obs["classification"] == cond]
        vals = sub.obs[col].values
        dev_records.append({"stage": stage, "condition": cond,
                             "mean": vals.mean(), "sem": vals.std()/np.sqrt(len(vals))})

df_dev = pd.DataFrame(dev_records)
df_dev.to_csv(os.path.join(OUT_DIR, "T5_Bcell_dev_scores.csv"), index=False)

fig_dev, axes_dev = plt.subplots(1, 2, figsize=(18, 6))

# 左：各发育阶段在3组的评分（grouped bar）
stages = list(BCELL_DEV.keys())
x = np.arange(len(stages))
width = 0.25
for ci, (cond, color) in enumerate(COND_COLORS.items()):
    means = [df_dev[(df_dev["stage"]==s) & (df_dev["condition"]==cond)]["mean"].values
             for s in stages]
    means = [m[0] if len(m) else 0 for m in means]
    axes_dev[0].bar(x + ci*width - width, means, width, label=cond, color=color, alpha=0.8)
axes_dev[0].set_xticks(x)
axes_dev[0].set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
axes_dev[0].set_ylabel("Mean module score")
axes_dev[0].set_title("B cell developmental stage scores by condition", fontweight="bold")
axes_dev[0].legend(frameon=False)

# 逐阶段标注 ACT vs Control 的P值
y_ann = axes_dev[0].get_ylim()[1]
for si, stage in enumerate(stages):
    col = f"dev_{stage}"
    if col not in adata_b.obs.columns:
        continue
    ctrl_v = adata_b[adata_b.obs["classification"]=="Control"].obs[col].values
    act_v  = adata_b[adata_b.obs["classification"]=="SLE ACT"].obs[col].values
    if len(ctrl_v) < 5 or len(act_v) < 5:
        continue
    _, p = stats.mannwhitneyu(act_v, ctrl_v, alternative="two-sided")
    stars = pval_to_stars(p)
    if stars != "ns":
        x_bar = x[si] + width   # ACT bar 位置
        axes_dev[0].text(x_bar, y_ann * 1.02, stars, ha="center", fontsize=8, color="#D6604D")

# 右：各B亚型在3组中的细胞比例
b_type_prop = adata_b.obs.groupby(["classification","b_subtype"]).size().unstack(fill_value=0)
b_type_prop = b_type_prop.div(b_type_prop.sum(axis=1), axis=0) * 100
b_type_prop.T.plot(kind="bar", ax=axes_dev[1],
                   color=[COND_COLORS.get(c, "gray") for c in b_type_prop.index],
                   alpha=0.8, edgecolor="gray", lw=0.3)
axes_dev[1].set_xlabel("")
axes_dev[1].set_ylabel("% of B cells")
axes_dev[1].set_title("B cell subtype proportions by condition", fontweight="bold")
axes_dev[1].legend(title="Condition", frameon=False, fontsize=8)
axes_dev[1].tick_params(axis="x", rotation=30)

fig_dev.tight_layout()
fig_dev.savefig(os.path.join(OUT_DIR, "T5_Bcell_development.pdf"), bbox_inches="tight")
fig_dev.savefig(os.path.join(OUT_DIR, "T5_Bcell_development.png"), bbox_inches="tight")
plt.close(fig_dev)

# ═══════════════════════════════════════════════════════════════════════════════
# T6. 受体编辑基因 × 自噬关系
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[T6] 受体编辑 × 自噬分析...")

avail_edit = [g for g in EDIT_GENES if g in adata_b.var_names]
print(f"  受体编辑基因可用: {avail_edit}")

# 计算B细胞的自噬评分
adata_b.obs["autophagy_score"] = autophagy_score(adata_b)
adata_b.obs["edit_score"]      = module_score(adata_b, avail_edit)

for gene in avail_edit:
    e = get_expr(adata_b, gene)
    if e is not None:
        adata_b.obs[f"edit_{gene}"] = e

# 相关性：各受体编辑基因 vs 自噬评分（B细胞内，按B亚型）
corr_records = []
b_subtypes_valid = adata_b.obs["b_subtype"].value_counts()
b_subtypes_valid = b_subtypes_valid[b_subtypes_valid >= 20].index.tolist()

for bst in b_subtypes_valid:
    sub_b = adata_b[adata_b.obs["b_subtype"] == bst]
    atg_s = sub_b.obs["autophagy_score"].values
    for gene in avail_edit:
        eg = sub_b.obs.get(f"edit_{gene}", pd.Series([np.nan]*len(sub_b))).values
        if np.std(eg) < 1e-6 or np.std(atg_s) < 1e-6:
            continue
        r, p = stats.spearmanr(eg, atg_s)
        corr_records.append({"b_subtype": bst, "edit_gene": gene,
                              "spearman_r": r, "p_value": p, "n": len(sub_b)})

df_corr = pd.DataFrame(corr_records)
if not df_corr.empty:
    # BH校正
    _, df_corr["p_adj"], _, _ = multipletests(df_corr["p_value"], method="fdr_bh")
    df_corr.to_csv(os.path.join(OUT_DIR, "T6_edit_autophagy_corr.csv"), index=False)

    # 相关性热图
    pivot_r = df_corr.pivot(index="b_subtype", columns="edit_gene", values="spearman_r")
    pivot_p = df_corr.pivot(index="b_subtype", columns="edit_gene", values="p_adj")

    fig_corr, axes_corr = plt.subplots(1, 2, figsize=(16, max(len(b_subtypes_valid)*0.8+1, 5)))

    vabs = np.nanmax(np.abs(pivot_r.values))
    im_r = axes_corr[0].imshow(pivot_r.values, aspect="auto", cmap="RdBu_r",
                                vmin=-vabs, vmax=vabs)
    axes_corr[0].set_xticks(range(len(pivot_r.columns)))
    axes_corr[0].set_xticklabels(pivot_r.columns, rotation=45, ha="right", fontsize=9)
    axes_corr[0].set_yticks(range(len(pivot_r)))
    axes_corr[0].set_yticklabels(pivot_r.index, fontsize=9)
    # 标注显著性
    for ri in range(pivot_r.shape[0]):
        for ci in range(pivot_r.shape[1]):
            p_val = pivot_p.values[ri, ci] if not np.isnan(pivot_p.values[ri, ci]) else 1
            if p_val < 0.05:
                axes_corr[0].text(ci, ri, "*" if p_val < 0.05 else "",
                                  ha="center", va="center", fontsize=10)
    plt.colorbar(im_r, ax=axes_corr[0], shrink=0.6, label="Spearman r")
    axes_corr[0].set_title("Receptor editing genes vs\nautophagy score (Spearman r)", fontweight="bold")

    # 右：SLE ACT vs Control的受体编辑评分比较（各B亚型）
    edit_stat = []
    for bst in b_subtypes_valid:
        sub_b = adata_b[adata_b.obs["b_subtype"] == bst]
        ctrl_e = sub_b[sub_b.obs["classification"]=="Control"].obs["edit_score"].values
        act_e  = sub_b[sub_b.obs["classification"]=="SLE ACT"].obs["edit_score"].values
        if len(ctrl_e) < 5 or len(act_e) < 5:
            continue
        _, p = stats.mannwhitneyu(act_e, ctrl_e, alternative="two-sided")
        edit_stat.append({"b_subtype": bst,
                           "delta": act_e.mean() - ctrl_e.mean(),
                           "p": p})
    df_edit_stat = pd.DataFrame(edit_stat).sort_values("delta")
    if not df_edit_stat.empty:
        colors_e = ["#D6604D" if d > 0 else "#4393C3" for d in df_edit_stat["delta"]]
        y_e = np.arange(len(df_edit_stat))
        axes_corr[1].barh(y_e, df_edit_stat["delta"], color=colors_e, alpha=0.8, edgecolor="gray", lw=0.4)
        axes_corr[1].axvline(0, color="black", lw=0.8)
        axes_corr[1].set_yticks(y_e)
        axes_corr[1].set_yticklabels(df_edit_stat["b_subtype"], fontsize=9)
        axes_corr[1].set_xlabel("Δ Receptor editing score (SLE ACT − Control)")
        axes_corr[1].set_title("Receptor editing activity\nin SLE vs Control", fontweight="bold")
        for yi, row in enumerate(df_edit_stat.itertuples()):
            sig = "***" if row.p < 0.001 else "**" if row.p < 0.01 else "*" if row.p < 0.05 else ""
            if sig:
                axes_corr[1].text(row.delta, yi, f" {sig}", va="center", fontsize=9)

    fig_corr.tight_layout()
    fig_corr.savefig(os.path.join(OUT_DIR, "T6_receptor_editing_autophagy.pdf"), bbox_inches="tight")
    fig_corr.savefig(os.path.join(OUT_DIR, "T6_receptor_editing_autophagy.png"), bbox_inches="tight")
    plt.close(fig_corr)

# ── BCR同型替代分析（obs里的CITE-seq isotype diff）────────────────────────────
igh_cols = [c for c in adata_b.obs.columns if "IGH" in c and ".diff" in c]
if igh_cols:
    print(f"  使用CITE-seq isotype diff列: {len(igh_cols)}个")
    # 将主要isotype表达（来自RNA）整合到B亚型比较
    isotype_genes = [g for g in ["IGHM","IGHD","IGHG1","IGHG2","IGHG3","IGHG4",
                                  "IGHA1","IGHA2","IGHE"] if g in adata_b.var_names]
    iso_records = []
    for bst in b_subtypes_valid:
        for cond in ["Control","SLE INACT","SLE ACT"]:
            sub = adata_b[(adata_b.obs["b_subtype"]==bst) &
                          (adata_b.obs["classification"]==cond)]
            if len(sub) < 5:
                continue
            row = {"b_subtype": bst, "condition": cond}
            for g in isotype_genes:
                e = get_expr(sub, g)
                row[g] = e.mean() if e is not None else np.nan
            iso_records.append(row)

    df_iso = pd.DataFrame(iso_records)
    df_iso.to_csv(os.path.join(OUT_DIR, "T6_BCR_isotype_by_subtype.csv"), index=False)

    # 热图
    iso_pivot = df_iso.pivot_table(index=["b_subtype","condition"],
                                    columns=[], values=isotype_genes).reset_index()
    fig_iso, ax_iso = plt.subplots(figsize=(len(isotype_genes)*1.2+2, len(b_subtypes_valid)*1.5+1))
    plot_mat = df_iso.set_index(["b_subtype","condition"])[isotype_genes]
    if not plot_mat.empty:
        im_iso = ax_iso.imshow(plot_mat.values.astype(float), aspect="auto",
                                cmap="YlOrRd", vmin=0)
        ax_iso.set_xticks(range(len(isotype_genes)))
        ax_iso.set_xticklabels(isotype_genes, rotation=45, ha="right")
        ax_iso.set_yticks(range(len(plot_mat)))
        ax_iso.set_yticklabels([f"{b}|{c}" for b, c in plot_mat.index], fontsize=7)
        plt.colorbar(im_iso, ax=ax_iso, shrink=0.4, label="Mean log1p expr")
        ax_iso.set_title("BCR isotype expression by B subtype & condition", fontweight="bold")
    fig_iso.tight_layout()
    fig_iso.savefig(os.path.join(OUT_DIR, "T6_BCR_isotype_heatmap.pdf"), bbox_inches="tight")
    fig_iso.savefig(os.path.join(OUT_DIR, "T6_BCR_isotype_heatmap.png"), bbox_inches="tight")
    plt.close(fig_iso)

# ═══════════════════════════════════════════════════════════════════════════════
# T7. 统计分析：SLE vs Control 亚群特异性自噬变化
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[T7] 统计分析（逐基因 + BH校正 + 效应量）...")

# 使用精细标签
all_fine_types = adata.obs["fine_label"].value_counts()
all_fine_types = all_fine_types[all_fine_types >= 30].index.tolist()

stat_all = []
for ct in all_fine_types:
    sub_ct = adata[adata.obs["fine_label"] == ct]
    for cond_a, cond_b, label in [
        ("SLE ACT",   "Control", "ACT_vs_Ctrl"),
        ("SLE INACT", "Control", "INACT_vs_Ctrl"),
        ("SLE ACT",   "SLE INACT", "ACT_vs_INACT"),
    ]:
        sub_a = sub_ct[sub_ct.obs["classification"] == cond_a]
        sub_b = sub_ct[sub_ct.obs["classification"] == cond_b]
        if len(sub_a) < 10 or len(sub_b) < 10:
            continue
        for gene in avail_sig:
            ea = get_expr(sub_a, gene)
            eb = get_expr(sub_b, gene)
            if ea is None or eb is None:
                continue
            _, p = stats.mannwhitneyu(ea, eb, alternative="two-sided")
            d    = cohens_d(ea, eb)
            fc   = (ea.mean() + 1e-6) / (eb.mean() + 1e-6)
            stat_all.append({
                "cell_type": ct, "comparison": label, "gene": gene,
                "n_a": len(sub_a), "n_b": len(sub_b),
                "mean_a": ea.mean(), "mean_b": eb.mean(),
                "log2FC": np.log2(fc), "cohens_d": d, "p_value": p,
                "direction": "UP" if gene in UP_GENES else "DOWN"
            })

df_stat = pd.DataFrame(stat_all)
if not df_stat.empty:
    # BH 校正（全局）
    _, df_stat["p_adj"], _, _ = multipletests(df_stat["p_value"], method="fdr_bh")
    df_stat["sig"] = df_stat["p_adj"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns")
    df_stat = df_stat.sort_values(["comparison","cell_type","p_adj"])
    df_stat.to_csv(os.path.join(OUT_DIR, "T7_stat_results.csv"), index=False)
    print(f"  总记录数: {len(df_stat)}, 显著(p_adj<0.05): {(df_stat['p_adj']<0.05).sum()}")

    # ── 可视化1：气泡图（ACT vs Ctrl，基因 × 细胞类型）─────────────────────────
    df_act = df_stat[df_stat["comparison"] == "ACT_vs_Ctrl"].copy()
    pivot_fc  = df_act.pivot_table(index="cell_type", columns="gene", values="log2FC")
    pivot_sig = df_act.pivot_table(index="cell_type", columns="gene", values="p_adj")
    pivot_fc  = pivot_fc.reindex(columns=SIG_GENES)
    pivot_sig = pivot_sig.reindex(columns=SIG_GENES)

    fig_bub, ax_bub = plt.subplots(figsize=(len(SIG_GENES)*0.85+2, len(pivot_fc)*0.6+2))
    for ri, ct in enumerate(pivot_fc.index):
        for ci, gene in enumerate(pivot_fc.columns):
            fc_val = pivot_fc.loc[ct, gene]
            p_val  = pivot_sig.loc[ct, gene] if not pd.isna(pivot_sig.loc[ct, gene]) else 1.0
            if pd.isna(fc_val):
                continue
            size = max(5, -np.log10(p_val + 1e-10) * 30)
            color = "#D6604D" if fc_val > 0 else "#4393C3"
            ax_bub.scatter(ci, ri, s=size, c=color,
                           alpha=0.8, edgecolors="gray" if p_val < 0.05 else "none", lw=0.5)

    ax_bub.set_xticks(range(len(SIG_GENES)))
    ax_bub.set_xticklabels(SIG_GENES, rotation=45, ha="right", fontsize=9)
    ax_bub.axvline(len(UP_GENES) - 0.5, color="gray", lw=1, linestyle="--")
    ax_bub.set_yticks(range(len(pivot_fc)))
    ax_bub.set_yticklabels(pivot_fc.index, fontsize=8)
    ax_bub.set_title("SLE ACT vs Control: log2FC per gene × cell type\n(size=-log10 p_adj; edge=sig; red=up, blue=down)",
                      fontweight="bold")
    # 图例
    for sz_label, p_ex in [("p=0.05", 0.05), ("p=0.01", 0.01), ("p=0.001", 0.001)]:
        ax_bub.scatter([], [], s=-np.log10(p_ex+1e-10)*30, c="gray",
                       alpha=0.6, label=sz_label)
    ax_bub.legend(title="p_adj", frameon=False, fontsize=8, loc="lower right")
    fig_bub.tight_layout()
    fig_bub.savefig(os.path.join(OUT_DIR, "T7_bubble_ACT_vs_Ctrl.pdf"), bbox_inches="tight")
    fig_bub.savefig(os.path.join(OUT_DIR, "T7_bubble_ACT_vs_Ctrl.png"), bbox_inches="tight")
    plt.close(fig_bub)

    # ── 可视化2：Cohen's d 热图（SLE ACT vs Ctrl）────────────────────────────
    pivot_d = df_act.pivot_table(index="cell_type", columns="gene", values="cohens_d")
    pivot_d = pivot_d.reindex(columns=SIG_GENES)

    fig_cd, ax_cd = plt.subplots(figsize=(len(SIG_GENES)*0.85+2, len(pivot_d)*0.6+2))
    vabs = np.nanpercentile(np.abs(pivot_d.values), 95)
    im_cd = ax_cd.imshow(pivot_d.values.astype(float), aspect="auto",
                          cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax_cd.set_xticks(range(len(SIG_GENES)))
    ax_cd.set_xticklabels(SIG_GENES, rotation=45, ha="right", fontsize=9)
    ax_cd.axvline(len(UP_GENES) - 0.5, color="gray", lw=1, linestyle="--")
    ax_cd.set_yticks(range(len(pivot_d)))
    ax_cd.set_yticklabels(pivot_d.index, fontsize=8)
    # 标注显著性
    for ri, ct in enumerate(pivot_d.index):
        for ci, gene in enumerate(pivot_d.columns):
            p_val = pivot_sig.loc[ct, gene] if ct in pivot_sig.index else 1.0
            if not pd.isna(p_val) and p_val < 0.05:
                ax_cd.text(ci, ri, "*", ha="center", va="center", fontsize=8, color="white")
    plt.colorbar(im_cd, ax=ax_cd, shrink=0.5, label="Cohen's d")
    ax_cd.set_title("Effect size (Cohen's d): SLE ACT vs Control\n(* = p_adj < 0.05)",
                     fontweight="bold")
    fig_cd.tight_layout()
    fig_cd.savefig(os.path.join(OUT_DIR, "T7_cohens_d_heatmap.pdf"), bbox_inches="tight")
    fig_cd.savefig(os.path.join(OUT_DIR, "T7_cohens_d_heatmap.png"), bbox_inches="tight")
    plt.close(fig_cd)

    # ── 可视化3：DNT 和 ABC 专项 violin（最关注的亚群）─────────────────────────
    for focus_ct in ["DNT", "ABC (age-associated)", "Switched Memory B", "Naive B"]:
        sub_f = adata[adata.obs["fine_label"] == focus_ct]
        if len(sub_f) < 20:
            continue
        sub_f.obs["autophagy_score"] = autophagy_score(sub_f)

        fig_vf, axes_vf = plt.subplots(1, 2, figsize=(14, 5))

        # 自噬评分violin
        data_v = [sub_f[sub_f.obs["classification"]==c].obs["autophagy_score"].values
                  for c in ["Control","SLE INACT","SLE ACT"]]
        parts = axes_vf[0].violinplot(data_v, showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], COND_COLORS.values()):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        axes_vf[0].set_xticks([1,2,3])
        axes_vf[0].set_xticklabels(["Control","SLE INACT","SLE ACT"])
        axes_vf[0].set_ylabel("Autophagy score")
        axes_vf[0].set_title(f"{focus_ct}\nAutophagy score")
        add_stat_brackets(axes_vf[0], data_v, [1,2,3],
                          comparisons=[(0,2),(0,1),(1,2)], fontsize=8)

        # 17基因表达条形图
        gene_means = {}
        for cond in ["Control","SLE INACT","SLE ACT"]:
            sub_c = sub_f[sub_f.obs["classification"]==cond]
            gene_means[cond] = [get_expr(sub_c, g).mean()
                                 if get_expr(sub_c, g) is not None else 0
                                 for g in avail_sig]
        x_g = np.arange(len(avail_sig))
        w   = 0.25
        for ci, (cond, color) in enumerate(COND_COLORS.items()):
            axes_vf[1].bar(x_g + (ci-1)*w, gene_means[cond], w,
                           color=color, alpha=0.8, label=cond)
        axes_vf[1].axvline(len(avail_up) - 0.5, color="gray", lw=1, linestyle="--")
        axes_vf[1].set_xticks(x_g)
        axes_vf[1].set_xticklabels(avail_sig, rotation=45, ha="right", fontsize=8)
        axes_vf[1].set_ylabel("Mean log1p expr")
        axes_vf[1].set_title(f"{focus_ct}\n17 genes by condition")
        axes_vf[1].legend(frameon=False, fontsize=8)

        fig_vf.suptitle(f"Focus: {focus_ct} (n={len(sub_f)})", fontweight="bold")
        fig_vf.tight_layout()
        safe_name = focus_ct.replace(" ", "_").replace("(", "").replace(")", "")
        fig_vf.savefig(os.path.join(OUT_DIR, f"T7_focus_{safe_name}.pdf"), bbox_inches="tight")
        fig_vf.savefig(os.path.join(OUT_DIR, f"T7_focus_{safe_name}.png"), bbox_inches="tight")
        plt.close(fig_vf)

# ── 保存更新后的 adata ────────────────────────────────────────────────────────
print("\n[保存] 更新注释...")
adata.obs[["fine_label","t_subtype","b_subtype"]].to_csv(
    os.path.join(OUT_DIR, "fine_annotations.csv"))

print("\n完成！输出文件:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024**2
    print(f"  {f:<50} {size:.1f} MB")

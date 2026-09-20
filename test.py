#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终 KronRLS 训练 + 论文绘图（7:1.5:1.5 划分）
超参数：
  拓扑: WL_iter=2, WL/SP_ratio=0.3, α=0.1
  三元: λ_topo=0.6, λ_ecfp=0.1, λ_desc=0.3
输出：
  1. 模型文件（.pkl）
  2. 性能指标（.csv）
  3. 图5E-H（.png/.pdf）
  4. 绘图所需原始数据（.npz）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
import networkx as nx
import pickle
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
OUTPUT_DIR = r"E:\EDC\final_model_7_1.5_1.5"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)
CACHE_DIR = r"E:\EDC\cache_final"
os.makedirs(CACHE_DIR, exist_ok=True)

AFFINITY_FILE = r"E:\EDC\affinity.xlsx"
SMILES_FILE = r"E:\EDC\EDCs_SMILES.xlsx"

# ===== 最优超参数（调优结果） =====
OPTIMAL = {
    'wl_iter': 2,
    'wl_sp_ratio': 0.3,
    'alpha': 0.1,
    'lambda_topo': 0.5,
    'lambda_ecfp': 0.1,
    'lambda_desc': 0.4
}

ECFP_RADIUS = 2
ECFP_NBITS = 2048

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

print("="*70)
print("最终 KronRLS 训练 + 绘图准备")
print(f"超参数: {OPTIMAL}")
print("="*70)

# ============================================================
# 1. 图核函数
# ============================================================
def wl_kernel(G1, G2, n_iter=2):
    def wl_hash(graph, n_iter):
        labels = {node: data.get('label', '') for node, data in graph.nodes(data=True)}
        for _ in range(n_iter):
            new_labels = {}
            for node in graph.nodes():
                neighbor_labels = sorted([labels[n] for n in graph.neighbors(node)])
                new_label = str(labels[node]) + ''.join(neighbor_labels)
                new_labels[node] = new_label
            labels = new_labels
        return labels
    def label_histogram(labels):
        hist = defaultdict(int)
        for label in labels.values():
            hist[label] += 1
        return hist
    labels1 = wl_hash(G1, n_iter)
    labels2 = wl_hash(G2, n_iter)
    hist1 = label_histogram(labels1)
    hist2 = label_histogram(labels2)
    common = set(hist1.keys()) & set(hist2.keys())
    return sum(hist1[label] * hist2[label] for label in common)

def shortest_path_kernel(G1, G2):
    def all_pairs_shortest_path(graph):
        paths = dict(nx.all_pairs_shortest_path_length(graph))
        hist = defaultdict(int)
        for u in paths:
            for v, dist in paths[u].items():
                if u < v:
                    hist[dist] += 1
        return hist
    hist1 = all_pairs_shortest_path(G1)
    hist2 = all_pairs_shortest_path(G2)
    common = set(hist1.keys()) & set(hist2.keys())
    return sum(hist1[d] * hist2[d] for d in common)

def normalize_kernel(K):
    K_norm = np.zeros_like(K)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if K[i, i] > 0 and K[j, j] > 0:
                K_norm[i, j] = K[i, j] / np.sqrt(K[i, i] * K[j, j])
    return K_norm

def mol_to_graph(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G

# ============================================================
# 2. 加载数据
# ============================================================
print("\n加载数据...")
df_aff = pd.read_excel(AFFINITY_FILE, index_col=0)
protein_names = df_aff.columns.tolist()
n_proteins = len(protein_names)

df_smiles = pd.read_excel(SMILES_FILE)
cas_to_smiles = {}
for _, row in df_smiles.iterrows():
    cas = str(row['CAS']).strip()
    if cas.endswith('.log'):
        cas = cas[:-4]
    cas_to_smiles[cas] = row['SMILES']

molecule_cas = []
molecule_smiles = []
all_graphs = []

for cas in df_aff.index:
    cas_clean = str(cas).strip()
    if cas_clean.endswith('.log'):
        cas_clean = cas_clean[:-4]
    if cas_clean in cas_to_smiles:
        smi = cas_to_smiles[cas_clean]
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            molecule_cas.append(cas_clean)
            molecule_smiles.append(smi)
            all_graphs.append(mol_to_graph(mol))

n_mol = len(molecule_cas)
print(f"总分子数: {n_mol}")

# 7:1.5:1.5 划分
train_val_idx, test_idx = train_test_split(
    range(n_mol), test_size=TEST_RATIO, random_state=RANDOM_STATE, shuffle=True
)
val_ratio_from_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=val_ratio_from_train_val, random_state=RANDOM_STATE, shuffle=True
)

print(f"训练集: {len(train_idx)} ({len(train_idx)/n_mol*100:.1f}%)")
print(f"验证集: {len(val_idx)} ({len(val_idx)/n_mol*100:.1f}%)")
print(f"测试集: {len(test_idx)} ({len(test_idx)/n_mol*100:.1f}%)")

# ============================================================
# 3. 计算/加载核矩阵
# ============================================================
print("\n加载/计算核矩阵...")

# WL
wl_cache = os.path.join(CACHE_DIR, f"K_wl_iter{OPTIMAL['wl_iter']}_{n_mol}.npy")
if os.path.exists(wl_cache):
    K_wl = np.load(wl_cache)
else:
    K_wl = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if all_graphs[i] and all_graphs[j]:
                k = wl_kernel(all_graphs[i], all_graphs[j], n_iter=OPTIMAL['wl_iter'])
                K_wl[i, j] = k; K_wl[j, i] = k
    K_wl = normalize_kernel(K_wl); np.save(wl_cache, K_wl)
print("  WL ready")

# SP
sp_cache = os.path.join(CACHE_DIR, f"K_sp_{n_mol}.npy")
if os.path.exists(sp_cache):
    K_sp = np.load(sp_cache)
else:
    K_sp = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if all_graphs[i] and all_graphs[j]:
                k = shortest_path_kernel(all_graphs[i], all_graphs[j])
                K_sp[i, j] = k; K_sp[j, i] = k
    K_sp = normalize_kernel(K_sp); np.save(sp_cache, K_sp)
print("  SP ready")

# ECFP
ecfp_cache = os.path.join(CACHE_DIR, f"K_ecfp_{n_mol}.npy")
if os.path.exists(ecfp_cache):
    K_ecfp = np.load(ecfp_cache)
else:
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), ECFP_RADIUS, ECFP_NBITS) if Chem.MolFromSmiles(s) else None for s in molecule_smiles]
    K_ecfp = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if fps[i] and fps[j]:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                K_ecfp[i, j] = sim; K_ecfp[j, i] = sim
    K_ecfp = normalize_kernel(K_ecfp); np.save(ecfp_cache, K_ecfp)
print("  ECFP ready")

# Desc
desc_cache = os.path.join(CACHE_DIR, f"K_desc_{n_mol}.npy")
if os.path.exists(desc_cache):
    K_desc = np.load(desc_cache)
else:
    desc_list = []
    for smi in molecule_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc_list.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                              Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                              Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol)])
        else:
            desc_list.append([np.nan]*7)
    desc_array = np.nan_to_num(np.array(desc_list), nan=0.0)
    scaler_desc = StandardScaler()
    desc_scaled = scaler_desc.fit_transform(desc_array)
    K_desc = rbf_kernel(desc_scaled, gamma=1.0/desc_scaled.shape[1])
    K_desc = normalize_kernel(K_desc); np.save(desc_cache, K_desc)
print("  Desc ready")

# ============================================================
# 4. 组合核 & 亲和力矩阵
# ============================================================
print("\n准备数据和训练...")
K_topo = OPTIMAL['wl_sp_ratio'] * K_wl + (1 - OPTIMAL['wl_sp_ratio']) * K_sp
K_topo = normalize_kernel(K_topo)

K_comb = (OPTIMAL['lambda_topo'] * K_topo +
          OPTIMAL['lambda_ecfp'] * K_ecfp +
          OPTIMAL['lambda_desc'] * K_desc)
K_comb = normalize_kernel(K_comb)

K_train = K_comb[np.ix_(train_idx, train_idx)]
K_test = K_comb[np.ix_(test_idx, train_idx)]

# 亲和力
Y_full = df_aff.values
for j in range(Y_full.shape[1]):
    col_mean = np.nanmean(Y_full[:, j])
    Y_full[np.isnan(Y_full[:, j]), j] = col_mean

Y_train = Y_full[train_idx, :]
Y_test = Y_full[test_idx, :]

# 标准化（基于训练集）
Y_train_scaled = np.zeros_like(Y_train)
Y_test_scaled = np.zeros_like(Y_test)
scalers = []
for j in range(n_proteins):
    scaler = StandardScaler()
    Y_train_scaled[:, j] = scaler.fit_transform(Y_train[:, j].reshape(-1, 1)).flatten()
    Y_test_scaled[:, j] = scaler.transform(Y_test[:, j].reshape(-1, 1)).flatten()
    scalers.append(scaler)

K_protein = np.eye(n_proteins)
alpha = OPTIMAL['alpha']

# ============================================================
# 5. 训练
# ============================================================
U_l, S_l, _ = np.linalg.svd(K_train, full_matrices=False)
U_p, S_p, _ = np.linalg.svd(K_protein, full_matrices=False)

S_l = S_l.reshape(-1, 1)
S_p = S_p.reshape(1, -1)
lambda_mat = (S_l + alpha) * (S_p + alpha)

Y_hat = U_l.T @ Y_train_scaled @ U_p
coeff_hat = Y_hat / lambda_mat
C = U_l @ coeff_hat @ U_p.T
print(f"系数矩阵 C 形状: {C.shape}")

# ============================================================
# 6. 预测训练集和测试集
# ============================================================
def predict_all(K_cross, C, K_protein, scalers):
    Y_pred_scaled = K_cross @ C @ K_protein
    Y_pred = np.zeros_like(Y_pred_scaled)
    for j in range(n_proteins):
        Y_pred[:, j] = scalers[j].inverse_transform(Y_pred_scaled[:, j].reshape(-1, 1)).flatten()
    return Y_pred

# 训练集预测（K_cross 即 K_train 自身）
Y_train_pred = predict_all(K_train, C, K_protein, scalers)
Y_test_pred = predict_all(K_test, C, K_protein, scalers)

# ============================================================
# 7. 计算性能指标
# ============================================================
r2_train = r2_score(Y_train.flatten(), Y_train_pred.flatten())
rmse_train = np.sqrt(mean_squared_error(Y_train.flatten(), Y_train_pred.flatten()))
r2_test = r2_score(Y_test.flatten(), Y_test_pred.flatten())
rmse_test = np.sqrt(mean_squared_error(Y_test.flatten(), Y_test_pred.flatten()))

print(f"\n训练集 R² = {r2_train:.4f}, RMSE = {rmse_train:.4f}")
print(f"测试集 R² = {r2_test:.4f}, RMSE = {rmse_test:.4f}")

# 保存性能指标
metrics = {
    'train_R2': r2_train, 'train_RMSE': rmse_train,
    'test_R2': r2_test, 'test_RMSE': rmse_test,
    'train_size': len(train_idx), 'test_size': len(test_idx),
    'hyperparams': OPTIMAL
}
pd.Series(metrics).to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"))

# ============================================================
# 8. 绘图（图5E-H）
# ============================================================
print("\n生成图5E-H...")

# 展平数据用于绘图
y_train_true_flat = Y_train.flatten()
y_train_pred_flat = Y_train_pred.flatten()
y_test_true_flat = Y_test.flatten()
y_test_pred_flat = Y_test_pred.flatten()

res_train = y_train_true_flat - y_train_pred_flat
res_test = y_test_true_flat - y_test_pred_flat

# 保存原始数据供论文备用
np.savez(os.path.join(OUTPUT_DIR, "plot_data_fig5E_H.npz"),
         y_train_true=y_train_true_flat, y_train_pred=y_train_pred_flat,
         y_test_true=y_test_true_flat, y_test_pred=y_test_pred_flat,
         res_train=res_train, res_test=res_test)

# --- 图5E: 散点图（预测 vs 真实） ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_train_true_flat, y_train_pred_flat, s=5, alpha=0.5, label='Training', color='blue')
ax.scatter(y_test_true_flat, y_test_pred_flat, s=5, alpha=0.8, label='Test', color='orange')

# 参考线
min_val = min(y_train_true_flat.min(), y_test_true_flat.min())
max_val = max(y_train_true_flat.max(), y_test_true_flat.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='y=x')

# 测试集拟合线（可选）
z = np.polyfit(y_test_true_flat, y_test_pred_flat, 1)
p = np.poly1d(z)
ax.plot([min_val, max_val], p([min_val, max_val]), 'r-', linewidth=1.5, label=f'Fit (Test)')

ax.set_xlabel('True Affinity (kcal/mol)')
ax.set_ylabel('Predicted Affinity (kcal/mol)')
ax.legend(loc='upper left')
ax.set_title(f'Test R² = {r2_test:.4f}, RMSE = {rmse_test:.4f}')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "Fig5E_scatter.png"), dpi=300)
plt.savefig(os.path.join(FIGURE_DIR, "Fig5E_scatter.pdf"))
plt.close()
print("  Fig5E 保存完成")

# --- 图5F: 残差 vs 预测值 ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_train_pred_flat, res_train, s=5, alpha=0.5, label='Training', color='blue')
ax.scatter(y_test_pred_flat, res_test, s=5, alpha=0.8, label='Test', color='orange')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Predicted Affinity (kcal/mol)')
ax.set_ylabel('Residual (kcal/mol)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "Fig5F_residuals_vs_pred.png"), dpi=300)
plt.savefig(os.path.join(FIGURE_DIR, "Fig5F_residuals_vs_pred.pdf"))
plt.close()
print("  Fig5F 保存完成")

# --- 图5G: 残差直方图 ---
fig, ax = plt.subplots(figsize=(6, 5))
all_res = np.concatenate([res_train, res_test])
n, bins, patches = ax.hist(all_res, bins=50, density=True, alpha=0.7, color='gray', edgecolor='black')

# 拟合正态分布
mu, std = stats.norm.fit(all_res)
x = np.linspace(all_res.min(), all_res.max(), 100)
pdf = stats.norm.pdf(x, mu, std)
ax.plot(x, pdf, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.3f}, σ={std:.3f})')

ax.set_xlabel('Residual (kcal/mol)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "Fig5G_residual_hist.png"), dpi=300)
plt.savefig(os.path.join(FIGURE_DIR, "Fig5G_residual_hist.pdf"))
plt.close()
print("  Fig5G 保存完成")

# --- 图5H: Q-Q 图 ---
fig, ax = plt.subplots(figsize=(6, 5))
stats.probplot(all_res, dist="norm", plot=ax)
ax.get_lines()[0].set_marker('o')
ax.get_lines()[0].set_markersize(4)
ax.get_lines()[0].set_alpha(0.6)
ax.get_lines()[1].set_color('red')
ax.get_lines()[1].set_linewidth(2)
ax.set_title('')
ax.set_ylabel('Sample Quantiles')
ax.set_xlabel('Theoretical Quantiles')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "Fig5H_QQ_plot.png"), dpi=300)
plt.savefig(os.path.join(FIGURE_DIR, "Fig5H_QQ_plot.pdf"))
plt.close()
print("  Fig5H 保存完成")

# ============================================================
# 9. 保存模型
# ============================================================
model_data = {
    'C': C,
    'scalers': scalers,
    'train_idx': train_idx,
    'test_idx': test_idx,
    'hyperparams': OPTIMAL
}
with open(os.path.join(OUTPUT_DIR, 'kronrls_model.pkl'), 'wb') as f:
    pickle.dump(model_data, f)

print("\n" + "="*70)
print("训练完成！")
print(f"测试集 R² = {r2_test:.4f}, RMSE = {rmse_test:.4f}")
print(f"图表保存在: {FIGURE_DIR}")
print("="*70)
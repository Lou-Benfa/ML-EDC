#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
拓扑核（WL+SP）调优脚本
使用 7:1.5:1.5 划分，在验证集上选择最优参数
搜索空间：WL_iter=[1,2,3], WL/SP_ratio=[0.3,0.5,0.7], alpha=[0.1,0.5,1.0,2.0]
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
import networkx as nx
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
OUTPUT_DIR = r"E:\EDC\topo_tuning_7_1.5_1.5"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = r"E:\EDC\cache_topo_tuning"
os.makedirs(CACHE_DIR, exist_ok=True)

AFFINITY_FILE = r"E:\EDC\affinity.xlsx"
SMILES_FILE = r"E:\EDC\EDCs_SMILES.xlsx"

# 搜索空间
WL_ITERS = [1, 2, 3]
WL_SP_RATIOS = [0.3, 0.5, 0.7]          # WL 权重，SP = 1 - 此值
ALPHA_VALS = [0.1, 0.5, 1.0, 2.0]      # 正则化参数

# 数据划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

print("="*70)
print("拓扑核 (WL+SP) 调优 (7:1.5:1.5 划分)")
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
    """归一化核矩阵"""
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
# 2. 加载数据并划分（7:1.5:1.5）
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
# 3. 计算全数据集拓扑核（WL 和 SP）
# ============================================================
print("\n计算全数据集拓扑核...")

# 3a. WL 核（缓存不同迭代次数）
K_wl_cache = {}
for wl_iter in WL_ITERS:
    cache_file = os.path.join(CACHE_DIR, f"K_wl_full_iter{wl_iter}_{n_mol}.npy")
    if os.path.exists(cache_file):
        K_wl = np.load(cache_file)
        print(f"  WL iter={wl_iter} loaded from cache")
    else:
        print(f"  Computing WL iter={wl_iter}...")
        start = time.time()
        K_wl = np.zeros((n_mol, n_mol))
        for i in range(n_mol):
            for j in range(i, n_mol):
                if all_graphs[i] and all_graphs[j]:
                    k = wl_kernel(all_graphs[i], all_graphs[j], n_iter=wl_iter)
                    K_wl[i, j] = k
                    K_wl[j, i] = k
        K_wl = normalize_kernel(K_wl)
        np.save(cache_file, K_wl)
        print(f"  WL iter={wl_iter} computed in {time.time()-start:.1f}s")
    K_wl_cache[wl_iter] = K_wl

# 3b. SP 核
sp_cache_file = os.path.join(CACHE_DIR, f"K_sp_full_{n_mol}.npy")
if os.path.exists(sp_cache_file):
    K_sp_full = np.load(sp_cache_file)
    print("  SP loaded from cache")
else:
    print("  Computing SP...")
    start = time.time()
    K_sp_full = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if all_graphs[i] and all_graphs[j]:
                k = shortest_path_kernel(all_graphs[i], all_graphs[j])
                K_sp_full[i, j] = k
                K_sp_full[j, i] = k
    K_sp_full = normalize_kernel(K_sp_full)
    np.save(sp_cache_file, K_sp_full)
    print(f"  SP computed in {time.time()-start:.1f}s")

# ============================================================
# 4. 构建亲和力矩阵
# ============================================================
print("\n构建亲和力矩阵...")
Y_full = df_aff.values
for j in range(Y_full.shape[1]):
    col_mean = np.nanmean(Y_full[:, j])
    for i in range(Y_full.shape[0]):
        if np.isnan(Y_full[i, j]):
            Y_full[i, j] = col_mean

Y_train = Y_full[train_idx, :]
Y_val = Y_full[val_idx, :]
Y_test = Y_full[test_idx, :]

# 基于训练集标准化
Y_train_scaled = np.zeros_like(Y_train)
Y_val_scaled = np.zeros_like(Y_val)
Y_test_scaled = np.zeros_like(Y_test)
scalers = []
for j in range(n_proteins):
    scaler = StandardScaler()
    Y_train_scaled[:, j] = scaler.fit_transform(Y_train[:, j].reshape(-1, 1)).flatten()
    Y_val_scaled[:, j] = scaler.transform(Y_val[:, j].reshape(-1, 1)).flatten()
    Y_test_scaled[:, j] = scaler.transform(Y_test[:, j].reshape(-1, 1)).flatten()
    scalers.append(scaler)

K_protein = np.eye(n_proteins)

# ============================================================
# 5. 网格搜索（使用验证集选择参数）
# ============================================================
print("\n开始网格搜索（验证集评估）...")

# 预计算各 WL 迭代的拓扑核组合
# 注意：每次组合需要重新组合 WL 和 SP，再归一化，再切片
results = []
best_val_r2 = -np.inf
best_params = None
total = len(WL_ITERS) * len(WL_SP_RATIOS) * len(ALPHA_VALS)
count = 0

for wl_iter in WL_ITERS:
    K_wl = K_wl_cache[wl_iter]
    for wl_ratio in WL_SP_RATIOS:
        sp_ratio = 1.0 - wl_ratio
        
        # 组合全核，再归一化
        K_topo_full = wl_ratio * K_wl + sp_ratio * K_sp_full
        K_topo_full = normalize_kernel(K_topo_full)
        
        # 切片
        K_train = K_topo_full[np.ix_(train_idx, train_idx)]
        K_val_cross = K_topo_full[np.ix_(val_idx, train_idx)]
        
        for alpha in ALPHA_VALS:
            count += 1
            print(f"\n组合 {count}/{total}: WL_iter={wl_iter}, WL_ratio={wl_ratio:.1f}, alpha={alpha:.1f}")
            
            # 标准 KronRLS：直接对 K_train 和 K_protein 做 SVD
            U_l, S_l, _ = np.linalg.svd(K_train, full_matrices=False)
            U_p, S_p, _ = np.linalg.svd(K_protein, full_matrices=False)
            S_l = S_l.reshape(-1, 1)
            S_p = S_p.reshape(1, -1)
            lambda_mat = (S_l + alpha) * (S_p + alpha)
            
            Y_hat = U_l.T @ Y_train_scaled @ U_p
            coeff_hat = Y_hat / lambda_mat
            coeff = U_l @ coeff_hat @ U_p.T
            
            # 预测验证集（标准 KronRLS：K_cross @ C @ K_protein）
            Y_val_pred_scaled = K_val_cross @ coeff @ K_protein
            Y_val_pred = np.zeros_like(Y_val_pred_scaled)
            for j in range(n_proteins):
                Y_val_pred[:, j] = scalers[j].inverse_transform(
                    Y_val_pred_scaled[:, j].reshape(-1, 1)
                ).flatten()
            
            # 评估
            r2_val = r2_score(Y_val.flatten(), Y_val_pred.flatten())
            rmse_val = np.sqrt(mean_squared_error(Y_val.flatten(), Y_val_pred.flatten()))
            print(f"  验证集 R² = {r2_val:.4f}, RMSE = {rmse_val:.4f}")
            
            results.append({
                'wl_iter': wl_iter,
                'wl_ratio': wl_ratio,
                'alpha': alpha,
                'val_R2': r2_val,
                'val_RMSE': rmse_val
            })
            
            if r2_val > best_val_r2:
                best_val_r2 = r2_val
                best_params = {'wl_iter': wl_iter, 'wl_ratio': wl_ratio, 'alpha': alpha}

# ============================================================
# 6. 输出结果
# ============================================================
print("\n" + "="*70)
print("调优结果")
print("="*70)
print(f"最佳参数: WL_iter={best_params['wl_iter']}, WL_ratio={best_params['wl_ratio']:.1f}, α={best_params['alpha']:.1f}")
print(f"最佳验证集 R²: {best_val_r2:.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "topo_tuning_results.csv"), index=False)

np.save(os.path.join(OUTPUT_DIR, "best_topo_params.npy"), best_params)

print(f"\n结果保存至: {OUTPUT_DIR}")
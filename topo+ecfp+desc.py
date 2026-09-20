#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三元核权重调优脚本（7:1.5:1.5 划分）
固定拓扑内部参数（来自第一步调优），搜索三元权重组合
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
import networkx as nx
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
OUTPUT_DIR = r"E:\EDC\ternary_tuning_7_1.5_1.5"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = r"E:\EDC\cache_ternary_tuning"
os.makedirs(CACHE_DIR, exist_ok=True)

AFFINITY_FILE = r"E:\EDC\affinity.xlsx"
SMILES_FILE = r"E:\EDC\EDCs_SMILES.xlsx"

# 固定的拓扑最优参数（来自第一步调优）
FIXED_WL_ITER = 2
FIXED_WL_SP_RATIO = 0.3
FIXED_ALPHA = 0.1

# ECFP 参数
ECFP_RADIUS = 2
ECFP_NBITS = 2048

# 三元权重搜索（步长 0.1，总和 = 1.0）
WEIGHT_STEP = 0.1
TERNARY_COMBINATIONS = []
for lt in np.arange(0, 1.01, WEIGHT_STEP):
    for le in np.arange(0, 1.01 - lt, WEIGHT_STEP):
        ld = 1.0 - lt - le
        if ld >= 0:
            TERNARY_COMBINATIONS.append((round(lt, 2), round(le, 2), round(ld, 2)))

# 数据划分（与第一步一致，使用相同随机种子）
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

print("="*70)
print("三元核权重调优 (7:1.5:1.5 划分)")
print(f"固定: WL_iter={FIXED_WL_ITER}, WL/SP_ratio={FIXED_WL_SP_RATIO}, α={FIXED_ALPHA}")
print(f"搜索三元组合数: {len(TERNARY_COMBINATIONS)}")
print("="*70)

# ============================================================
# 1. 图核函数（同前）
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
# 2. 加载数据并划分（与第一步一致）
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

# 7:1.5:1.5 划分（与第一步使用相同随机种子）
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
# 3. 计算全数据集所有核（WL, SP, ECFP, 描述符）
# ============================================================
print("\n计算全数据集核矩阵...")

# 3a. WL 核（只用最优迭代次数）
wl_cache_file = os.path.join(CACHE_DIR, f"K_wl_iter{FIXED_WL_ITER}_{n_mol}.npy")
if os.path.exists(wl_cache_file):
    K_wl = np.load(wl_cache_file)
    print("  WL loaded from cache")
else:
    print("  Computing WL...")
    start = time.time()
    K_wl = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if all_graphs[i] and all_graphs[j]:
                k = wl_kernel(all_graphs[i], all_graphs[j], n_iter=FIXED_WL_ITER)
                K_wl[i, j] = k
                K_wl[j, i] = k
    K_wl = normalize_kernel(K_wl)
    np.save(wl_cache_file, K_wl)
    print(f"  WL computed in {time.time()-start:.1f}s")

# 3b. SP 核
sp_cache_file = os.path.join(CACHE_DIR, f"K_sp_{n_mol}.npy")
if os.path.exists(sp_cache_file):
    K_sp = np.load(sp_cache_file)
    print("  SP loaded from cache")
else:
    print("  Computing SP...")
    start = time.time()
    K_sp = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if all_graphs[i] and all_graphs[j]:
                k = shortest_path_kernel(all_graphs[i], all_graphs[j])
                K_sp[i, j] = k
                K_sp[j, i] = k
    K_sp = normalize_kernel(K_sp)
    np.save(sp_cache_file, K_sp)
    print(f"  SP computed in {time.time()-start:.1f}s")

# 3c. ECFP 核
ecfp_cache_file = os.path.join(CACHE_DIR, f"K_ecfp_{n_mol}.npy")
if os.path.exists(ecfp_cache_file):
    K_ecfp = np.load(ecfp_cache_file)
    print("  ECFP loaded from cache")
else:
    print("  Computing ECFP...")
    fps = []
    for smi in molecule_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=ECFP_NBITS))
        else:
            fps.append(None)
    K_ecfp = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if fps[i] is not None and fps[j] is not None:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                K_ecfp[i, j] = sim
                K_ecfp[j, i] = sim
    K_ecfp = normalize_kernel(K_ecfp)
    np.save(ecfp_cache_file, K_ecfp)
    print("  ECFP computed")

# 3d. 描述符核（RBF）
desc_cache_file = os.path.join(CACHE_DIR, f"K_desc_{n_mol}.npy")
if os.path.exists(desc_cache_file):
    K_desc = np.load(desc_cache_file)
    print("  Desc loaded from cache")
else:
    print("  Computing Desc...")
    desc_list = []
    for smi in molecule_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol)]
            desc_list.append(desc)
        else:
            desc_list.append([np.nan]*7)
    desc_array = np.nan_to_num(np.array(desc_list), nan=0.0)
    scaler_desc = StandardScaler()
    desc_scaled = scaler_desc.fit_transform(desc_array)
    gamma = 1.0 / desc_scaled.shape[1]
    K_desc = rbf_kernel(desc_scaled, gamma=gamma)
    K_desc = normalize_kernel(K_desc)
    np.save(desc_cache_file, K_desc)
    print("  Desc computed")

# ============================================================
# 4. 构建拓扑组合核（固定参数）
# ============================================================
print("\n构建拓扑核（固定参数）...")
K_topo = FIXED_WL_SP_RATIO * K_wl + (1 - FIXED_WL_SP_RATIO) * K_sp
K_topo = normalize_kernel(K_topo)

# ============================================================
# 5. 构建亲和力矩阵
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
alpha = FIXED_ALPHA

# ============================================================
# 6. 三元权重网格搜索（使用验证集）
# ============================================================
print(f"\n开始搜索 {len(TERNARY_COMBINATIONS)} 个三元组合...")
results = []
best_val_r2 = -np.inf
best_weights = None

# 预切片核矩阵（固定）
K_train_topo = K_topo[np.ix_(train_idx, train_idx)]
K_val_topo = K_topo[np.ix_(val_idx, train_idx)]
K_train_ecfp = K_ecfp[np.ix_(train_idx, train_idx)]
K_val_ecfp = K_ecfp[np.ix_(val_idx, train_idx)]
K_train_desc = K_desc[np.ix_(train_idx, train_idx)]
K_val_desc = K_desc[np.ix_(val_idx, train_idx)]

for idx, (lt, le, ld) in enumerate(TERNARY_COMBINATIONS):
    if lt == 0 and le == 0 and ld == 0:
        continue
    
# 组合训练核
    K_train = lt * K_train_topo + le * K_train_ecfp + ld * K_train_desc
    K_train = normalize_kernel(K_train)
    
    # 组合验证交叉核
    K_val_cross = lt * K_val_topo + le * K_val_ecfp + ld * K_val_desc
    
    # 标准 KronRLS：直接对 K_train 和 K_protein 做 SVD
    U_l, S_l, _ = np.linalg.svd(K_train, full_matrices=False)
    U_p, S_p, _ = np.linalg.svd(K_protein, full_matrices=False)
    S_l = S_l.reshape(-1, 1)
    S_p = S_p.reshape(1, -1)
    lambda_mat = (S_l + alpha) * (S_p + alpha)
    
    Y_hat = U_l.T @ Y_train_scaled @ U_p
    coeff_hat = Y_hat / lambda_mat
    coeff = U_l @ coeff_hat @ U_p.T
    
    # 预测验证集
    Y_val_pred_scaled = K_val_cross @ coeff @ K_protein
    Y_val_pred = np.zeros_like(Y_val_pred_scaled)
    for j in range(n_proteins):
        Y_val_pred[:, j] = scalers[j].inverse_transform(
            Y_val_pred_scaled[:, j].reshape(-1, 1)
        ).flatten()
    
    r2_val = r2_score(Y_val.flatten(), Y_val_pred.flatten())
    rmse_val = np.sqrt(mean_squared_error(Y_val.flatten(), Y_val_pred.flatten()))
    
    results.append({
        'lambda_topo': lt,
        'lambda_ecfp': le,
        'lambda_desc': ld,
        'val_R2': r2_val,
        'val_RMSE': rmse_val
    })
    
    if (idx + 1) % 20 == 0:
        print(f"  组合 {idx+1}/{len(TERNARY_COMBINATIONS)}: Topo={lt:.1f}, ECFP={le:.1f}, Desc={ld:.1f} -> R²={r2_val:.4f}")
    
    if r2_val > best_val_r2:
        best_val_r2 = r2_val
        best_weights = {'lambda_topo': lt, 'lambda_ecfp': le, 'lambda_desc': ld}

# ============================================================
# 7. 输出结果
# ============================================================
print(f"\n最佳三元权重: Topo={best_weights['lambda_topo']:.1f}, ECFP={best_weights['lambda_ecfp']:.1f}, Desc={best_weights['lambda_desc']:.1f}")
print(f"最佳验证集 R²: {best_val_r2:.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "ternary_weight_results.csv"), index=False)
np.save(os.path.join(OUTPUT_DIR, "best_ternary_weights.npy"), best_weights)

print(f"\nTop 10 权重组合:")
top10 = df_results.sort_values('val_R2', ascending=False).head(10)
print(top10[['lambda_topo', 'lambda_ecfp', 'lambda_desc', 'val_R2']].to_string(index=False))

print(f"\n结果保存至: {OUTPUT_DIR}")
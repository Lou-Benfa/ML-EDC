#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KronRLS 模型评估（分子拆分 + LOPO）
修复 LOPO 蛋白核矩阵
"""

import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from scipy.stats import shapiro, skew, kurtosis, probplot
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import networkx as nx
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
OUTPUT_DIR = r"E:\EDC\evaluation_results0908"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AFFINITY_FILE = r"E:\EDC\affinity.xlsx"
SMILES_FILE = r"E:\EDC\EDCs_SMILES.xlsx"
POCKET_FILE = r"E:\EDC\pocket_residues_from_docking\pocket_features_summary.csv"

# 最优参数
FIXED_WL_ITER = 2
FIXED_WL_SP_RATIO = 0.3
FIXED_ALPHA = 0.1

# 加载最优三元权重
TERNARY_WEIGHTS_FILE = r"E:\EDC\ternary_tuning_7_1.5_1.5\best_ternary_weights.npy"
if os.path.exists(TERNARY_WEIGHTS_FILE):
    weights = np.load(TERNARY_WEIGHTS_FILE, allow_pickle=True).item()
    LAMBDA_TOPO = weights['lambda_topo']
    LAMBDA_ECFP = weights['lambda_ecfp']
    LAMBDA_DESC = weights['lambda_desc']
else:
    LAMBDA_TOPO, LAMBDA_ECFP, LAMBDA_DESC = 0.5, 0.1, 0.4

ECFP_RADIUS = 2
ECFP_NBITS = 2048
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("="*70)
print("KronRLS 模型评估 (分子拆分 + LOPO)")
print(f"三元权重: Topo={LAMBDA_TOPO}, ECFP={LAMBDA_ECFP}, Desc={LAMBDA_DESC}")
print(f"WL迭代={FIXED_WL_ITER}, WL/SP比例={FIXED_WL_SP_RATIO}, α={FIXED_ALPHA}")
print("="*70)

# ============================================================
# 新增：辅助函数（用于计算详细统计指标）
# ============================================================
def calculate_metrics(y_true, y_pred, label=""):
    """
    计算回归指标并返回QQ图数据。
    返回字典：metrics, scatter_df, qq_df
    """
    # 去除NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    
    if n == 0:
        # 空数据返回空结果
        metrics = {
            'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan,
            'Skewness': np.nan, 'Kurtosis': np.nan,
            'Shapiro_W': np.nan, 'Shapiro_p': np.nan,
            'Sample_Size': 0
        }
        scatter_df = pd.DataFrame({'True': [], 'Predicted': [], 'Residual': []})
        qq_df = pd.DataFrame({'Theoretical_Quantiles': [], 'Sample_Quantiles': []})
        return metrics, scatter_df, qq_df
    
    # 基本指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    residuals = y_true - y_pred
    
    # 偏度与峰度
    sk = skew(residuals)
    ku = kurtosis(residuals, fisher=True)  # Fisher定义（正态峰度为0）
    
    # Shapiro-Wilk检验（仅当样本量 3≤n≤5000 时计算）
    if 3 <= n <= 5000:
        shapiro_stat, shapiro_p = shapiro(residuals)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    # QQ图数据（理论分位数 vs 样本分位数）
    if n > 0:
        # probplot返回 (理论分位数, 排序后的样本值)
        osm, osr = probplot(residuals, dist="norm", fit=False)[:2]
    else:
        osm, osr = np.array([]), np.array([])
    
    # 散点图数据
    scatter_df = pd.DataFrame({
        'True': y_true,
        'Predicted': y_pred,
        'Residual': residuals
    })
    
    # QQ图数据
    qq_df = pd.DataFrame({
        'Theoretical_Quantiles': osm,
        'Sample_Quantiles': osr
    })
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Skewness': sk,
        'Kurtosis': ku,
        'Shapiro_W': shapiro_stat,
        'Shapiro_p': shapiro_p,
        'Sample_Size': n
    }
    return metrics, scatter_df, qq_df

# ============================================================
# 1. 图核函数（与训练一致）
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

def combined_topo_kernel(G1, G2, wl_w=0.5, sp_w=0.5, n_iter=2):
    return wl_w * wl_kernel(G1, G2, n_iter) + sp_w * shortest_path_kernel(G1, G2)

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

# 口袋图构建（与 optimized_lopo.py 一致）
def build_pocket_graph(prot, df_pocket):
    if prot not in df_pocket.index:
        G = nx.Graph()
        for i in range(10):
            G.add_node(i, label='H')
        for i in range(9):
            G.add_edge(i, i+1)
        return G
    total = int(df_pocket.loc[prot, 'total'])
    if total <= 0:
        total = 10
    hydrophobic = int(df_pocket.loc[prot, 'hydrophobic_ratio'] * total)
    charged = int(df_pocket.loc[prot, 'charged_ratio'] * total)
    polar = int(df_pocket.loc[prot, 'polar_ratio'] * total)
    residues = ['H'] * hydrophobic + ['C'] * charged + ['P'] * polar
    remaining = total - len(residues)
    if remaining > 0:
        residues += ['O'] * remaining
    G = nx.Graph()
    for i, res in enumerate(residues):
        G.add_node(i, label=res)
    for i in range(len(residues) - 1):
        G.add_edge(i, i+1)
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
print(f"总分子数: {n_mol}, 蛋白数: {n_proteins}")

# ============================================================
# 3. 计算或加载全数据集核矩阵（分子核 + 蛋白核）
# ============================================================
CACHE_DIR = r"E:\EDC\cache_full_ternary"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_or_compute_kernel(name, compute_func):
    cache_path = os.path.join(CACHE_DIR, f"{name}_{n_mol if 'K_' in name else 'protein'}.npy")
    if os.path.exists(cache_path):
        print(f"  加载缓存 {name}...")
        return np.load(cache_path)
    else:
        print(f"  计算 {name}...")
        start = time.time()
        K = compute_func()
        np.save(cache_path, K)
        print(f"  缓存 {name} (耗时 {time.time()-start:.1f}s)")
        return K

# 分子核（WL, SP, ECFP, DESC）
def compute_wl():
    K = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if all_graphs[i] and all_graphs[j]:
                k = wl_kernel(all_graphs[i], all_graphs[j], n_iter=FIXED_WL_ITER)
                K[i, j] = k
                K[j, i] = k
    return normalize_kernel(K)

K_wl = load_or_compute_kernel("K_wl", compute_wl)

def compute_sp():
    K = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if all_graphs[i] and all_graphs[j]:
                k = shortest_path_kernel(all_graphs[i], all_graphs[j])
                K[i, j] = k
                K[j, i] = k
    return normalize_kernel(K)

K_sp = load_or_compute_kernel("K_sp", compute_sp)

def compute_ecfp():
    fps = []
    for smi in molecule_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=ECFP_NBITS))
        else:
            fps.append(None)
    K = np.zeros((n_mol, n_mol))
    for i in range(n_mol):
        for j in range(i, n_mol):
            if fps[i] and fps[j]:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                K[i, j] = sim
                K[j, i] = sim
    return normalize_kernel(K)

K_ecfp = load_or_compute_kernel("K_ecfp", compute_ecfp)

def compute_desc():
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
    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(desc_array)
    gamma = 1.0 / desc_scaled.shape[1]
    K = rbf_kernel(desc_scaled, gamma=gamma)
    return normalize_kernel(K)

K_desc = load_or_compute_kernel("K_desc", compute_desc)

# 组合拓扑
K_topo = FIXED_WL_SP_RATIO * K_wl + (1 - FIXED_WL_SP_RATIO) * K_sp
K_topo = normalize_kernel(K_topo)

# 融合三元核
K_ligand = LAMBDA_TOPO * K_topo + LAMBDA_ECFP * K_ecfp + LAMBDA_DESC * K_desc
K_ligand = normalize_kernel(K_ligand)
print("\n分子核融合完成")

# ---- 新增：计算蛋白核矩阵（基于口袋图） ----
def compute_protein_kernel():
    df_pocket = pd.read_excel(POCKET_FILE) if POCKET_FILE.endswith('.xlsx') else pd.read_csv(POCKET_FILE, index_col=0)
    # 确保索引为蛋白名
    if 'Unnamed: 0' in df_pocket.columns:
        df_pocket.set_index('Unnamed: 0', inplace=True)
    # 构建口袋图
    pocket_graphs = {prot: build_pocket_graph(prot, df_pocket) for prot in protein_names}
    K_prot = np.zeros((n_proteins, n_proteins))
    for i, pi in enumerate(protein_names):
        for j, pj in enumerate(protein_names):
            K_prot[i, j] = combined_topo_kernel(pocket_graphs[pi], pocket_graphs[pj], wl_w=FIXED_WL_SP_RATIO, sp_w=1-FIXED_WL_SP_RATIO, n_iter=FIXED_WL_ITER)
    return normalize_kernel(K_prot)

K_protein = load_or_compute_kernel("K_protein", compute_protein_kernel)
print("蛋白核矩阵加载完成")

# ============================================================
# 4. 构建亲和力矩阵并标准化
# ============================================================
print("\n构建亲和力矩阵...")
Y_full = df_aff.values
for j in range(Y_full.shape[1]):
    col_mean = np.nanmean(Y_full[:, j])
    for i in range(Y_full.shape[0]):
        if np.isnan(Y_full[i, j]):
            Y_full[i, j] = col_mean

# 全局标准化（每个蛋白独立）
scaler_per_protein = []
Y_scaled = np.zeros_like(Y_full)
for j in range(n_proteins):
    scaler = StandardScaler()
    Y_scaled[:, j] = scaler.fit_transform(Y_full[:, j].reshape(-1, 1)).flatten()
    scaler_per_protein.append(scaler)

# 保存标准化器
with open(os.path.join(OUTPUT_DIR, "scalers.pkl"), 'wb') as f:
    pickle.dump(scaler_per_protein, f)

# ============================================================
# 7. LOPO 评估
# ============================================================
print("\n" + "="*70)
print("LOPO 评估 (留一蛋白)")
print("="*70)

# 提前计算分子核的SVD以优化效率
print("预计算分子核SVD...")
U_l, S_l, _ = np.linalg.svd(K_ligand, full_matrices=False)
S_l = S_l.reshape(-1, 1)  # 列向量
print("分子核SVD完成")

lopo_detailed_stats = []  # 用于汇总所有折叠的详细统计信息
lopo_results = []  # 保留简单的R²和RMSE汇总（兼容旧格式）

for test_prot_idx in range(n_proteins):
    test_prot = protein_names[test_prot_idx]
    train_prot_idx = [i for i in range(n_proteins) if i != test_prot_idx]

    # 蛋白核子集（真实核）
    K_protein_train = K_protein[np.ix_(train_prot_idx, train_prot_idx)]
    K_protein_test = K_protein[test_prot_idx:test_prot_idx+1, train_prot_idx]  # 1 x (n_proteins-1)

    # 训练亲和力（所有分子，训练蛋白列）
    Y_train_prot_scaled = Y_scaled[:, train_prot_idx]  # (n_mol, n_train_proteins)

    # 对 K_protein_train 做 SVD
    U_p, S_p, _ = np.linalg.svd(K_protein_train, full_matrices=False)
    S_p = S_p.reshape(1, -1)  # 行向量
    lambda_mat = (S_l + FIXED_ALPHA) * (S_p + FIXED_ALPHA)

    # 解系数
    Y_hat = U_l.T @ Y_train_prot_scaled @ U_p
    coeff_hat = Y_hat / lambda_mat
    coeff = U_l @ coeff_hat @ U_p.T

    # ===== 测试集预测与评估 =====
    Y_pred_scaled = K_ligand @ coeff @ K_protein_test.T   # (n_mol, 1)
    Y_pred = scaler_per_protein[test_prot_idx].inverse_transform(Y_pred_scaled).flatten()
    Y_true = Y_full[:, test_prot_idx]

    # 计算测试集详细指标
    test_metrics, test_scatter, test_qq = calculate_metrics(Y_true, Y_pred, label=f"Test_{test_prot}")

    # 保存测试集数据
    test_scatter.to_csv(os.path.join(OUTPUT_DIR, f"lopo_{test_prot}_test_scatter.csv"), index=False)
    test_qq.to_csv(os.path.join(OUTPUT_DIR, f"lopo_{test_prot}_test_qq.csv"), index=False)
    
    # 保留原有的简单预测文件（兼容性）
    df_test_prot = pd.DataFrame({
        'CAS': molecule_cas,
        'True_Affinity': Y_true,
        'Pred_Affinity': Y_pred
    })
    df_test_prot.to_csv(os.path.join(OUTPUT_DIR, f"lopo_{test_prot}_predictions.csv"), index=False)

    # ===== 训练集预测与评估 =====
    Y_pred_train_scaled = K_ligand @ coeff @ K_protein_train.T
    Y_pred_train = np.zeros_like(Y_pred_train_scaled)
    for idx_prot, prot_idx in enumerate(train_prot_idx):
        Y_pred_train[:, idx_prot] = scaler_per_protein[prot_idx].inverse_transform(
            Y_pred_train_scaled[:, idx_prot].reshape(-1, 1)
        ).flatten()
    Y_true_train = Y_full[:, train_prot_idx]

    # 展平训练集（所有分子 × 所有训练蛋白）
    y_true_train_flat = Y_true_train.flatten()
    y_pred_train_flat = Y_pred_train.flatten()

    # 计算训练集详细指标
    train_metrics, train_scatter, train_qq = calculate_metrics(y_true_train_flat, y_pred_train_flat, label=f"Train_{test_prot}")

    # 保存训练集数据
    train_scatter.to_csv(os.path.join(OUTPUT_DIR, f"lopo_{test_prot}_train_scatter.csv"), index=False)
    train_qq.to_csv(os.path.join(OUTPUT_DIR, f"lopo_{test_prot}_train_qq.csv"), index=False)
    
    # 保留原有的训练预测文件（兼容性）
    df_train_prot = pd.DataFrame({
        'CAS': np.repeat(molecule_cas, len(train_prot_idx)),
        'Protein': np.repeat([protein_names[i] for i in train_prot_idx], n_mol),
        'True_Affinity': Y_true_train.flatten(),
        'Pred_Affinity': Y_pred_train.flatten()
    })
    df_train_prot.to_csv(os.path.join(OUTPUT_DIR, f"lopo_{test_prot}_train_predictions.csv"), index=False)

    # ===== 汇总统计 =====
    # 添加测试集和训练集的统计到详细汇总表
    for prefix, m_dict in [("Test", test_metrics), ("Train", train_metrics)]:
        row = {'Protein': test_prot, 'Set': prefix}
        row.update(m_dict)
        lopo_detailed_stats.append(row)
    
    # 简单汇总（兼容旧格式）
    lopo_results.append({
        'Protein': test_prot,
        'Test_R2': test_metrics['R2'],
        'Test_RMSE': test_metrics['RMSE']
    })
    
    print(f"  LOPO {test_prot}: Test R²={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.4f}, Train R²={train_metrics['R2']:.4f}")

# ===== 保存汇总结果 =====
# 详细统计汇总表
df_detailed_stats = pd.DataFrame(lopo_detailed_stats)
df_detailed_stats.to_csv(os.path.join(OUTPUT_DIR, "lopo_detailed_stats.csv"), index=False)
print(f"\n详细统计汇总保存至: {OUTPUT_DIR}/lopo_detailed_stats.csv")

# 简单汇总表（兼容旧格式）
df_lopo_summary = pd.DataFrame(lopo_results)
df_lopo_summary.to_csv(os.path.join(OUTPUT_DIR, "lopo_summary.csv"), index=False)
print(f"LOPO简单汇总保存至: {OUTPUT_DIR}/lopo_summary.csv")

# ============================================================
# 8. 保存参数
# ============================================================
params = {
    'wl_iter': FIXED_WL_ITER,
    'wl_sp_ratio': FIXED_WL_SP_RATIO,
    'alpha': FIXED_ALPHA,
    'lambda_topo': LAMBDA_TOPO,
    'lambda_ecfp': LAMBDA_ECFP,
    'lambda_desc': LAMBDA_DESC
}
np.save(os.path.join(OUTPUT_DIR, "evaluation_params.npy"), params)
print(f"\n评估参数保存至: {OUTPUT_DIR}/evaluation_params.npy")

print("\n" + "="*70)
print("✅ 评估完成!")
print(f"结果保存在: {OUTPUT_DIR}")
print("="*70)
"""
量表属性图构建模块
==================
方式C：以互信息（Mutual Information）作为边权
  - 节点：136个量表属性
  - 边  ：MI(xi, xj) 超过阈值的属性对
  - 边权：归一化的互信息值

流程：
  1. 统计每对属性的联合选项分布 P(xi, xj)
  2. 计算互信息矩阵 MI ∈ R^{n×n}
  3. 百分位阈值过滤，得到稀疏邻接矩阵 A
  4. 归一化边权 W_ij = MI_ij / max(MI)
  5. 构建 PyG Data 对象（edge_index + edge_weight）
  6. 节点初始特征由 ScaleTokenizer 提供
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 复用上一步的工具函数
# ──────────────────────────────────────────────

def load_scale(path: str):
    df = pd.read_excel(path)
    id_col    = df.columns[0]
    attr_cols = df.columns[1:].tolist()
    option_counts = []
    for col in attr_cols:
        max_val = int(df[col].max())
        option_counts.append(max_val + 1 if df[col].min() == 0 else max_val)
    return df, id_col, attr_cols, option_counts


def preprocess(df: pd.DataFrame, attr_cols: list) -> torch.LongTensor:
    data = df[attr_cols].values.copy()
    for i, col in enumerate(attr_cols):
        if data[:, i].min() == 1:
            data[:, i] -= 1
    return torch.tensor(data, dtype=torch.long)


class ScaleTokenizer(nn.Module):
    def __init__(self, attr_cols, option_counts, d_model=64, prior_reliability=None):
        super().__init__()
        self.n_attrs = len(attr_cols)
        self.attr_emb = nn.Embedding(self.n_attrs, d_model)
        self.option_embs = nn.ModuleList([
            nn.Embedding(k, d_model) for k in option_counts
        ])
        prior = torch.ones(self.n_attrs) if prior_reliability is None \
                else torch.tensor(prior_reliability, dtype=torch.float32)
        self.register_buffer('prior', prior.unsqueeze(1))
        self.register_buffer('attr_idx', torch.arange(self.n_attrs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        attr_part = self.attr_emb(self.attr_idx).unsqueeze(0).expand(B, -1, -1)
        val_parts = [self.option_embs[i](x[:, i]) for i in range(self.n_attrs)]
        val_part  = torch.stack(val_parts, dim=1)
        return (attr_part + val_part) * self.prior


# ──────────────────────────────────────────────
# 核心：互信息矩阵计算
# ──────────────────────────────────────────────

def compute_mi_matrix(X_np: np.ndarray, attr_cols: list) -> np.ndarray:
    """
    计算所有属性对之间的互信息。

    参数
    ----
    X_np     : shape [N, n_attrs]，0-indexed 整数数组
    attr_cols: 属性列名列表

    返回
    ----
    MI : shape [n_attrs, n_attrs]，对称矩阵，对角线为0
    """
    n = len(attr_cols)
    MI = np.zeros((n, n), dtype=np.float32)

    print(f'计算互信息矩阵（{n}×{n} = {n*n} 对）...')
    total = n * (n - 1) // 2
    done  = 0

    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_info_score(X_np[:, i], X_np[:, j])
            MI[i, j] = mi
            MI[j, i] = mi
            done += 1
            if done % 1000 == 0:
                print(f'  {done}/{total} ({100*done/total:.1f}%)')

    print(f'  完成 {total} 对计算')
    return MI


# ──────────────────────────────────────────────
# 图构建
# ──────────────────────────────────────────────

def build_graph(
    MI: np.ndarray,
    percentile: float = 25.0,
) -> tuple:
    """
    根据互信息矩阵构建稀疏图。

    参数
    ----
    MI         : 互信息矩阵 [n, n]
    percentile : 阈值百分位（参考MIRNet的Q_alpha=25）

    返回
    ----
    edge_index  : torch.LongTensor [2, num_edges]
    edge_weight : torch.FloatTensor [num_edges]，归一化到[0,1]
    adj_matrix  : np.ndarray [n, n]，稀疏邻接矩阵（0/1）
    """
    n = MI.shape[0]

    # 取上三角非零值计算阈值
    upper = MI[np.triu_indices(n, k=1)]
    nonzero = upper[upper > 0]
    threshold = np.percentile(nonzero, percentile)

    print(f'\n互信息统计：')
    print(f'  非零MI对数    : {len(nonzero)}')
    print(f'  MI 均值       : {nonzero.mean():.4f}')
    print(f'  MI 最大值     : {nonzero.max():.4f}')
    print(f'  阈值 Q{percentile:.0f}      : {threshold:.4f}')

    # 构建邻接矩阵
    adj = (MI >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)

    # 归一化边权：W_ij = MI_ij / max(MI)
    mi_max = MI.max()
    W = MI / mi_max  # 归一化到 [0, 1]

    # 提取边（只保留 adj=1 的位置）
    rows, cols = np.where(adj > 0)
    edge_index  = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(W[rows, cols], dtype=torch.float32)

    num_edges = edge_index.size(1)
    density   = num_edges / (n * (n - 1))

    print(f'  边数量（有向）: {num_edges}')
    print(f'  图密度        : {density:.3f}（1.0为全连接）')
    print(f'  平均每节点度  : {num_edges / n:.1f}')

    return edge_index, edge_weight, adj


# ──────────────────────────────────────────────
# 组装 PyG Data 对象
# ──────────────────────────────────────────────

def build_pyg_data(
    X: torch.LongTensor,
    tokenizer: ScaleTokenizer,
    edge_index: torch.LongTensor,
    edge_weight: torch.FloatTensor,
    batch_size: int = 32,
) -> list:
    """
    为每个 batch 生成 PyG Data 对象列表。
    每个 Data 包含：
      x           : 节点特征 [n_attrs, d_model]（来自Tokenizer，取batch均值作为图级初始化）
      edge_index  : 边索引   [2, num_edges]（所有患者共享同一图结构）
      edge_weight : 边权重   [num_edges]
      patient_tokens: 每个患者的token [B, n_attrs, d_model]（用于后续per-patient推理）
    """
    N = X.size(0)
    batches = []

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        x_b   = X[start:end]                          # [B, n_attrs]
        with torch.no_grad():
            tokens = tokenizer(x_b)                   # [B, n_attrs, d_model]

        # 图节点特征：batch内所有患者的token均值，作为图级表示
        # shape: [n_attrs, d_model]
        node_feat = tokens.mean(dim=0)

        data = Data(
            x            = node_feat,                 # [n_attrs, d_model]
            edge_index   = edge_index,                # [2, num_edges]
            edge_attr    = edge_weight.unsqueeze(1),  # [num_edges, 1]
            patient_tokens = tokens,                  # [B, n_attrs, d_model]
            num_nodes    = node_feat.size(0),
        )
        batches.append(data)

    return batches


def rank_mi_pairs(
    MI: np.ndarray,
    attr_cols: list,
    topk: int = 50,
    save_path: str = None,
):
    """
    对所有属性对按互信息从高到低排序。

    参数
    ----
    MI        : [n, n] 互信息矩阵
    attr_cols : 属性名
    topk      : 输出前k个（None表示全部）
    save_path : 是否保存为csv
    """
    n = MI.shape[0]
    pairs = []

    # 只取上三角，避免重复
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, MI[i, j]))

    # 排序（降序）
    pairs.sort(key=lambda x: x[2], reverse=True)

    # 截断
    if topk is not None:
        pairs = pairs[:topk]

    # 打印
    print('\n互信息排序（Top {}）：'.format(topk if topk else 'All'))
    print(f'  {"idx_i":^6} {"idx_j":^6} {"MI":^10}  属性对')
    print(f'  {"-"*6} {"-"*6} {"-"*10}  {"-"*30}')

    for i, j, mi in pairs:
        name_i = attr_cols[i][:20]
        name_j = attr_cols[j][:20]
        print(f'  [{i:3d}] [{j:3d}] {mi:10.4f}  {name_i}  ↔  {name_j}')

    # 可选保存
    if save_path is not None:
        df = pd.DataFrame([
            {
                "i": i,
                "j": j,
                "attr_i": attr_cols[i],
                "attr_j": attr_cols[j],
                "MI": mi
            }
            for i, j, mi in pairs
        ])
        df.to_csv(save_path, index=False)
        print(f'\n已保存到: {save_path}')

    return pairs


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

if __name__ == '__main__':

    DATA_PATH  = './scale.xlsx'
    D_MODEL    = 64
    PERCENTILE = 25.0
    BATCH_SIZE = 32

    # 1. 加载数据
    print('=' * 55)
    print('步骤1：加载数据')
    print('=' * 55)
    df, id_col, attr_cols, option_counts = load_scale(DATA_PATH)
    X = preprocess(df, attr_cols)
    X_np = X.numpy()
    print(f'数据 shape: {X.shape}')

    # 2. 计算互信息矩阵
    print()
    print('=' * 55)
    print('步骤2：计算互信息矩阵')
    print('=' * 55)
    MI = compute_mi_matrix(X_np, attr_cols)
    print(f'MI 矩阵 shape: {MI.shape}')

    rank_mi_pairs(MI, attr_cols, topk=50, save_path='./mi_rank.csv')

    # 3. 构建图
    print()
    print('=' * 55)
    print('步骤3：构建图结构')
    print('=' * 55)
    edge_index, edge_weight, adj = build_graph(MI, percentile=PERCENTILE)

    # 4. 初始化 Tokenizer
    print()
    print('=' * 55)
    print('步骤4：初始化 Tokenizer 并生成节点特征')
    print('=' * 55)
    tokenizer = ScaleTokenizer(attr_cols, option_counts, d_model=D_MODEL)

    # 5. 构建 PyG Data 列表
    batches = build_pyg_data(X, tokenizer, edge_index, edge_weight, BATCH_SIZE)
    print(f'Batch 数量: {len(batches)}')

    # 6. 验证第一个 batch
    print()
    print('=' * 55)
    print('步骤5：验证第一个 Batch 的 Data 对象')
    print('=' * 55)
    d0 = batches[0]
    print(f'节点特征 x          : {d0.x.shape}      → [n_attrs={len(attr_cols)}, d_model={D_MODEL}]')
    print(f'边索引 edge_index   : {d0.edge_index.shape}  → [2, num_edges]')
    print(f'边权重 edge_attr    : {d0.edge_attr.shape} → [num_edges, 1]')
    print(f'患者tokens          : {d0.patient_tokens.shape} → [B, n_attrs, d_model]')
    print(f'节点数              : {d0.num_nodes}')
    print()

    # 7. 输出高权重边示例（互信息最高的10条边）
    print('互信息最高的10条属性关联边：')
    topk = edge_weight.topk(10)
    print(f'  {"属性i":^6}  {"属性j":^6}  {"MI权重":^8}')
    print(f'  {"-"*6}  {"-"*6}  {"-"*8}')
    seen = set()
    count = 0
    for val, idx in zip(topk.values, topk.indices):
        i = edge_index[0, idx].item()
        j = edge_index[1, idx].item()
        pair = (min(i,j), max(i,j))
        if pair not in seen:
            seen.add(pair)
            name_i = attr_cols[i][:18]
            name_j = attr_cols[j][:18]
            print(f'  [{i:3d}]{name_i:18s}  ↔  [{j:3d}]{name_j:18s}  {val.item():.4f}')
            count += 1
            if count >= 10:
                break

    # 8. 保存互信息矩阵和图结构供后续使用
    torch.save({
        'MI'          : torch.tensor(MI),
        'edge_index'  : edge_index,
        'edge_weight' : edge_weight,
        'adj'         : torch.tensor(adj),
        'attr_cols'   : attr_cols,
        'option_counts': option_counts,
    }, './new_graph_data.pt')
    print()
    print('图数据已保存至 new_graph_data.pt')
    print()
    print('=' * 55)
    print('图初始化完成 ✓')
    print('=' * 55)
"""
GAT 模块 + 可信度估计头
========================
输入：患者量表原始数据  X  [B, n_attrs]
输出：
  reliability  [B, n_attrs]     每个属性的可信度权重 r_i ∈ (0,1)
  f_scale      [B, d_model]     可信度加权聚合后的量表表示（供后续融合使用）
  hat_T        [B, n_attrs, 2d] 残差拼接后的节点特征（可选，供调试）

架构流程：
  X
  ↓  ScaleTokenizer
  T  [B, n, d]          节点初始特征 v_i^(0)
  ↓  GATv2Conv × L层
  T' [B, n, d]          消息传递后的节点特征 v_i^(L)
  ↓  残差拼接
  T_hat [B, n, 2d]      [v_i^(0) || v_i^(L)]
  ↓  ReliabilityHead (MLP)
  r  [B, n]             每属性可信度
  ↓  加权平均
  f_scale [B, d]        量表整体表示

参考 MIRNet：
  - GATv2Conv（动态注意力）
  - 共现置信度加权：α̃_ij = α_ij × W_ij（MI归一化边权）
  - 残差拼接保留初始视觉特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GATv2Conv


# ══════════════════════════════════════════════
# 模块一：ScaleTokenizer（与之前一致）
# ══════════════════════════════════════════════

class ScaleTokenizer(nn.Module):
    """
    将量表属性索引转换为 token 向量序列。
    token_i = p_i * (E_attr[i] + E_val[i](x_i))
    """

    def __init__(self, attr_cols, option_counts, d_model=64,
                 prior_reliability=None):
        super().__init__()
        self.n_attrs = len(attr_cols)
        self.d_model = d_model

        # 属性身份嵌入
        self.attr_emb = nn.Embedding(self.n_attrs, d_model)

        # 每个属性独立的选项嵌入表
        self.option_embs = nn.ModuleList([
            nn.Embedding(k, d_model) for k in option_counts
        ])

        # 先验可信度（固定，不参与梯度）
        if prior_reliability is None:
            prior = torch.ones(self.n_attrs)
        else:
            prior = torch.tensor(prior_reliability, dtype=torch.float32)
        self.register_buffer('prior', prior.unsqueeze(1))      # [n, 1]
        self.register_buffer('attr_idx', torch.arange(self.n_attrs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : [B, n_attrs]  0-indexed 选项索引
        return : [B, n_attrs, d_model]
        """
        B = x.size(0)
        attr_part = self.attr_emb(self.attr_idx)               # [n, d]
        attr_part = attr_part.unsqueeze(0).expand(B, -1, -1)   # [B, n, d]

        val_parts = [self.option_embs[i](x[:, i])
                     for i in range(self.n_attrs)]
        val_part  = torch.stack(val_parts, dim=1)              # [B, n, d]

        tokens = (attr_part + val_part) * self.prior           # [B, n, d]
        return tokens


# ══════════════════════════════════════════════
# 模块二：Domain-Aware GAT
# ══════════════════════════════════════════════

class DomainAwareGAT(nn.Module):
    """
    基于 GATv2Conv 的属性关系图注意力网络。

    核心改进（参考 MIRNet）：
      1. edge_attr = MI归一化边权，调制注意力强度
         α̃_ij = α_ij × W_ij
      2. 先验重要性 p_i 在 Tokenizer 阶段已注入节点特征
      3. 残差拼接：hat_t_i = [v_i^(0) || v_i^(L)]

    参数
    ----
    d_model   : 节点特征维度（= Tokenizer 的 d_model）
    n_heads   : 注意力头数
    n_layers  : GAT 层数
    dropout   : 注意力 dropout
    """

    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.d_model  = d_model

        # 每层 GATv2Conv
        # 输入 d_model，输出 d_model（多头 concat 后再投影回 d_model）
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels  = d_model,
                    out_channels = d_model // n_heads,
                    heads        = n_heads,
                    edge_dim     = 1,        # MI边权维度
                    concat       = True,     # concat多头 → d_model
                    dropout      = dropout,
                    add_self_loops = False,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x          : [B×n, d_model]  展平后的节点特征
        edge_index : [2, E]
        edge_attr  : [E, 1]          MI归一化边权

        return     : [B×n, d_model]  更新后的节点特征
        """
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr=edge_attr)  # [B×n, d_model]
            h = norm(h)
            h = F.elu(h)
            h = self.dropout(h)
            x = x + h   # 层间残差
        return x


# ══════════════════════════════════════════════
# 模块三：可信度估计头
# ══════════════════════════════════════════════

class ReliabilityHead(nn.Module):
    """
    将残差拼接后的节点特征映射为可信度标量。

    输入：hat_t_i = [v_i^(0) || v_i^(L)]  维度 2*d_model
    输出：r_i ∈ (0, 1)

    用小型 MLP 实现，sigmoid 输出。
    """

    def __init__(self, d_model=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, hat_T: torch.Tensor) -> torch.Tensor:
        """
        hat_T  : [B, n_attrs, 2*d_model]
        return : [B, n_attrs]
        """
        return self.mlp(hat_T).squeeze(-1)   # [B, n]


# ══════════════════════════════════════════════
# 整体：ScaleGATEncoder
# ══════════════════════════════════════════════

class ScaleGATEncoder(nn.Module):
    """
    量表侧完整编码器：
      Tokenizer → GAT → 残差拼接 → 可信度头 → 加权聚合

    输入：
      x          : [B, n_attrs]   量表原始选项索引（0-indexed）
      edge_index : [2, E]         图边（训练前离线构建，共享）
      edge_attr  : [E, 1]         MI归一化边权

    输出：
      f_scale    : [B, d_model]   可信度加权量表表示（送入融合层）
      reliability: [B, n_attrs]   每属性可信度权重
      hat_T      : [B, n_attrs, 2d] 残差特征（可选，供调试/EM使用）
    """

    def __init__(self, attr_cols, option_counts,
                 d_model=64, n_heads=4, n_layers=2,
                 dropout=0.1, prior_reliability=None):
        super().__init__()
        self.n_attrs = len(attr_cols)
        self.d_model = d_model

        self.tokenizer  = ScaleTokenizer(
            attr_cols, option_counts, d_model, prior_reliability)
        self.gat        = DomainAwareGAT(
            d_model, n_heads, n_layers, dropout)
        self.rel_head   = ReliabilityHead(d_model)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        B, n = x.size(0), self.n_attrs

        # ── Step 1：Tokenizer → 初始节点特征 ──────
        T0 = self.tokenizer(x)                        # [B, n, d]

        # ── Step 2：展平为 PyG 所需的节点矩阵 ──────
        # PyG 的 GATv2Conv 期望 [num_nodes_total, d]
        # 这里把 B 个患者的图拼成一张大图（batch graph）
        # 每个患者的节点索引偏移 i*n
        T0_flat = T0.reshape(B * n, self.d_model)      # [B*n, d]

        # 构建 batch 级 edge_index（每个患者复制一份图，偏移节点索引）
        offsets    = torch.arange(B, device=x.device) * n  # [B]
        ei_batched = edge_index.unsqueeze(0) + \
                     offsets.view(B, 1, 1)              # [B, 2, E]
        ei_batched = ei_batched.permute(1, 0, 2).reshape(2, -1)  # [2, B*E]

        # 边属性直接重复 B 次
        ea_batched = edge_attr.repeat(B, 1)            # [B*E, 1]

        # ── Step 3：GAT 消息传递 ───────────────────
        TL_flat = self.gat(T0_flat, ei_batched, ea_batched)  # [B*n, d]
        TL      = TL_flat.reshape(B, n, self.d_model)        # [B, n, d]

        # ── Step 4：残差拼接（参考 MIRNet 预测头）──
        hat_T = torch.cat([T0, TL], dim=-1)            # [B, n, 2d]

        # ── Step 5：可信度估计 ─────────────────────
        reliability = self.rel_head(hat_T)             # [B, n]

        # ── Step 6：可信度加权聚合 → 量表表示 ──────
        # f_scale = Σ r_i * hat_t_i[:d] / Σ r_i
        # 注意：聚合时用 TL（GAT更新后）而非 hat_T
        r_w     = reliability.unsqueeze(-1)            # [B, n, 1]
        f_scale = (TL * r_w).sum(dim=1) / \
                  (r_w.sum(dim=1) + 1e-8)              # [B, d]

        return f_scale, reliability, hat_T


# ══════════════════════════════════════════════
# 数据预处理工具（与之前一致）
# ══════════════════════════════════════════════

def load_scale(path):
    df = pd.read_excel(path)
    id_col    = df.columns[0]
    attr_cols = df.columns[1:].tolist()
    option_counts = []
    for col in attr_cols:
        mx = int(df[col].max())
        option_counts.append(mx + 1 if df[col].min() == 0 else mx)
    return df, id_col, attr_cols, option_counts


def preprocess(df, attr_cols):
    data = df[attr_cols].values.copy()
    for i in range(len(attr_cols)):
        if data[:, i].min() == 1:
            data[:, i] -= 1
    return torch.tensor(data, dtype=torch.long)


# ══════════════════════════════════════════════
# 主流程：演示前向传播 + 可信度输出
# ══════════════════════════════════════════════

if __name__ == '__main__':

    DATA_PATH  = './scale.xlsx'
    GRAPH_PATH = './graph_data.pt'
    D_MODEL    = 64
    N_HEADS    = 4
    N_LAYERS   = 2
    DROPOUT    = 0.1
    BATCH_SIZE = 32

    # ── 加载数据 ──────────────────────────────
    print('=' * 55)
    print('Step 1: 加载数据')
    print('=' * 55)
    df, id_col, attr_cols, option_counts = load_scale(DATA_PATH)
    X = preprocess(df, attr_cols)
    print(f'患者数: {len(df)},  属性数: {len(attr_cols)}')

    # ── 加载图结构 ────────────────────────────
    print()
    print('=' * 55)
    print('Step 2: 加载图结构')
    print('=' * 55)
    g          = torch.load(GRAPH_PATH, weights_only=False)
    edge_index = g['edge_index']                    # [2, E]
    edge_attr  = g['edge_weight'].unsqueeze(1)      # [E, 1]
    print(f'节点数: {len(attr_cols)}')
    print(f'边数  : {edge_index.size(1)}')
    print(f'边权范围: [{edge_attr.min():.4f}, {edge_attr.max():.4f}]')

    # ── 初始化模型 ────────────────────────────
    print()
    print('=' * 55)
    print('Step 3: 初始化 ScaleGATEncoder')
    print('=' * 55)
    model = ScaleGATEncoder(
        attr_cols      = attr_cols,
        option_counts  = option_counts,
        d_model        = D_MODEL,
        n_heads        = N_HEADS,
        n_layers       = N_LAYERS,
        dropout        = DROPOUT,
        prior_reliability = None,   # 全1，可替换为专家赋值列表
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_params:,}')
    print(f'  Tokenizer : '
          f'{sum(p.numel() for p in model.tokenizer.parameters()):,}')
    print(f'  GAT       : '
          f'{sum(p.numel() for p in model.gat.parameters()):,}')
    print(f'  RelHead   : '
          f'{sum(p.numel() for p in model.rel_head.parameters()):,}')

    # ── 取一个 batch 做前向传播 ───────────────
    print()
    print('=' * 55)
    print('Step 4: 前向传播（一个 batch）')
    print('=' * 55)
    x_batch = X[:BATCH_SIZE]                        # [32, 136]

    model.eval()
    with torch.no_grad():
        f_scale, reliability, hat_T = model(
            x_batch, edge_index, edge_attr)

    print(f'f_scale     shape: {f_scale.shape}')
    print(f'  → 量表整体表示，后续送入 Cross-Attention 融合层')
    print(f'reliability shape: {reliability.shape}')
    print(f'  → 每个患者每个属性的可信度权重')
    print(f'hat_T       shape: {hat_T.shape}')
    print(f'  → 残差拼接节点特征，供 EM 迭代使用')

    # ── 可信度统计 ────────────────────────────
    print()
    print('=' * 55)
    print('Step 5: 可信度统计（随机初始化阶段，仅验证形状）')
    print('=' * 55)
    r_np = reliability.numpy()
    print(f'可信度均值: {r_np.mean():.4f}')
    print(f'可信度标准差: {r_np.std():.4f}')
    print(f'可信度范围: [{r_np.min():.4f}, {r_np.max():.4f}]')

    print()
    print('前5个患者、前8个属性的可信度：')
    header = '         ' + ''.join([f'attr{i:3d}' for i in range(8)])
    print(header)
    for i in range(5):
        row = f'patient{i:2d}:' + ''.join(
            [f'  {r_np[i, j]:.3f}' for j in range(8)])
        print(row)

    # ── 输出说明 ──────────────────────────────
    print()
    print('=' * 55)
    print('输出接口说明（供后续融合使用）')
    print('=' * 55)
    print(f'f_scale      [B, {D_MODEL}]'
          f'  → 送入 Cross-Attention 的量表 Query/Value')
    print(f'reliability  [B, {len(attr_cols)}]'
          f'  → 可解释性输出，标注可疑属性')
    print(f'hat_T        [B, {len(attr_cols)}, {2*D_MODEL}]'
          f'  → 供 EM 迭代更新可信度')
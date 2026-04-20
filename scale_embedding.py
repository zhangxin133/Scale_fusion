"""
量表属性 Tokenizer — Embedding 嵌入模块
========================================
数据结构：
  - 136个属性列（二值 0/1 或多值 1~k）
  - 每个属性有独立的选项嵌入表
  - 最终每个患者输出 shape: [n_attrs, d_model]

核心公式：
  token_i = p_i * (E_attr[i] + E_val[i](x_i))
  其中 p_i 为先验可信度（默认全1，可由专家赋值）
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# 1. 读取数据，分析每个属性的选项数
# ──────────────────────────────────────────────

def load_scale(path: str):
    df = pd.read_excel(path)
    id_col   = df.columns[0]
    attr_cols = df.columns[1:].tolist()

    # 统计每个属性的选项数（最大值即选项数，二值属性为2）
    option_counts = []
    for col in attr_cols:
        max_val = int(df[col].max())
        # 二值属性值域为{0,1}，选项数记为2
        # 多值属性值域为{1,...,k}，选项数记为k
        option_counts.append(max_val + 1 if df[col].min() == 0 else max_val)

    return df, id_col, attr_cols, option_counts


# ──────────────────────────────────────────────
# 2. Tokenizer 模块定义
# ──────────────────────────────────────────────

class ScaleTokenizer(nn.Module):
    """
    将量表的 n 个属性转换为 token 序列。

    参数
    ----
    attr_cols    : 属性列名列表，长度 n
    option_counts: 每个属性的选项数列表，长度 n
    d_model      : embedding 维度
    prior_reliability : 先验可信度列表，长度 n，值域 [0,1]
                        None 表示全部设为 1.0
    """

    def __init__(
        self,
        attr_cols: list,
        option_counts: list,
        d_model: int = 64,
        prior_reliability: list = None,
    ):
        super().__init__()

        self.attr_cols    = attr_cols
        self.option_counts = option_counts
        self.d_model      = d_model
        self.n_attrs      = len(attr_cols)

        # 属性身份嵌入：每个属性一个可学习向量
        self.attr_emb = nn.Embedding(self.n_attrs, d_model)

        # 每个属性独立的选项嵌入表
        self.option_embs = nn.ModuleList([
            nn.Embedding(k, d_model)
            for k in option_counts
        ])

        # 先验可信度：固定缩放因子，不参与梯度更新
        if prior_reliability is None:
            prior = torch.ones(self.n_attrs)
        else:
            prior = torch.tensor(prior_reliability, dtype=torch.float32)
        # shape: [n_attrs, 1]，广播用
        self.register_buffer('prior', prior.unsqueeze(1))

        # 属性索引（固定，供查表用）
        attr_idx = torch.arange(self.n_attrs)
        self.register_buffer('attr_idx', attr_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.LongTensor, shape [B, n_attrs]
            每个患者每个属性选择的选项索引（从0开始）

        返回
        ----
        tokens : torch.FloatTensor, shape [B, n_attrs, d_model]
        """
        B = x.size(0)

        # 属性身份嵌入：[n_attrs, d_model] → [B, n_attrs, d_model]
        attr_part = self.attr_emb(self.attr_idx)           # [n, d]
        attr_part = attr_part.unsqueeze(0).expand(B, -1, -1)  # [B, n, d]

        # 选项值嵌入：逐属性查各自的嵌入表
        val_parts = []
        for i, emb in enumerate(self.option_embs):
            # x[:, i]: shape [B]，查第 i 个属性的选项嵌入
            val_parts.append(emb(x[:, i]))                 # [B, d]
        val_part = torch.stack(val_parts, dim=1)           # [B, n, d]

        # 相加得到 token，再乘先验可信度
        tokens = (attr_part + val_part) * self.prior       # [B, n, d]

        return tokens


# ──────────────────────────────────────────────
# 3. 数据预处理：原始选项值 → 0-indexed 索引
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame, attr_cols: list) -> torch.LongTensor:
    """
    将 DataFrame 中的属性值转换为 0-indexed LongTensor。

    二值属性（{0,1}）：直接使用
    多值属性（{1,...,k}）：减1变为 {0,...,k-1}
    """
    data = df[attr_cols].values.copy()

    for i, col in enumerate(attr_cols):
        col_min = data[:, i].min()
        if col_min == 1:
            data[:, i] -= 1   # {1,...,k} → {0,...,k-1}
        # 二值 {0,1} 不变

    return torch.tensor(data, dtype=torch.long)


# ──────────────────────────────────────────────
# 4. 完整演示
# ──────────────────────────────────────────────

if __name__ == '__main__':

    DATA_PATH = './scale.xlsx'
    D_MODEL   = 64
    BATCH_SIZE = 4

    # 加载数据
    df, id_col, attr_cols, option_counts = load_scale(DATA_PATH)
    print(f'患者数量    : {len(df)}')
    print(f'属性数量    : {len(attr_cols)}')
    print(f'选项数范围  : min={min(option_counts)}, max={max(option_counts)}')
    print(f'Embedding维度: {D_MODEL}')
    print()

    # 预处理：原始值 → 0-indexed 索引
    X = preprocess(df, attr_cols)          # [N, n_attrs]
    print(f'输入张量 shape : {X.shape}')
    print(f'输入张量示例（前2个患者，前5个属性）:')
    print(X[:2, :5])
    print()

    # 初始化 Tokenizer
    # 示例先验可信度：全部为1.0（无先验偏置）
    # 实际使用时可由专家赋值，如 [0.9, 0.6, 0.8, ...]
    tokenizer = ScaleTokenizer(
        attr_cols=attr_cols,
        option_counts=option_counts,
        d_model=D_MODEL,
        prior_reliability=None,
    )

    total_params = sum(p.numel() for p in tokenizer.parameters())
    print(f'Tokenizer 可学习参数量: {total_params:,}')
    print()

    # 取一个 batch 做前向传播
    x_batch = X[:BATCH_SIZE]               # [B, n_attrs]
    tokens  = tokenizer(x_batch)           # [B, n_attrs, d_model]

    print(f'输出 tokens shape : {tokens.shape}')
    print(f'  含义: [{BATCH_SIZE}个患者, {len(attr_cols)}个属性, {D_MODEL}维embedding]')
    print()

    # 查看第0个患者、第0个属性的token
    print(f'患者0 属性0 的 token（前8维）:')
    print(tokens[0, 0, :8].detach().numpy().round(4))

    # 验证：不同选项值应产生不同token
    # 人工构造两个患者，只有属性0不同
    x_test      = X[:2].clone()
    x_test[0, 0] = 0
    x_test[1, 0] = 1
    t_test = tokenizer(x_test)
    diff = (t_test[0, 0] - t_test[1, 0]).abs().mean().item()
    print()
    print(f'属性0选项0 vs 选项1 的token平均差异: {diff:.4f}（应>0，验证选项区分有效）')
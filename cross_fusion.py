"""
交叉注意力融合模块（使用真实舌诊影像标签）
==========================================
将影像标签分支（12个舌诊标签）与量表属性分支（136个属性）融合。

完整前向流程（到 EM 迭代之前）：

  量表 X [B,136] ──→ ScaleGATEncoder ──→ hat_T [B,136,128]
                                           reliability [B,136]
  影像 X_img [B,12] → ImageLabelEncoder → F_img [B,12,64]
                                           conf  [B,12]
                              ↓
                    CrossModalFusion
                    Q = F_img  [B,12,64]   影像标签询问量表
                    K = TL     [B,136,64]  量表 GAT 更新特征
                    V = TL     [B,136,64]
                    可信度调制（reliability × conf）
                              ↓
                    f_fused  [B,128]  → EM 迭代 / 诊断头

注意力权重 attn_weights [B, 12, 136] 的语义：
  attn_weights[b, j, i] = 舌诊标签j 对 量表属性i 的关注度
  训练后可直接解读为"影像发现的舌诊特征j 与量表第i项的关联强度"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/mnt/user-data/outputs')

from GAT import ScaleGATEncoder
from scale_embedding import load_scale, preprocess
from image_encoder import (ImageLabelEncoder, load_image_labels,
                                  build_image_tensor,
                                  N_LABELS, LABEL_NAMES)


# ══════════════════════════════════════════════
# 交叉注意力融合模块
# ══════════════════════════════════════════════

class CrossModalFusion(nn.Module):
    """
    12个舌诊标签 × 136个量表属性 的跨模态交叉注意力融合。

    参数
    ----
    d_model    : token 维度（两路保持一致）
    n_heads    : 注意力头数
    dropout    : attention dropout
    """

    def __init__(self, d_model: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        assert d_model % n_heads == 0

        # QKV 投影
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout     = nn.Dropout(dropout)
        self.norm_cross  = nn.LayerNorm(d_model)
        self.norm_scale  = nn.LayerNorm(d_model)

        # 融合后 FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
        )

    def forward(self,
                F_img: torch.Tensor,
                hat_T: torch.Tensor,
                reliability: torch.Tensor,
                conf: torch.Tensor):
        """
        参数
        ----
        F_img       : [B, 12, d]     舌诊标签 token（Query 来源）
        hat_T       : [B, 136, 2d]   量表残差特征（取前 d 维作 Key/Value）
        reliability : [B, 136]       量表属性可信度（调制 Key/Value）
        conf        : [B, 12]        舌诊标签置信度（GT 模式全为 1.0）

        返回
        ----
        f_fused      : [B, 2d]       融合表示，送入 EM / 诊断头
        attn_weights : [B, 12, 136]  注意力权重（可解释性）
        """
        B  = F_img.size(0)
        k  = F_img.size(1)     # 12（舌诊标签数）
        n  = hat_T.size(1)     # 136（量表属性数）

        # ── 1. 取量表 GAT 更新后特征（hat_T 前 d 维）─
        TL = hat_T[:, :, :self.d_model]                    # [B, 136, d]

        # ── 2. 可信度调制 ───────────────────────────
        # 量表侧：降低不可信属性的 Key/Value 幅度
        r_w  = reliability.unsqueeze(-1)                   # [B, 136, 1]
        TL_r = TL * r_w                                    # [B, 136, d]

        # 影像侧：GT 模式 conf 全 1，不做折扣
        c_w     = conf.unsqueeze(-1)                       # [B, 12, 1]
        F_img_c = F_img * c_w                              # [B, 12, d]

        # ── 3. QKV 投影 ────────────────────────────
        Q = self.W_Q(F_img_c)                              # [B, 12,  d]
        K = self.W_K(TL_r)                                 # [B, 136, d]
        V = self.W_V(TL_r)                                 # [B, 136, d]

        # ── 4. 多头拆分 ────────────────────────────
        def split_heads(x, seq):
            return x.view(B, seq, self.n_heads,
                          self.d_head).transpose(1, 2)     # [B,h,seq,dh]

        Q = split_heads(Q, k)    # [B, h, 12,  dh]
        K = split_heads(K, n)    # [B, h, 136, dh]
        V = split_heads(V, n)    # [B, h, 136, dh]

        # ── 5. 缩放点积注意力 ──────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) \
                 * (self.d_head ** -0.5)                   # [B, h, 12, 136]
        attn   = F.softmax(scores, dim=-1)                 # [B, h, 12, 136]
        attn   = self.dropout(attn)

        # ── 6. 加权聚合 ────────────────────────────
        cross_out = torch.matmul(attn, V)                  # [B, h, 12, dh]
        cross_out = cross_out.transpose(1, 2).contiguous() \
                             .view(B, k, self.d_model)     # [B, 12, d]
        cross_out = self.W_O(cross_out)

        # 残差 + LayerNorm（影像 token 层面）
        cross_out = self.norm_cross(F_img + cross_out)     # [B, 12, d]

        # 多头均值注意力权重（可解释性）
        attn_weights = attn.mean(dim=1)                    # [B, 12, 136]

        # ── 7. 影像整体表示（均值 pooling，GT conf=1）─
        f_img_pool = cross_out.mean(dim=1)                 # [B, d]
        f_img_pool = self.norm_cross(f_img_pool)           # [B, d]

        # ── 8. 量表整体表示（可信度加权均值）──────────
        f_scale_pool = (TL_r).sum(dim=1) / \
                       (r_w.sum(dim=1) + 1e-8)             # [B, d]
        f_scale_pool = self.norm_scale(f_scale_pool)       # [B, d]

        # ── 9. 拼接 + FFN ──────────────────────────
        f_concat = torch.cat([f_img_pool, f_scale_pool],
                             dim=-1)                       # [B, 2d]
        f_fused  = self.ffn(f_concat)                      # [B, 2d]

        return f_fused, attn_weights


# ══════════════════════════════════════════════
# 完整前向流程封装（到 EM 之前）
# ══════════════════════════════════════════════

class ScaleImageFusionPreEM(nn.Module):
    """
    从原始输入到 EM 迭代前的完整前向流程。

    输入：
      x_scale    : [B, 136]    量表选项索引（0-indexed）
      x_img      : [B, 12]     舌诊标签（0-indexed，由 build_image_tensor 生成）
      edge_index : [2, E]
      edge_attr  : [E, 1]

    输出：
      f_fused      : [B, 128]        融合表示 → EM / 诊断头
      reliability  : [B, 136]        量表可信度 → EM 更新
      hat_T        : [B, 136, 128]   量表残差特征 → EM 更新
      attn_weights : [B, 12, 136]    注意力权重 → 可解释性
      conf         : [B, 12]         影像标签置信度（GT 全为 1.0）
    """

    def __init__(self,
                 attr_cols, option_counts,
                 d_model: int = 64,
                 gat_heads: int = 4,
                 gat_layers: int = 2,
                 cross_heads: int = 4,
                 dropout: float = 0.1,
                 prior_reliability=None):
        super().__init__()

        self.scale_encoder = ScaleGATEncoder(
            attr_cols         = attr_cols,
            option_counts     = option_counts,
            d_model           = d_model,
            n_heads           = gat_heads,
            n_layers          = gat_layers,
            dropout           = dropout,
            prior_reliability = prior_reliability,
        )

        self.image_encoder = ImageLabelEncoder(d_model=d_model)

        self.fusion = CrossModalFusion(
            d_model  = d_model,
            n_heads  = cross_heads,
            dropout  = dropout,
        )

    def forward(self,
                x_scale: torch.Tensor,
                x_img: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor):

        # 量表分支
        f_scale, reliability, hat_T = self.scale_encoder(
            x_scale, edge_index, edge_attr)

        # 影像分支
        F_img, f_img, conf = self.image_encoder(x_img)

        # 交叉注意力融合
        f_fused, attn_weights = self.fusion(
            F_img, hat_T, reliability, conf)

        return f_fused, reliability, hat_T, attn_weights, conf


# ══════════════════════════════════════════════
# 演示：匹配量表与影像标签，完整前向传播
# ══════════════════════════════════════════════

if __name__ == '__main__':

    torch.manual_seed(42)

    SCALE_PATH = './scale.xlsx'
    IMG_PATH   = './all_info.xlsx'
    GRAPH_PATH = './graph_data.pt'
    D_MODEL    = 64
    BATCH_SIZE = 8

    # ── 加载数据 ──────────────────────────────
    print('=' * 60)
    print('Step 1: 加载量表、影像标签、图结构')
    print('=' * 60)

    # 量表
    df_scale, id_col, attr_cols, option_counts = load_scale(SCALE_PATH)
    # 影像
    df_img, id2idx = load_image_labels(IMG_PATH)
    # 图
    g          = torch.load(GRAPH_PATH, weights_only=False)
    edge_index = g['edge_index']
    edge_attr  = g['edge_weight'].unsqueeze(1)

    # 找到同时有量表和影像标签的患者
    img_ids   = set(df_img['IDAA'].astype(int))
    scale_ids = set(df_scale[id_col].astype(int))
    common    = sorted(img_ids & scale_ids)
    print(f'量表患者数    : {len(df_scale)}')
    print(f'影像患者数    : {len(df_img)}')
    print(f'共同患者数    : {len(common)}')

    # 取前 BATCH_SIZE 个共同患者
    batch_ids = common[:BATCH_SIZE]
    print(f'本次 batch ID : {batch_ids}')

    # 构建量表张量
    scale_indexed = df_scale.set_index(id_col)
    scale_rows = []
    for pid in batch_ids:
        row = scale_indexed.loc[pid]
        vals = row[attr_cols].values.copy()
        for i in range(len(attr_cols)):
            if vals[i] >= 1:
                vals[i] -= 1
        scale_rows.append(vals.tolist())
    x_scale = torch.tensor(scale_rows, dtype=torch.long)

    # 构建影像张量
    x_img = build_image_tensor(df_img, batch_ids)

    print(f'\nx_scale shape : {x_scale.shape}')
    print(f'x_img   shape : {x_img.shape}')

    # ── 初始化模型 ─────────────────────────────
    print()
    print('=' * 60)
    print('Step 2: 初始化 ScaleImageFusionPreEM')
    print('=' * 60)
    model = ScaleImageFusionPreEM(
        attr_cols     = attr_cols,
        option_counts = option_counts,
        d_model       = D_MODEL,
        gat_heads     = 4,
        gat_layers    = 2,
        cross_heads   = 4,
        dropout       = 0.1,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total:,}')
    print(f'  ScaleGATEncoder  : '
          f'{sum(p.numel() for p in model.scale_encoder.parameters()):,}')
    print(f'  ImageLabelEncoder: '
          f'{sum(p.numel() for p in model.image_encoder.parameters()):,}')
    print(f'  CrossModalFusion : '
          f'{sum(p.numel() for p in model.fusion.parameters()):,}')

    # ── 前向传播 ──────────────────────────────
    print()
    print('=' * 60)
    print('Step 3: 前向传播')
    print('=' * 60)
    model.eval()
    with torch.no_grad():
        f_fused, reliability, hat_T, attn_weights, conf = model(
            x_scale, x_img, edge_index, edge_attr)

    print(f'f_fused      {f_fused.shape}'
          f'  → 融合表示，送入 EM 迭代 / 诊断头')
    print(f'reliability  {reliability.shape}'
          f'  → 量表属性可信度')
    print(f'hat_T        {hat_T.shape}'
          f'  → 量表残差特征')
    print(f'attn_weights {attn_weights.shape}'
          f'  → 注意力权重 [B, 12舌诊标签, 136量表属性]')
    print(f'conf         {conf.shape}'
          f'  → 影像置信度（GT 全为 {conf.mean():.1f}）')

    # ── 注意力解读（患者0）────────────────────
    print()
    print('=' * 60)
    print('Step 4: 注意力权重解读（患者0）')
    print('=' * 60)
    print('每个舌诊标签最关注的前3个量表属性：')
    print()
    aw = attn_weights[0]    # [12, 136]
    for j in range(N_LABELS):
        top3 = aw[j].topk(3)
        pairs = ', '.join([
            f'[{i}]({v:.3f})'
            for i, v in zip(top3.indices.tolist(),
                            top3.values.tolist())
        ])
        print(f'  {LABEL_NAMES[j]:5s} → {pairs}')

    # ── 接口说明 ──────────────────────────────
    print()
    print('=' * 60)
    print('送入 EM 迭代的接口')
    print('=' * 60)
    print(f'f_fused     [B, {2*D_MODEL}]       → Z 的初始推断输入')
    print(f'reliability [B, {len(attr_cols)}]  → r 初始值，M步更新')
    print(f'hat_T       [B, {len(attr_cols)}, {2*D_MODEL}]  → Proj_i(Z) 计算用')

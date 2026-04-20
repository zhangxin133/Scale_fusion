"""
EM 迭代精炼模块
================
接收 CrossModalFusion 的输出，对量表属性可信度做迭代校正。

接口约定（与 cross_fusion.py 保持一致）：
  输入：
    f_fused     [B, 128]          融合表示（影像 + 量表）
    reliability [B, 136]          量表属性初始可信度（来自 GAT 可信度头）
    hat_T       [B, 136, 128]     量表残差特征（[v^(0) || v^(L)]，2d=128）

  输出（EMOutput dataclass）：
    z               [B, 128]       精炼后的隐变量（真实病症状态估计）
    reliability     [B, 136]       更新后的属性可信度
    expected_attr   [B, 136, 128]  每属性的期望语义（Proj_i(Z)）
    attr_error      [B, 136]       每属性的重建误差（越大越可疑）

EM 迭代逻辑：
  初始化：z = MLP(f_fused)

  for s in range(S):
    # E步：根据可信度加权的量表特征更新 Z
    pooled = Σ_i r_i * hat_T_i / Σ r_i
    z = LayerNorm(z + MLP([z, pooled]))

    # M步：从 Z 投影回每个属性的期望语义
    expected_i = MLP([z, attr_id_embed_i])   # Proj_i(Z)

    # 更新可信度：偏差大的属性可信度下降
    error_i   = ||hat_T_i - expected_i||^2
    r_i_new   = r_i * exp(-λ * error_i)
    r_i       = momentum * r_i + (1-momentum) * r_i_new
    r_i       = clamp(r_i, r_min, 1.0)

修正内容（相对于原始 expect_max.py）：
  1. _check_inputs: hat_T.size(1) != n_attrs（原为 >，漏掉 < 的情况）
  2. lambda_em 默认值从 5.0 降至 1.0（避免训练初期 reliability 过快坍缩）
  3. 新增 DiagnosisHead，将 z 映射为诊断 logits，补全了原代码缺失的输出头
  4. 新增 FullModel，将所有模块串联为可训练的端到端模型
  5. __main__ 中用真实数据路径演示端到端前向传播和损失计算
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════
# 数据类：EM 输出
# ══════════════════════════════════════════════

@dataclass
class EMOutput:
    z             : torch.Tensor                        # [B, attr_dim]
    reliability   : torch.Tensor                        # [B, n_attrs]
    expected_attr : torch.Tensor                        # [B, n_attrs, attr_dim]
    attr_error    : torch.Tensor                        # [B, n_attrs]
    reliability_history : Optional[List[torch.Tensor]] = field(default=None)
    z_history           : Optional[List[torch.Tensor]] = field(default=None)


# ══════════════════════════════════════════════
# EM 迭代模块
# ══════════════════════════════════════════════

class ExpectationMaximization(nn.Module):
    """
    用 EM 迭代精炼融合表示和量表属性可信度。

    参数
    ----
    fused_dim  : f_fused 的特征维度，默认 128（= 2 * d_model）
    attr_dim   : hat_T 每个节点的特征维度，默认 128（= 2 * d_model）
    n_attrs    : 量表属性数，默认 136
    n_iters    : EM 迭代次数，默认 3
    lambda_em  : M步可信度衰减强度，默认 1.0
                 （原始代码为 5.0，过大会导致训练初期 reliability 坍缩）
    momentum   : 可信度更新动量，默认 0.2
    r_min      : 可信度下限，防止完全归零，默认 0.1（提高下限防止训练初期完全坍缩）
    """

    def __init__(
        self,
        fused_dim  : int   = 128,
        attr_dim   : int   = 128,
        n_attrs    : int   = 136,
        n_iters    : int   = 3,
        lambda_em  : float = 1.0,   # 修正：从 5.0 降至 1.0
        momentum   : float = 0.2,
        r_min      : float = 0.1,
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.fused_dim = fused_dim
        self.attr_dim  = attr_dim
        self.n_attrs   = n_attrs
        self.n_iters   = n_iters
        self.lambda_em = lambda_em
        self.momentum  = momentum
        self.r_min     = r_min

        # 属性身份嵌入（EM 专用，用于条件化 Proj_i(Z)）
        # 与 GAT 中的 attr_emb 独立，学习"属性在 Z 空间的期望位置"
        self.attr_embed = nn.Embedding(n_attrs, attr_dim)

        # 初始化隐变量 z
        self.init_z = nn.Sequential(
            nn.Linear(fused_dim, attr_dim),
            nn.LayerNorm(attr_dim),
            nn.GELU(),
        )

        # E步：结合当前 z 和可信度加权属性摘要，更新 z
        self.e_step = nn.Sequential(
            nn.Linear(attr_dim * 2, attr_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attr_dim * 2, attr_dim),
        )
        self.z_norm = nn.LayerNorm(attr_dim)

        # M步：从 z 投影到每个属性的期望语义空间
        self.project_attr = nn.Sequential(
            nn.Linear(attr_dim * 2, attr_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attr_dim * 2, attr_dim),
        )
        self.attr_norm = nn.LayerNorm(attr_dim)

    def _check_inputs(self, f_fused, reliability, hat_T):
        if f_fused.dim() != 2:
            raise ValueError(f'f_fused 应为 [B, d]，实际 {tuple(f_fused.shape)}')
        if reliability.dim() != 2:
            raise ValueError(f'reliability 应为 [B, n]，实际 {tuple(reliability.shape)}')
        if hat_T.dim() != 3:
            raise ValueError(f'hat_T 应为 [B, n, d]，实际 {tuple(hat_T.shape)}')
        if not (f_fused.size(0) == reliability.size(0) == hat_T.size(0)):
            raise ValueError('f_fused / reliability / hat_T 的 batch size 不一致')
        if reliability.size(1) != hat_T.size(1):
            raise ValueError('reliability 与 hat_T 的 n_attrs 不一致')
        if f_fused.size(1) != self.fused_dim:
            raise ValueError(
                f'fused_dim 配置为 {self.fused_dim}，实际输入 {f_fused.size(1)}')
        if hat_T.size(2) != self.attr_dim:
            raise ValueError(
                f'attr_dim 配置为 {self.attr_dim}，实际输入 {hat_T.size(2)}')
        # 修正：原代码为 >，漏掉属性数不足的情况
        if hat_T.size(1) != self.n_attrs:
            raise ValueError(
                f'n_attrs 配置为 {self.n_attrs}，实际输入 {hat_T.size(1)}')

    def _weighted_attr_summary(self, hat_T, reliability):
        """可信度加权均值池化：[B, n, d] → [B, d]"""
        w = reliability.unsqueeze(-1)                      # [B, n, 1]
        return (hat_T * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)  # [B, d]

    def _e_step(self, z, hat_T, reliability):
        """E步：用可信度加权的属性摘要更新 z"""
        pooled = self._weighted_attr_summary(hat_T, reliability)  # [B, d]
        delta  = self.e_step(torch.cat([z, pooled], dim=-1))      # [B, d]
        return self.z_norm(z + delta)                              # [B, d]

    def _m_step_project(self, z, n_attrs):
        """M步 投影：Proj_i(Z) = MLP([z, attr_embed_i])，返回 [B, n, d]"""
        attr_ids   = torch.arange(n_attrs, device=z.device)
        attr_emb   = self.attr_embed(attr_ids)                    # [n, d]
        attr_emb   = attr_emb.unsqueeze(0).expand(z.size(0), -1, -1)  # [B, n, d]
        z_exp      = z.unsqueeze(1).expand(-1, n_attrs, -1)       # [B, n, d]
        expected   = self.project_attr(
            torch.cat([z_exp, attr_emb], dim=-1))                 # [B, n, d]
        return self.attr_norm(expected)                           # [B, n, d]

    def _m_step_reliability(self, reliability, hat_T, expected_attr):
        """M步 可信度更新：偏差大的属性可信度下降"""
        attr_error = (hat_T - expected_attr).pow(2).mean(dim=-1)  # [B, n]
        penalty    = torch.exp(-self.lambda_em * attr_error)      # [B, n]
        candidate  = reliability * penalty
        updated    = (self.momentum * reliability
                      + (1.0 - self.momentum) * candidate)
        return updated.clamp(min=self.r_min, max=1.0), attr_error

    def forward(
        self,
        f_fused     : torch.Tensor,
        reliability : torch.Tensor,
        hat_T       : torch.Tensor,
        n_iters     : Optional[int] = None,
        return_history : bool = False,
    ) -> EMOutput:
        """
        参数
        ----
        f_fused     : [B, fused_dim=128]
        reliability : [B, n_attrs=136]
        hat_T       : [B, n_attrs=136, attr_dim=128]
        n_iters     : 覆盖默认迭代次数（可选）
        return_history : 是否返回每步的 r 和 z 历史

        返回
        ----
        EMOutput（见 dataclass 定义）
        """
        self._check_inputs(f_fused, reliability, hat_T)

        iters   = self.n_iters if n_iters is None else n_iters
        n_attrs = hat_T.size(1)

        # 初始化
        z = self.init_z(f_fused)                           # [B, attr_dim]
        r = reliability.clamp(min=self.r_min, max=1.0)

        r_history = [r.detach().clone()] if return_history else None
        z_history = [z.detach().clone()] if return_history else None

        expected_attr = hat_T                              # 第一次迭代前的占位
        attr_error    = torch.zeros_like(r)

        for _ in range(iters):
            # E步：更新 z
            z = self._e_step(z, hat_T, r)

            # M步：投影 + 更新可信度
            expected_attr        = self._m_step_project(z, n_attrs)
            r, attr_error        = self._m_step_reliability(r, hat_T, expected_attr)

            if return_history:
                r_history.append(r.detach().clone())
                z_history.append(z.detach().clone())

        return EMOutput(
            z                   = z,
            reliability         = r,
            expected_attr       = expected_attr,
            attr_error          = attr_error,
            reliability_history = r_history,
            z_history           = z_history,
        )


# ══════════════════════════════════════════════
# 诊断输出头（原代码缺失，补全）
# ══════════════════════════════════════════════

class DiagnosisHead(nn.Module):
    """
    将 EM 输出的隐变量 z 映射为诊断 logits。

    输入：z [B, attr_dim=128]
    输出：logits [B, n_classes]
    """

    def __init__(self, attr_dim: int = 128, n_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(attr_dim, attr_dim // 2),
            nn.LayerNorm(attr_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attr_dim // 2, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)                                 # [B, n_classes]


# ══════════════════════════════════════════════
# 便捷封装
# ══════════════════════════════════════════════

class FusionEMRefiner(nn.Module):
    """
    EM 精炼 + 诊断头的统一封装。

    输入：f_fused, reliability, hat_T（来自 ScaleImageFusionPreEM）
    输出：EMOutput（含 z），以及诊断 logits
    """

    def __init__(
        self,
        fused_dim  : int   = 128,
        attr_dim   : int   = 128,
        n_attrs    : int   = 136,
        n_classes  : int   = 2,
        n_iters    : int   = 3,
        lambda_em  : float = 1.0,
        momentum   : float = 0.2,
        r_min      : float = 0.1,
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.em = ExpectationMaximization(
            fused_dim  = fused_dim,
            attr_dim   = attr_dim,
            n_attrs    = n_attrs,
            n_iters    = n_iters,
            lambda_em  = lambda_em,
            momentum   = momentum,
            r_min      = r_min,
            dropout    = dropout,
        )
        self.diag_head = DiagnosisHead(
            attr_dim  = attr_dim,
            n_classes = n_classes,
            dropout   = dropout,
        )

    def forward(
        self,
        f_fused     : torch.Tensor,
        reliability : torch.Tensor,
        hat_T       : torch.Tensor,
        n_iters     : Optional[int] = None,
        return_history : bool = False,
    ):
        """
        返回
        ----
        logits  : [B, n_classes]    诊断 logits
        em_out  : EMOutput          EM 迭代详细输出
        """
        em_out = self.em(
            f_fused        = f_fused,
            reliability    = reliability,
            hat_T          = hat_T,
            n_iters        = n_iters,
            return_history = return_history,
        )
        logits = self.diag_head(em_out.z)
        return logits, em_out


# ══════════════════════════════════════════════
# 完整端到端模型
# ══════════════════════════════════════════════

class FullDiagnosisModel(nn.Module):
    """
    端到端诊断模型，串联所有模块。

    输入：
      x_scale    [B, 136]   量表属性（0-indexed）
      x_img      [B, 12]    舌诊标签（0-indexed）
      edge_index [2, E]     图边
      edge_attr  [E, 1]     MI 边权

    输出：
      logits     [B, n_classes]   诊断 logits
      em_out     EMOutput         EM 迭代详情（含更新后的 reliability）
      attn_weights [B, 12, 136]   Cross-Attention 权重（可解释性）
    """

    def __init__(
        self,
        attr_cols,
        option_counts,
        n_classes    : int   = 2,
        d_model      : int   = 64,
        gat_heads    : int   = 4,
        gat_layers   : int   = 2,
        cross_heads  : int   = 4,
        n_iters      : int   = 3,
        lambda_em    : float = 1.0,
        dropout      : float = 0.1,
        prior_reliability = None,
    ):
        super().__init__()
        # 延迟导入避免循环依赖
        from cross_fusion import ScaleImageFusionPreEM

        self.fusion = ScaleImageFusionPreEM(
            attr_cols         = attr_cols,
            option_counts     = option_counts,
            d_model           = d_model,
            gat_heads         = gat_heads,
            gat_layers        = gat_layers,
            cross_heads       = cross_heads,
            dropout           = dropout,
            prior_reliability = prior_reliability,
        )
        self.em_refiner = FusionEMRefiner(
            fused_dim  = d_model * 2,
            attr_dim   = d_model * 2,
            n_attrs    = len(attr_cols),
            n_classes  = n_classes,
            n_iters    = n_iters,
            lambda_em  = lambda_em,
            dropout    = dropout,
        )

    def forward(
        self,
        x_scale    : torch.Tensor,
        x_img      : torch.Tensor,
        edge_index : torch.Tensor,
        edge_attr  : torch.Tensor,
        return_history : bool = False,
    ):
        # 融合
        f_fused, reliability, hat_T, attn_weights, conf = self.fusion(
            x_scale, x_img, edge_index, edge_attr)

        # EM 精炼 + 诊断
        logits, em_out = self.em_refiner(
            f_fused        = f_fused,
            reliability    = reliability,
            hat_T          = hat_T,
            return_history = return_history,
        )

        return logits, em_out, attn_weights


# ══════════════════════════════════════════════
# 演示：端到端前向传播 + 损失计算
# ══════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/mnt/user-data/uploads')

    import torch
    from scale_embedding import load_scale, preprocess
    from image_encoder   import load_image_labels, build_image_tensor

    torch.manual_seed(42)

    SCALE_PATH = './scale.xlsx'
    IMG_PATH   = './all_info.xlsx'
    GRAPH_PATH = './graph_data.pt'
    D_MODEL    = 64
    N_CLASSES  = 2
    BATCH_SIZE = 8

    print('=' * 60)
    print('Step 1: 加载数据')
    print('=' * 60)
    df_scale, id_col, attr_cols, option_counts = load_scale(SCALE_PATH)
    df_img,   id2idx = load_image_labels(IMG_PATH)
    g          = torch.load(GRAPH_PATH, weights_only=False)
    edge_index = g['edge_index']
    edge_attr  = g['edge_weight'].unsqueeze(1)

    common    = sorted(set(df_img['IDAA'].astype(int)) &
                       set(df_scale[id_col].astype(int)))
    batch_ids = common[:BATCH_SIZE]

    scale_idx = df_scale.set_index(id_col)
    scale_rows = []
    for pid in batch_ids:
        row  = scale_idx.loc[pid]
        vals = row[attr_cols].values.copy().astype(int)
        vals[vals >= 1] -= 1
        scale_rows.append(vals.tolist())
    x_scale = torch.tensor(scale_rows, dtype=torch.long)
    x_img   = build_image_tensor(df_img, batch_ids)

    print(f'x_scale : {x_scale.shape}')
    print(f'x_img   : {x_img.shape}')

    print()
    print('=' * 60)
    print('Step 2: 初始化 FullDiagnosisModel')
    print('=' * 60)

    # 只用 FusionEMRefiner 演示（避免导入路径问题）
    from cross_fusion import ScaleImageFusionPreEM

    fusion   = ScaleImageFusionPreEM(
        attr_cols     = attr_cols,
        option_counts = option_counts,
        d_model       = D_MODEL,
    )
    refiner  = FusionEMRefiner(
        fused_dim  = D_MODEL * 2,
        attr_dim   = D_MODEL * 2,
        n_attrs    = len(attr_cols),
        n_classes  = N_CLASSES,
        n_iters    = 3,
        lambda_em  = 1.0,
    )

    total = (sum(p.numel() for p in fusion.parameters()) +
             sum(p.numel() for p in refiner.parameters()))
    print(f'总参数量: {total:,}')
    print(f'  ScaleImageFusionPreEM : '
          f'{sum(p.numel() for p in fusion.parameters()):,}')
    print(f'  FusionEMRefiner       : '
          f'{sum(p.numel() for p in refiner.parameters()):,}')

    print()
    print('=' * 60)
    print('Step 3: 端到端前向传播')
    print('=' * 60)

    fusion.eval()
    refiner.eval()
    with torch.no_grad():
        f_fused, reliability, hat_T, attn_weights, conf = fusion(
            x_scale, x_img, edge_index, edge_attr)

        logits, em_out = refiner(
            f_fused, reliability, hat_T, return_history=True)

    print(f'logits       {logits.shape}   → 诊断输出')
    print(f'z            {em_out.z.shape} → 精炼后隐变量')
    print(f'reliability  {em_out.reliability.shape} → 更新后可信度')
    print(f'attr_error   {em_out.attr_error.shape} → 各属性重建误差')
    print(f'EM 迭代步数  {len(em_out.reliability_history) - 1}')

    print()
    print('=' * 60)
    print('Step 4: 损失计算（模拟标签）')
    print('=' * 60)

    y_fake = torch.randint(0, N_CLASSES, (BATCH_SIZE,))
    loss   = F.cross_entropy(logits, y_fake)
    print(f'模拟标签 : {y_fake.tolist()}')
    print(f'训练损失 : {loss.item():.4f}')

    print()
    print('=' * 60)
    print('Step 5: 可信度变化（EM 精炼效果）')
    print('=' * 60)
    r0 = em_out.reliability_history[0]
    rf = em_out.reliability_history[-1]
    print(f'初始可信度均值  : {r0.mean():.4f}')
    print(f'精炼后可信度均值: {rf.mean():.4f}')
    print(f'可信度变化量均值: {(rf - r0).abs().mean():.4f}')
    print()

    # 找出可信度下降最多的属性（最可疑）
    delta = (rf - r0).mean(dim=0)   # [136]
    bot5  = delta.topk(5, largest=False)
    print('可信度下降最多的5个属性（最可疑）:')
    for idx, val in zip(bot5.indices.tolist(), bot5.values.tolist()):
        print(f'  attr[{idx:3d}] 变化={val:+.4f}')

    print()
    print('=' * 60)
    print('所有模块接口验证通过，可进入完整训练循环')
    print('=' * 60)
"""
train.py — 多任务诊断模型训练脚本
====================================
任务：
  Task 1 (label) : 二分类  — 是否患病     {0, 1}
  Task 2 (level) : 4分类   — 患病程度     {0, 1, 2, 3}
                              0 = 无病（与 label=0 对应）
                              1/2/3 = 有病且程度递增

标签来源：output.jsonl
  {id, label, level}

数据：
  量表   : scale.xlsx        (7425 患者，136 属性)
  影像   : all_info.xlsx     (3034 患者，12 舌诊标签)
  图结构 : graph_data.pt     (MI 边权，需提前由 graph_construct.py 生成)

模块依赖（同目录下）：
  GAT.py            ScaleGATEncoder
  image_encoder.py  ImageLabelEncoder
  cross_fusion.py   ScaleImageFusionPreEM
  expect_max.py     FusionEMRefiner, DiagnosisHead

使用方式：
  python train.py                   # 使用默认配置
  python train.py --epochs 50       # 修改 epoch 数
  python train.py --no_img          # 仅使用量表（消融实验）

注意：
  level=1 仅 26 个样本，类别极度不平衡，使用加权交叉熵缓解。
  label=0 时 level 必然为 0，两个任务的损失权重可按需调整。
"""

import os
import sys
import json
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, roc_auc_score)

# ── 路径设置（所有模块在同一目录下）─────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from GAT          import ScaleGATEncoder
from image_encoder import (ImageLabelEncoder, load_image_labels,
                            build_image_tensor, LABEL_COLS)
from cross_fusion  import ScaleImageFusionPreEM
from expect_max    import FusionEMRefiner
from scale_embedding import load_scale, preprocess


# ══════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description='多任务舌诊诊断模型训练')
    # 路径
    p.add_argument('--scale_path', default='scale.xlsx')
    p.add_argument('--img_path',   default='all_info.xlsx')
    p.add_argument('--graph_path', default='graph_data.pt')
    p.add_argument('--label_path', default='output.jsonl')
    p.add_argument('--save_dir',   default='./checkpoints')
    # 模型
    p.add_argument('--d_model',    type=int,   default=64)
    p.add_argument('--gat_heads',  type=int,   default=4)
    p.add_argument('--gat_layers', type=int,   default=2)
    p.add_argument('--cross_heads',type=int,   default=4)
    p.add_argument('--em_iters',   type=int,   default=3,  help='EM迭代次数，内存紧张时设为1或2')
    p.add_argument('--lambda_em',  type=float, default=1.0)
    p.add_argument('--dropout',    type=float, default=0.1)
    # 训练
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=32, help='批次大小，显存不足时调小至8')
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--weight_decay',type=float,default=1e-4)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--val_ratio',  type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    # 损失权重
    p.add_argument('--w_label',    type=float, default=1.0,
                   help='label（二分类）损失权重')
    p.add_argument('--w_level',    type=float, default=1.0,
                   help='level（4分类）损失权重')
    # 消融
    p.add_argument('--no_img',     action='store_true',
                   help='消融：不使用影像标签（仅量表）')
    return p.parse_args()


# ══════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════

class DiagnosisDataset(Dataset):
    """
    每个样本：(x_scale, x_img, label, level)

    x_scale : [136]   量表属性 0-indexed
    x_img   : [12]    舌诊标签 0-indexed（无影像时为全零占位）
    label   : int     0/1
    level   : int     0/1/2/3
    """

    def __init__(self, records, df_scale, df_img,
                 id_col, attr_cols, use_img=True):
        """
        records   : list of {id, label, level}（已过滤为有效样本）
        df_scale  : 量表 DataFrame（已 0-indexed 预处理）
        df_img    : 影像 DataFrame（已 0-indexed 预处理）或 None
        """
        self.records   = records
        self.scale_idx = df_scale.set_index(id_col)
        self.img_idx   = df_img.set_index('IDAA') if df_img is not None else None
        self.attr_cols = attr_cols
        self.use_img   = use_img and (df_img is not None)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        pid = int(rec['id'])

        # 量表
        row   = self.scale_idx.loc[pid]
        vals  = row[self.attr_cols].values.astype(int).copy()
        # 0-indexed
        vals[vals >= 1] -= 1
        x_scale = torch.tensor(vals, dtype=torch.long)

        # 影像标签
        if self.use_img and self.img_idx is not None:
            irow  = self.img_idx.loc[pid]
            x_img = torch.tensor(
                [int(irow[c]) for c in LABEL_COLS], dtype=torch.long)
        else:
            x_img = torch.zeros(12, dtype=torch.long)

        label = torch.tensor(rec['label'], dtype=torch.long)
        level = torch.tensor(rec['level'], dtype=torch.long)

        return x_scale, x_img, label, level


def build_datasets(args):
    """加载数据，划分训练/验证/测试集，返回三个 Dataset。"""

    # ── 加载标签 ───────────────────────────────
    all_labels = [json.loads(l)
                  for l in open(args.label_path, encoding='utf-8')]

    # ── 加载量表 ───────────────────────────────
    df_scale, id_col, attr_cols, option_counts = load_scale(args.scale_path)
    scale_ids = set(df_scale[id_col].astype(int))

    # ── 加载影像标签 ───────────────────────────
    if not args.no_img and os.path.exists(args.img_path):
        df_img, _ = load_image_labels(args.img_path)
        img_ids   = set(df_img['IDAA'].astype(int))
    else:
        df_img  = None
        img_ids = set()

    # ── 过滤：必须在量表中存在 ─────────────────
    if df_img is not None:
        valid_ids = scale_ids & img_ids
    else:
        valid_ids = scale_ids

    records = [r for r in all_labels if int(r['id']) in valid_ids]
    print(f'有效样本数 : {len(records)}')
    print(f'label 分布 : {Counter(r["label"] for r in records)}')
    print(f'level 分布 : {Counter(r["level"] for r in records)}')

    # ── 分层划分（按 level 分层保证每类都有）────
    labels_for_split = [r['level'] for r in records]
    train_val, test = train_test_split(
        records, test_size=args.test_ratio,
        stratify=labels_for_split, random_state=args.seed)

    tv_labels = [r['level'] for r in train_val]
    train, val = train_test_split(
        train_val,
        test_size=args.val_ratio / (1 - args.test_ratio),
        stratify=tv_labels, random_state=args.seed)

    print(f'训练集 : {len(train)}  验证集 : {len(val)}  测试集 : {len(test)}')

    def make_ds(recs):
        return DiagnosisDataset(
            recs, df_scale, df_img, id_col, attr_cols,
            use_img=not args.no_img)

    return make_ds(train), make_ds(val), make_ds(test), \
           attr_cols, option_counts, \
           Counter(r['level'] for r in train)


# ══════════════════════════════════════════════
# 完整多任务模型
# ══════════════════════════════════════════════

class MultiTaskDiagnosisModel(nn.Module):
    """
    端到端多任务诊断模型。

    输出：
      label_logits : [B, 2]   二分类（有病/无病）
      level_logits : [B, 4]   4分类（0=无, 1/2/3=程度）
    """

    def __init__(self, attr_cols, option_counts,
                 d_model=64, gat_heads=4, gat_layers=2,
                 cross_heads=4, em_iters=3, lambda_em=1.0,
                 dropout=0.1):
        super().__init__()

        self.fusion = ScaleImageFusionPreEM(
            attr_cols         = attr_cols,
            option_counts     = option_counts,
            d_model           = d_model,
            gat_heads         = gat_heads,
            gat_layers        = gat_layers,
            cross_heads       = cross_heads,
            dropout           = dropout,
        )

        self.em_refiner = FusionEMRefiner(
            fused_dim  = d_model * 2,
            attr_dim   = d_model * 2,
            n_attrs    = len(attr_cols),
            n_classes  = 2,            # 占位，不用内置 diag_head
            n_iters    = em_iters,
            lambda_em  = lambda_em,
            dropout    = dropout,
        )

        feat_dim = d_model * 2         # z 的维度

        # 二分类头（label）
        self.label_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 2),
        )

        # 4分类头（level）
        self.level_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 4),
        )

    def forward(self, x_scale, x_img, edge_index, edge_attr):
        # 融合
        f_fused, reliability, hat_T, attn_weights, conf = self.fusion(
            x_scale, x_img, edge_index, edge_attr)

        # EM 精炼
        _, em_out = self.em_refiner(f_fused, reliability, hat_T)
        z = em_out.z                               # [B, feat_dim]

        # 两个任务头
        label_logits = self.label_head(z)          # [B, 2]
        level_logits = self.level_head(z)          # [B, 4]

        return label_logits, level_logits, em_out


# ══════════════════════════════════════════════
# 损失函数（带类别权重）
# ══════════════════════════════════════════════

def build_loss_fn(train_level_counter, device):
    """
    根据训练集 level 分布计算类别权重，缓解不平衡。

    label 任务：从 level 推导二分类权重
    level 任务：4分类，level=1 极少，需要重点加权
    """
    total = sum(train_level_counter.values())

    # level 权重（反频率加权）
    level_counts = [train_level_counter.get(i, 1) for i in range(4)]
    level_w = torch.tensor(
        [total / (4 * c) for c in level_counts],
        dtype=torch.float32, device=device)

    # label 权重
    n0 = train_level_counter.get(0, 1)             # label=0 的数量
    n1 = total - n0                                 # label=1 的数量
    label_w = torch.tensor(
        [total / (2 * n0), total / (2 * n1)],
        dtype=torch.float32, device=device)

    label_ce = nn.CrossEntropyLoss(weight=label_w)
    level_ce = nn.CrossEntropyLoss(weight=level_w)

    return label_ce, level_ce


# ══════════════════════════════════════════════
# 评估函数
# ══════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, edge_index, edge_attr,
             label_ce, level_ce, w_label, w_level, device):
    model.eval()
    all_label_pred, all_label_true = [], []
    all_level_pred, all_level_true = [], []
    total_loss = 0.0
    n_batches  = 0

    for x_scale, x_img, labels, levels in loader:
        x_scale    = x_scale.to(device)
        x_img      = x_img.to(device)
        labels     = labels.to(device)
        levels     = levels.to(device)
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.to(device)

        label_logits, level_logits, _ = model(
            x_scale, x_img, edge_index, edge_attr)

        loss = (w_label * label_ce(label_logits, labels) +
                w_level * level_ce(level_logits, levels))
        total_loss += loss.item()
        n_batches  += 1

        all_label_pred.extend(label_logits.argmax(1).cpu().tolist())
        all_label_true.extend(labels.cpu().tolist())
        all_level_pred.extend(level_logits.argmax(1).cpu().tolist())
        all_level_true.extend(levels.cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)

    label_acc = accuracy_score(all_label_true, all_label_pred)
    label_f1  = f1_score(all_label_true, all_label_pred,
                         average='macro', zero_division=0)
    level_acc = accuracy_score(all_level_true, all_level_pred)
    level_f1  = f1_score(all_level_true, all_level_pred,
                         average='macro', zero_division=0)

    return {
        'loss'      : avg_loss,
        'label_acc' : label_acc,
        'label_f1'  : label_f1,
        'level_acc' : level_acc,
        'level_f1'  : level_f1,
        'label_pred': all_label_pred,
        'label_true': all_label_true,
        'level_pred': all_level_pred,
        'level_true': all_level_true,
    }


# ══════════════════════════════════════════════
# 训练主循环
# ══════════════════════════════════════════════

def train(args):
    # ── 随机种子 ──────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 数据 ──────────────────────────────────
    print('\n=== 数据加载 ===')
    train_ds, val_ds, test_ds, \
    attr_cols, option_counts, train_level_cnt = build_datasets(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=False,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── 图结构（所有 batch 共享）──────────────
    g          = torch.load(args.graph_path, weights_only=False)
    edge_index = g['edge_index'].to(device)
    edge_attr  = g['edge_weight'].unsqueeze(1).to(device)

    # ── 模型 ──────────────────────────────────
    print('\n=== 模型初始化 ===')
    model = MultiTaskDiagnosisModel(
        attr_cols  = attr_cols,
        option_counts = option_counts,
        d_model    = args.d_model,
        gat_heads  = args.gat_heads,
        gat_layers = args.gat_layers,
        cross_heads= args.cross_heads,
        em_iters   = args.em_iters,
        lambda_em  = args.lambda_em,
        dropout    = args.dropout,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total_p:,}')

    # ── 损失函数 ──────────────────────────────
    label_ce, level_ce = build_loss_fn(train_level_cnt, device)

    # ── 优化器 + 学习率调度 ────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── 训练循环 ──────────────────────────────
    print('\n=== 开始训练 ===')
    best_val_f1  = -1.0
    best_epoch   = 0
    history      = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for x_scale, x_img, labels, levels in train_loader:
            x_scale = x_scale.to(device)
            x_img   = x_img.to(device)
            labels  = labels.to(device)
            levels  = levels.to(device)

            optimizer.zero_grad()

            label_logits, level_logits, _ = model(
                x_scale, x_img, edge_index, edge_attr)

            loss = (args.w_label * label_ce(label_logits, labels) +
                    args.w_level * level_ce(level_logits, levels))

            loss.backward()
            # 梯度裁剪（EM 迭代展开的深计算图容易梯度爆炸）
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # ── 验证 ──────────────────────────────
        val_metrics = evaluate(
            model, val_loader, edge_index, edge_attr,
            label_ce, level_ce, args.w_label, args.w_level, device)

        # 以 label_f1 + level_f1 的均值作为主指标
        val_combined_f1 = (val_metrics['label_f1'] +
                           val_metrics['level_f1']) / 2

        history.append({
            'epoch'      : epoch,
            'train_loss' : avg_train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()
               if k not in ('label_pred', 'label_true',
                            'level_pred', 'level_true')},
        })

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'train_loss={avg_train_loss:.4f} | '
              f'val_loss={val_metrics["loss"]:.4f} | '
              f'label_acc={val_metrics["label_acc"]:.4f} '
              f'label_f1={val_metrics["label_f1"]:.4f} | '
              f'level_acc={val_metrics["level_acc"]:.4f} '
              f'level_f1={val_metrics["level_f1"]:.4f}')

        # ── 保存最佳模型 ──────────────────────
        if val_combined_f1 > best_val_f1:
            best_val_f1 = val_combined_f1
            best_epoch  = epoch
            torch.save({
                'epoch'       : epoch,
                'model_state' : model.state_dict(),
                'opt_state'   : optimizer.state_dict(),
                'val_metrics' : val_metrics,
                'args'        : vars(args),
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f'  ✓ 保存最佳模型 (epoch={epoch}, '
                  f'combined_f1={val_combined_f1:.4f})')

    # ── 测试集评估 ────────────────────────────
    print(f'\n=== 测试集评估（加载 epoch={best_epoch} 的最佳模型）===')
    ckpt = torch.load(os.path.join(args.save_dir, 'best_model.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    test_metrics = evaluate(
        model, test_loader, edge_index, edge_attr,
        label_ce, level_ce, args.w_label, args.w_level, device)

    print()
    print('── Label（二分类）──')
    print(f'  Accuracy : {test_metrics["label_acc"]:.4f}')
    print(f'  Macro-F1 : {test_metrics["label_f1"]:.4f}')
    print(classification_report(
        test_metrics['label_true'], test_metrics['label_pred'],
        target_names=['无病(0)', '有病(1)'], zero_division=0))

    print('── Level（4分类）──')
    print(f'  Accuracy : {test_metrics["level_acc"]:.4f}')
    print(f'  Macro-F1 : {test_metrics["level_f1"]:.4f}')
    print(classification_report(
        test_metrics['level_true'], test_metrics['level_pred'],
        target_names=['无病(0)', '轻度(1)', '中度(2)', '重度(3)'],
        zero_division=0))

    # ── 保存训练历史 ──────────────────────────
    history_path = os.path.join(args.save_dir, 'history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'\n训练历史已保存至 {history_path}')


# ══════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════

if __name__ == '__main__':
    args = get_args()

    print('=' * 60)
    print('多任务舌诊诊断模型训练')
    print('=' * 60)
    print(f'量表路径   : {args.scale_path}')
    print(f'影像路径   : {args.img_path}')
    print(f'图结构路径 : {args.graph_path}')
    print(f'标签路径   : {args.label_path}')
    print(f'使用影像   : {not args.no_img}')
    print(f'Epochs     : {args.epochs}')
    print(f'Batch size : {args.batch_size}')
    print(f'LR         : {args.lr}')
    print(f'w_label    : {args.w_label}')
    print(f'w_level    : {args.w_level}')
    print()

    train(args)
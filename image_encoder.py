"""
影像标签编码器（基于真实舌诊数据）
====================================
数据来源：all_info.xlsx
  IDAA              : 患者ID（与量表 唯一识别编码 对应）
  tongue apperance  : 舌色  int {1..6}
  dark tongue       : 暗舌  bool
  red spot          : 红点  bool
  tongue body       : 舌体  int {1,2,3}
  tooth marks       : 齿痕  bool
  small crack       : 小裂纹 bool
  central big crack : 中央大裂 bool
  tongue coating    : 舌苔  int {1..5}
  coating color     : 苔色  int {1,2,3,5} → 0-indexed {0,1,2,3}
  tongue quality    : 苔质  int {1..5}
  flower-stripping  : 花剥苔 bool
  pricking tongue   : 芒刺  bool

编码策略：
  bool 列  → 0/1，查独立 Embedding(2, d_model)
  int  列  → 0-indexed，查独立 Embedding(k, d_model)
  token_j  = label_id_emb[j] + val_emb[j](x_j)
  GT 模式置信度全为 1.0

输出接口：
  F_img : [B, 12, d_model]   每标签 token（Cross-Attention Query）
  f_img : [B, d_model]       整体影像表示
  conf  : [B, 12]            置信度（GT 全为 1.0）
"""

import torch
import torch.nn as nn
import pandas as pd


# ── 标签元信息（顺序固定，不可随意更改）────────────────────
LABEL_META = [
    # (列名,                    选项数,  类型,    中文名)
    ('tongue apperance',        6,     'int',   '舌色'),
    ('dark tongue',             2,     'bool',  '暗舌'),
    ('red spot',                2,     'bool',  '红点'),
    ('tongue body',             3,     'int',   '舌体'),
    ('tooth marks',             2,     'bool',  '齿痕'),
    ('small crack',             2,     'bool',  '小裂纹'),
    ('central big crack',       2,     'bool',  '中央大裂'),
    ('tongue coating',          5,     'int',   '舌苔'),
    ('coating color',           4,     'int',   '苔色'),   # {1,2,3,5}→{0,1,2,3}
    ('tongue quality',          5,     'int',   '苔质'),
    ('flower-stripping tongue', 2,     'bool',  '花剥苔'),
    ('pricking tongue',         2,     'bool',  '芒刺'),
]

N_LABELS    = len(LABEL_META)                   # 12
LABEL_COLS  = [m[0] for m in LABEL_META]
LABEL_NAMES = [m[3] for m in LABEL_META]
OPTION_CNTS = [m[1] for m in LABEL_META]

# coating color 原始值到 0-indexed 的映射
_CC_MAP = {1: 0, 2: 1, 3: 2, 5: 3}


# ══════════════════════════════════════════════
# 数据加载与预处理
# ══════════════════════════════════════════════

def load_image_labels(path: str):
    """
    加载 all_info.xlsx，统一完成 0-indexed 预处理。

    所有标签列均转换为 0-indexed 整数，后续直接送入 Embedding。

    返回
    ----
    df_img : 预处理后的 DataFrame（IDAA 保留，标签列均为 0-indexed int）
    id2idx : {patient_id(int) -> 行号} 查找字典
    """
    df = pd.read_excel(path)

    for col, _, typ, _ in LABEL_META:
        if typ == 'bool':
            df[col] = df[col].astype(int)              # True/False → 1/0
        elif col == 'coating color':
            df[col] = df[col].map(_CC_MAP).astype(int) # {1,2,3,5} → {0,1,2,3}
        else:
            df[col] = df[col].astype(int) - 1          # 1-indexed → 0-indexed

    id2idx = {int(pid): i for i, pid in enumerate(df['IDAA'])}
    return df, id2idx


def build_image_tensor(df_img: pd.DataFrame,
                       patient_ids) -> torch.LongTensor:
    """
    按患者 ID 列表顺序构建影像标签张量。
    df_img 中的标签列已经是 0-indexed，直接读取即可。

    参数
    ----
    df_img      : load_image_labels 返回的 DataFrame
    patient_ids : 患者 ID 的可迭代序列（int 或可转 int）

    返回
    ----
    X_img : [N, 12]  LongTensor，0-indexed
    """
    indexed = df_img.set_index('IDAA')
    rows = []
    for pid in patient_ids:
        row = indexed.loc[int(pid)]
        rows.append([int(row[c]) for c in LABEL_COLS])
    return torch.tensor(rows, dtype=torch.long)


# ══════════════════════════════════════════════
# 影像标签编码器
# ══════════════════════════════════════════════

class ImageLabelEncoder(nn.Module):
    """
    将 12 个舌诊标签编码为 token 序列。

    参数
    ----
    d_model : token 维度（须与量表侧 ScaleGATEncoder 一致）
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model  = d_model
        self.n_labels = N_LABELS

        # 标签身份嵌入（类比量表侧的 attr_emb）
        self.label_id_emb = nn.Embedding(N_LABELS, d_model)

        # 每个标签独立的值嵌入表（类比量表侧的 option_embs）
        self.val_embs = nn.ModuleList([
            nn.Embedding(k, d_model) for k in OPTION_CNTS
        ])

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # 固定索引 buffer
        self.register_buffer('label_idx', torch.arange(N_LABELS))

    def forward(self, x: torch.LongTensor):
        """
        参数
        ----
        x : [B, 12]  0-indexed 标签值（由 build_image_tensor 生成）

        返回
        ----
        F_img : [B, 12, d_model]  每标签 token
        f_img : [B, d_model]      整体影像表示（均值 pooling）
        conf  : [B, 12]           置信度（GT 模式全为 1.0）
        """
        B = x.size(0)

        # 标签身份嵌入
        id_emb = self.label_id_emb(self.label_idx)        # [12, d]
        id_emb = id_emb.unsqueeze(0).expand(B, -1, -1)    # [B, 12, d]

        # 值嵌入（每标签查各自的嵌入表）
        val_list = [self.val_embs[j](x[:, j])
                    for j in range(self.n_labels)]
        val_emb  = torch.stack(val_list, dim=1)            # [B, 12, d]

        # 相加 + 投影
        F_img = self.out_proj(val_emb + id_emb)            # [B, 12, d]

        # GT 模式：置信度全 1
        conf  = torch.ones(B, self.n_labels,
                           device=x.device)                # [B, 12]

        # 整体表示（均值 pooling）
        f_img = F_img.mean(dim=1)                          # [B, d]

        return F_img, f_img, conf


# ══════════════════════════════════════════════
# 演示
# ══════════════════════════════════════════════

if __name__ == '__main__':

    IMG_PATH   = './all_info.xlsx'
    D_MODEL    = 64
    BATCH_SIZE = 8

    print('=' * 55)
    print('Step 1: 加载影像标签数据')
    print('=' * 55)
    df_img, id2idx = load_image_labels(IMG_PATH)
    print(f'影像患者数: {len(df_img)}')
    print(f'标签数量  : {N_LABELS}')
    print()
    print('标签列表:')
    for j, (col, k, typ, name) in enumerate(LABEL_META):
        print(f'  [{j:2d}] {name:6s}  {typ}  {k}选项  ← {col}')

    print()
    print('=' * 55)
    print('Step 2: 构建批次张量')
    print('=' * 55)
    batch_ids = list(df_img['IDAA'])[:BATCH_SIZE]
    X_img     = build_image_tensor(df_img, batch_ids)
    print(f'batch_ids  : {batch_ids}')
    print(f'X_img shape: {X_img.shape}')
    print()
    print('示例（患者0）:')
    for j, (_, _, _, name) in enumerate(LABEL_META):
        print(f'  {name}: {X_img[0, j].item()}')

    print()
    print('=' * 55)
    print('Step 3: 编码器前向传播')
    print('=' * 55)
    encoder = ImageLabelEncoder(d_model=D_MODEL)
    print(f'总参数量: {sum(p.numel() for p in encoder.parameters()):,}')
    print(f'  label_id_emb : '
          f'{sum(p.numel() for p in encoder.label_id_emb.parameters()):,}')
    print(f'  val_embs     : '
          f'{sum(p.numel() for e in encoder.val_embs for p in e.parameters()):,}')
    print(f'  out_proj     : '
          f'{sum(p.numel() for p in encoder.out_proj.parameters()):,}')

    encoder.eval()
    with torch.no_grad():
        F_img, f_img, conf = encoder(X_img)

    print()
    print(f'F_img : {F_img.shape}  → Cross-Attention Query')
    print(f'f_img : {f_img.shape}  → 整体影像表示')
    print(f'conf  : {conf.shape}   → 置信度（GT 全为 {conf.mean():.1f}）')
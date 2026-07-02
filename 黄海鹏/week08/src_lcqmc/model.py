import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig, BertModel


# ── BiEncoder ─────────────────────────────────────────────────────────────

class BiEncoder(nn.Module):
    def __init__(self, bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
        super().__init__()
        assert pool in ("cls", "mean", "max"), f"pool 须为 cls/mean/max，收到: {pool}"

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        self.pool    = pool
        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids, attention_mask, token_type_ids):
        """
        单句编码，返回 L2 归一化后的句向量 [B, H]

        L2 归一化后：cosine_sim(u, v) == dot(u, v)
        可用矩阵乘法批量计算所有两两相似度，适合向量检索场景（如 RAG）
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        vec = self._pool(out.last_hidden_state, attention_mask)  # [B, H]
        vec = self.dropout(vec)
        return F.normalize(vec, p=2, dim=-1)

    def forward(self, batch_a, batch_b):
        """返回 (emb_a, emb_b)，各形状 [B, H]，可直接计算余弦相似度"""
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b

    def _pool(self, last_hidden, attention_mask):
        if self.pool == "cls":
            return last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        if self.pool == "mean":
            sum_h = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_h / count

        if self.pool == "max":
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values


# ── CrossEncoder ──────────────────────────────────────────────────────────

class CrossEncoder(nn.Module):
    def __init__(self, bert_path, dropout=0.1, num_hidden_layers=None):
        super().__init__()

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        hidden_size  = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """返回 logits [B, 2]，未经 softmax（CrossEntropyLoss 内部处理）"""
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_vec = out.last_hidden_state[:, 0, :]  # [B, H]
        cls_vec = self.dropout(cls_vec)
        return self.classifier(cls_vec)            # [B, 2]


# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    """构建 BiEncoder 并打印参数量。"""
    model = BiEncoder(bert_path, pool=pool, dropout=dropout,
                      num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"BiEncoder (pool={pool}, layers={num_hidden_layers or 12})")
    return model


def build_crossencoder(bert_path, dropout=0.1, num_hidden_layers=None):
    """构建 CrossEncoder 并打印参数量。"""
    model = CrossEncoder(bert_path, dropout=dropout,
                         num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"CrossEncoder (layers={num_hidden_layers or 12})")
    return model


def _print_param_info(model, name):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert  = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: {name}")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")

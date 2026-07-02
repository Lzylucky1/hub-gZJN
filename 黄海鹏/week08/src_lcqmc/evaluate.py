
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from transformers import BertTokenizer

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "lcqmc"
BERT_PATH  = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
FIG_DIR    = OUTPUT_DIR / "figures"


# ── BiEncoder 评估 ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_biencoder(model, loader, device, find_threshold=True, threshold=0.5):
    """
    BiEncoder 评估：计算每对句子的余弦相似度，然后在 val 集上搜索最优阈值。

    返回 dict：
      similarities : list[float]  每对的余弦相似度
      labels       : list[int]    真实标签
      accuracy     : float        最优阈值下的准确率
      f1           : float        最优阈值下的 F1（weighted）
      threshold    : float        最优阈值（若 find_threshold=True）
      auc          : float        ROC-AUC（不依赖阈值）
    """
    model.eval()
    all_sims, all_labels = [], []

    for batch in loader:
        batch_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids":      batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        all_sims.extend(sims)
        all_labels.extend(batch["label"].tolist())

    sims   = np.array(all_sims)
    labels = np.array(all_labels)

    if find_threshold:
        threshold = _find_best_threshold(sims, labels)

    preds    = (sims >= threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, average="weighted", zero_division=0)

    # AUC：若 labels 只有一类（如 LCQMC test 全为 0），跳过
    try:
        auc = roc_auc_score(labels, sims)
    except ValueError:
        auc = float("nan")

    return {
        "similarities": all_sims,
        "labels":       all_labels,
        "accuracy":     accuracy,
        "f1":           f1,
        "threshold":    threshold,
        "auc":          auc,
    }


def _find_best_threshold(sims, labels):
    """枚举 [0.0, 1.0] 区间 101 个候选阈值，返回使 weighted-F1 最高的那个。"""
    best_f1, best_thresh = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims >= t).astype(int)
        f1    = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t
    return float(best_thresh)


# ── CrossEncoder 评估 ─────────────────────────────────────────────────────

@torch.no_grad()
def eval_crossencoder(model, loader, device):
    """
    CrossEncoder 评估：与分类任务完全一致，取 argmax 得预测标签。

    返回 dict：
      accuracy : float
      f1       : float（weighted）
      logits   : list[list[float]]
      labels   : list[int]
    """
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"]

        logits = model(input_ids, attention_mask, token_type_ids).cpu()
        all_logits.extend(logits.tolist())
        all_labels.extend(labels.tolist())

    preds    = np.argmax(all_logits, axis=1)
    labels   = np.array(all_labels)
    accuracy = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "logits":   all_logits,
        "labels":   all_labels,
        "accuracy": accuracy,
        "f1":       f1,
    }


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_similarity_distribution(sims, labels, threshold, save_path, title="相似度分布"):
    """
    绘制正/负样本的余弦相似度分布直方图（BiEncoder 专属）。

    教学价值：
      - 理想情况：正样本集中在 [threshold, 1]，负样本集中在 [-1, threshold)
      - 分布重叠越少 → 模型判别能力越强
    """
    sims   = np.array(sims)
    labels = np.array(labels)

    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_sims, bins=50, alpha=0.6, label=f"正样本 (n={len(pos_sims)})",
            color="#2196F3", density=True)
    ax.hist(neg_sims, bins=50, alpha=0.6, label=f"负样本 (n={len(neg_sims)})",
            color="#F44336", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"最优阈值 = {threshold:.2f}")
    ax.set_xlabel("余弦相似度")
    ax.set_ylabel("密度")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def plot_similarity_comparison(results, save_path):
    """
    绘制多个 BiEncoder 模型的相似度分布对比图。
    
    参数:
      results : list[dict]  每个元素包含 {name, similarities, labels, threshold}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    
    for idx, (res, color) in enumerate(zip(results, colors)):
        sims   = np.array(res["similarities"])
        labels = np.array(res["labels"])
        pos_sims = sims[labels == 1]
        neg_sims = sims[labels == 0]
        
        axes[0].hist(pos_sims, bins=30, alpha=0.5, label=f"{res['name']} (正样本)",
                     color=color, density=True)
        axes[1].hist(neg_sims, bins=30, alpha=0.5, label=f"{res['name']} (负样本)",
                     color=color, density=True)
    
    for i, title in enumerate(["正样本相似度分布", "负样本相似度分布"]):
        axes[i].set_xlabel("余弦相似度")
        axes[i].set_ylabel("密度")
        axes[i].set_title(title)
        axes[i].legend()
        axes[i].grid(axis="y", alpha=0.3)
    
    fig.suptitle(f"BiEncoder 模型对比（{len(results)} 个策略）", fontsize=12)
    fig.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  对比图已保存 → {save_path}")


def print_comparison_table(results):
    """打印多个模型的评估结果对比表格。"""
    print(f"\n{'='*70}")
    print(f"{'模型':<20} {'阈值':>8} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print(f"{'='*70}")
    for res in results:
        print(f"{res['name']:<20} {res['threshold']:>8.2f} "
              f"{res['accuracy']:>10.4f} {res['f1']:>10.4f} {res['auc']:>10.4f}")
    print(f"{'='*70}")
    
    best_f1 = max(results, key=lambda r: r["f1"])
    print(f"\n最优 F1 模型: {best_f1['name']} (F1={best_f1['f1']:.4f})")


# ── 独立运行入口 ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="文本匹配模型评估")
    parser.add_argument("--model_type", required=True, choices=["biencoder", "crossencoder"])
    parser.add_argument("--ckpt",       required=True, type=str, nargs="+",
                        help="checkpoint .pt 文件路径（支持传入多个进行对比）")
    parser.add_argument("--data_dir",   default=str(DATA_DIR), type=str)
    parser.add_argument("--bert_path",  default=str(BERT_PATH), type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_length", default=64,  type=int, help="BiEncoder 单句最大长度")
    parser.add_argument("--split",      default="validation", choices=["validation", "test"],
                        help="评估的数据集分割（LCQMC test 无正样本，建议用 validation）")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"加载 checkpoint(s): {args.ckpt}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.model_type == "biencoder":
        from model import build_biencoder
        from dataset import PairDataset
        from torch.utils.data import DataLoader

        ds     = PairDataset(data_path, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        results = []
        for ckpt_path in args.ckpt:
            print(f"\n{'='*50}")
            print(f"处理 checkpoint: {ckpt_path}")
            
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            saved_args = ckpt.get("args", {})
            
            model = build_biencoder(
                bert_path=args.bert_path,
                pool=saved_args.get("pool", "mean"),
                num_hidden_layers=saved_args.get("num_hidden_layers"),
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])

            metrics = eval_biencoder(model, loader, device)
            
            model_name = Path(ckpt_path).stem
            print(f"模型: {model_name}")
            print(f"  最优阈值: {metrics['threshold']:.2f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1      : {metrics['f1']:.4f}")
            print(f"  AUC     : {metrics['auc']:.4f}")

            preds = (np.array(metrics["similarities"]) >= metrics["threshold"]).astype(int)
            print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")

            results.append({
                "name":          model_name,
                "similarities":  metrics["similarities"],
                "labels":        metrics["labels"],
                "threshold":     metrics["threshold"],
                "accuracy":      metrics["accuracy"],
                "f1":            metrics["f1"],
                "auc":           metrics["auc"],
            })

            plot_similarity_distribution(
                metrics["similarities"], metrics["labels"], metrics["threshold"],
                save_path=FIG_DIR / f"biencoder_{model_name}_{args.split}_sim_dist.png",
                title=f"BiEncoder {model_name} 相似度分布（{args.split}）",
            )

        if len(results) > 1:
            print_comparison_table(results)
            
            plot_similarity_comparison(
                results,
                save_path=FIG_DIR / f"biencoder_comparison_{args.split}.png",
            )

    elif args.model_type == "crossencoder":
        from model import build_crossencoder
        from dataset import CrossEncoderDataset
        from torch.utils.data import DataLoader

        for ckpt_path in args.ckpt:
            print(f"\n{'='*50}")
            print(f"处理 checkpoint: {ckpt_path}")
            
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            saved_args = ckpt.get("args", {})
            
            model = build_crossencoder(
                bert_path=args.bert_path,
                num_hidden_layers=saved_args.get("num_hidden_layers"),
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])

            ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            metrics = eval_crossencoder(model, loader, device)
            
            model_name = Path(ckpt_path).stem
            print(f"模型: {model_name}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1      : {metrics['f1']:.4f}")

            preds = np.argmax(metrics["logits"], axis=1)
            print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")


if __name__ == "__main__":
    main()

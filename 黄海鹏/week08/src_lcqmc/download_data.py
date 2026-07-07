import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SRC_DIR, "..", "data"))

random.seed(42)

def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# 统一不同数据集的字段名格式
def normalize_row(row):
    """统一字段名：sentence1 / sentence2 / label（0/1 整数）"""
    s1 = (row.get("sentence1") or row.get("text1") or row.get("query") or "")
    s2 = (row.get("sentence2") or row.get("text2") or row.get("candidate") or "")
    # score 字段在 C-MTEB / FinanceMTEB 中就是 0/1
    label = int(row.get("label", row.get("score", 0)))
    return {"sentence1": str(s1), "sentence2": str(s2), "label": label}

# 打印数据集统计信息
def print_stats(rows, split, path):
    pos = sum(1 for r in rows if r["label"] == 1)
    neg = len(rows) - pos
    print(f"  {split:12s}: {len(rows):>7,} 条  正样本 {pos:>6,}  负样本 {neg:>6,}  -> {path}")

# 预览数据集前 n 条
def preview(path, n=3):
    print(f"\n  [预览 前 {n} 条]")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            tag = "✓ 相似" if r["label"] == 1 else "✗ 不相似"
            print(f"    [{tag}] {r['sentence1']!r}  ||  {r['sentence2']!r}")


# 下载 LCQMC 数据集
def download_lcqmc():
    print(f"\n{'='*55}")
    print("下载 LCQMC（C-MTEB/LCQMC）...")
    out_dir = os.path.join(DATA_DIR, "lcqmc")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("C-MTEB/LCQMC")

    split_map = {"train": "train", "validation": "validation", "test": "test"}
    for split_out, split_in in split_map.items():
        if split_in not in ds:
            print(f"  [SKIP] {split_in} 不存在")
            continue
        rows = [normalize_row(r) for r in ds[split_in]]
        out_path = os.path.join(out_dir, f"{split_out}.jsonl")
        save_jsonl(rows, out_path)
        print_stats(rows, split_out, out_path)

    preview(os.path.join(out_dir, "train.jsonl"))

    print(f"\n{'='*55}")
    print("下载 LCQMC（clue/lcqmc）...")
    out_dir = os.path.join(DATA_DIR, "lcqmc")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("clue", "lcqmc")

    split_map = {"train": "train", "validation": "validation", "test": "test"}
    for split_out, split_in in split_map.items():
        if split_in not in ds:
            print(f"  [SKIP] {split_in} 不存在")
            continue
        rows = [normalize_row(r) for r in ds[split_in]]
        out_path = os.path.join(out_dir, f"{split_out}.jsonl")
        save_jsonl(rows, out_path)
        print_stats(rows, split_out, out_path)

    preview(os.path.join(out_dir, "train.jsonl"))

def main():
    print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    print(f"数据保存目录: {DATA_DIR}")

    download_lcqmc()

    print("\n\n全部完成。")


if __name__ == "__main__":
    main()


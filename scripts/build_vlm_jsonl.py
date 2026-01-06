import argparse, json, os
from collections import Counter
import pandas as pd

def norm_hashtags(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip()

def build_target(caption, hashtags) -> str:
    caption = "" if caption is None or (isinstance(caption, float) and pd.isna(caption)) else str(caption).strip()
    hashtags = norm_hashtags(hashtags)

    # 안전장치: "tag1 tag2" 형태면 # 붙이기
    if hashtags and not hashtags.startswith("#"):
        hashtags = " ".join([t if t.startswith("#") else f"#{t}" for t in hashtags.split()])

    # 캡션이 빈 경우도 있을 수 있으니, 태그만 있는 경우도 허용할지 선택 가능
    # 지금은 "둘 다 비면"만 막는다.
    out = caption.strip()
    if hashtags:
        out = (out + "\n\n" + hashtags).strip() if out else hashtags.strip()
    return out.strip()

def is_ok(v) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "ok", "success", "passed", "pass"}

def norm_split(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().lower()
    if "train" in s:
        return "train"
    if "val" in s:
        return "val"
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--img_root", required=True, help="e.g., data/raw/images")

    ap.add_argument("--split_col", default="split")
    ap.add_argument("--ok_col", default="gen_ok")
    ap.add_argument("--caption_col", default="en_caption")      # ✅ 변경
    ap.add_argument("--tag_col", default="en_hashtags")         # ✅ 변경
    ap.add_argument("--filename_col", default="file_name")      # ✅ 변경 (로컬은 file_name 기반)
    ap.add_argument("--id_col", default=None)

    ap.add_argument("--no_ok_filter", action="store_true")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--print_skips", type=int, default=0)

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    # split normalize
    if args.split_col not in df.columns:
        raise ValueError(f"Missing split col: {args.split_col}")
    df[args.split_col] = df[args.split_col].apply(norm_split)

    print("[INFO] rows loaded:", len(df))
    print("[INFO] split counts (normalized):")
    print(df[args.split_col].value_counts(dropna=False))

    # ok filter
    if (not args.no_ok_filter) and (args.ok_col in df.columns):
        before = len(df)
        df = df[df[args.ok_col].apply(is_ok)].copy()
        after = len(df)
        print(f"[INFO] ok filter ON: {before} -> {after}")
    else:
        print("[INFO] ok filter OFF (or ok_col missing)")

    train_df = df[df[args.split_col] == "train"].copy()
    val_df   = df[df[args.split_col] == "val"].copy()
    print(f"[INFO] after filters: train={len(train_df)}, val={len(val_df)}")

    prompt = "Write an Instagram-style caption and hashtags for this image. Use 2-3 emojis and keep it vivid."

    def row_to_item(row, split_name, idx):
        # ✅ 로컬 이미지 경로: img_root/train(or val)/file_name
        if args.filename_col not in row or pd.isna(row[args.filename_col]):
            raise ValueError(f"Missing filename_col={args.filename_col}")
        fn = str(row[args.filename_col]).strip()
        if not fn:
            raise ValueError("Empty filename")

        img_path = os.path.join(args.img_root, split_name, fn)
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        target = build_target(row.get(args.caption_col), row.get(args.tag_col))
        if not target:
            raise ValueError("Empty target (caption/hashtags)")

        ex_id = (
            str(row[args.id_col])
            if (args.id_col and args.id_col in row and pd.notna(row[args.id_col]))
            else f"{split_name}_{idx:06d}"
        )

        return {
            "id": ex_id,
            "image": img_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target},
                    ],
                },
            ],
        }

    def dump_jsonl(sub_df, out_path, split_name):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        bad_reasons = Counter()
        wrote = 0
        printed = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for i, (_, row) in enumerate(sub_df.iterrows()):
                try:
                    item = row_to_item(row, split_name, i)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    wrote += 1
                except Exception as e:
                    bad_reasons[type(e).__name__] += 1
                    if args.print_skips and printed < args.print_skips:
                        print(f"[SKIP] {split_name} idx={i} reason={type(e).__name__}: {e}")
                        printed += 1

        print(f"\n[{split_name}] wrote={wrote}, skipped={len(sub_df)-wrote} -> {out_path}")
        if bad_reasons:
            print(f"[{split_name}] top skip reasons:", bad_reasons.most_common(10))

    dump_jsonl(train_df, args.out_train, "train")
    dump_jsonl(val_df, args.out_val, "val")

if __name__ == "__main__":
    main()

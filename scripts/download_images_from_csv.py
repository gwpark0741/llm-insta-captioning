"""
download_all_images.py

- 생성 결과 CSV를 기반으로 COCO 이미지를 전부 다운로드
- train / val split 자동 분리
- resume 지원 (이미 존재하는 파일은 skip)
- tqdm 진행바 + 실패 로그 저장

실행 예시:
  python3 scripts/download_all_images.py \
    --csv outputs/insta_en_generated_trainval.csv \
    --out_dir data/images \
    --only_gen_ok
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
from tqdm import tqdm


# ======================
# 로그 유틸
# ======================
def log(msg: str):
    print(msg, flush=True)


# ======================
# CSV 로드
# ======================
def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["image_id", "split", "file_name", "image_url"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    return df


# ======================
# gen_ok 안전 변환
# ======================
def parse_gen_ok(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower().strip() in ("true", "1", "yes")
    if isinstance(x, (int, float)):
        return bool(x)
    return False


# ======================
# URL / 저장경로 결정
# ======================
def resolve_url_and_path(row: pd.Series, out_dir: Path) -> Tuple[str, Path]:
    split = str(row["split"]).strip()
    file_name = str(row["file_name"]).strip()
    url = str(row["image_url"]).strip()

    if not url or url.lower() == "nan":
        url = f"http://images.cocodataset.org/{split}/{file_name}"

    save_path = out_dir / split / file_name
    return url, save_path


# ======================
# 단일 이미지 다운로드
# ======================
def download_one(
    url: str,
    save_path: Path,
    timeout: int = 30,
    retries: int = 3,
    sleep_sec: float = 0.8,
) -> Tuple[bool, str]:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # resume
    if save_path.exists() and save_path.stat().st_size > 0:
        return True, "skip"

    tmp_path = save_path.with_suffix(".part")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

            os.replace(tmp_path, save_path)
            return True, "ok"

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

            if attempt < retries:
                time.sleep(sleep_sec)
            else:
                return False, str(e)

    return False, "unknown_error"


# ======================
# 실패 로그
# ======================
def append_fail_log(path: Path, row: pd.Series, url: str, reason: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"{row.get('image_id')}\t{row.get('split')}\t"
            f"{row.get('file_name')}\t{url}\t{reason}\n"
        )


# ======================
# 메인 실행
# ======================
def run(csv_path: Path, out_dir: Path, only_gen_ok: bool):
    df = load_csv(csv_path)

    if only_gen_ok and "gen_ok" in df.columns:
        before = len(df)
        df = df[df["gen_ok"].apply(parse_gen_ok)].copy()
        log(f"Filtered gen_ok: {before} → {len(df)}")

    log(f"Total images to download: {len(df)}")
    log(f"Output dir: {out_dir}")

    fail_log = out_dir.parent / "logs" / "download_failed.tsv"

    ok_cnt = skip_cnt = fail_cnt = 0

    pbar = tqdm(df.iterrows(), total=len(df), desc="Downloading images")

    for _, row in pbar:
        try:
            url, save_path = resolve_url_and_path(row, out_dir)
            ok, status = download_one(url, save_path)

            if ok and status == "skip":
                skip_cnt += 1
            elif ok:
                ok_cnt += 1
            else:
                fail_cnt += 1
                append_fail_log(fail_log, row, url, status)

            pbar.set_postfix(ok=ok_cnt, skip=skip_cnt, fail=fail_cnt)

        except Exception as e:
            fail_cnt += 1
            append_fail_log(fail_log, row, "", f"row_error: {e}")

    log("\n✅ Download finished")
    log(f"ok={ok_cnt}, skip={skip_cnt}, fail={fail_cnt}")
    log(f"fail log: {fail_log}")


# ======================
# Entry point
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="generated CSV path")
    parser.add_argument("--out_dir", default="data/images", help="output image root dir")
    parser.add_argument("--only_gen_ok", action="store_true", help="download only gen_ok rows")
    args = parser.parse_args()

    run(
        csv_path=Path(args.csv),
        out_dir=Path(args.out_dir),
        only_gen_ok=args.only_gen_ok,
    )


if __name__ == "__main__":
    main()

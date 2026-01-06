"""
generate_trainval.py

- Qwen2.5-3B-Instruct로 COCO(train+val 혼합) 원본 5캡션(cap1~cap5)을 입력으로 받아
  Instagram 스타일 영어 caption + hashtags(JSON)를 생성하고,
  결과/실패 로그를 CSV로 저장합니다.

실행:
  python3 scripts/generate_trainval.py
"""

# ======================
# 1) 필요한 패키지 모두 로드
# ======================
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ======================
# 2) 경로/파일명/설정값 정의
# ======================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

COCO_CSV = DATA_DIR / "coco_bottle_bowl_5caps.csv"
FEWSHOT_JSON = DATA_DIR / "insta_caption_5_en_kr.json"

OUT_PATH = OUTPUT_DIR / "insta_en_generated_trainval.csv"
FAIL_PATH = OUTPUT_DIR / "insta_en_failed_trainval.csv"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

BATCH_SIZE = 4
SAVE_EVERY = 100
MAX_NEW_TOKENS = 120

TEMP_MAIN = 0.8
TOP_P_MAIN = 0.9
TEMP_RETRY = 0.55
TOP_P_RETRY = 0.85

DTYPE = torch.float16

TARGET_SPLITS = {"train2017", "val2017"}


# ======================
# 유틸: 로그
# ======================

# [log] 디버깅을 쉽게 하기 위해 출력 메시지를 즉시 flush 하는 로그 함수
def log(msg: str) -> None:
    print(msg, flush=True)


# ======================
# 유틸: 경로 확인/폴더 생성
# ======================

# [ensure_paths] 입력 파일 존재 여부를 확인하고 outputs 폴더를 생성하는 함수
def ensure_paths() -> None:
    """필수 파일 존재 확인 + outputs 폴더 생성."""
    if not COCO_CSV.exists():
        raise FileNotFoundError(f"[Missing] COCO_CSV not found: {COCO_CSV}")
    if not FEWSHOT_JSON.exists():
        raise FileNotFoundError(f"[Missing] FEWSHOT_JSON not found: {FEWSHOT_JSON}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# 데이터 로드
# ======================

# [load_coco_dataframe] COCO CSV를 로드하고, 파이프라인에 필요한 최소 컬럼이 존재하는지 검증하는 함수
def load_coco_dataframe() -> pd.DataFrame:
    """COCO CSV 로드 및 필수 컬럼 검증."""
    df = pd.read_csv(COCO_CSV)

    required_cols = ["image_id", "split", "cap1", "cap2", "cap3", "cap4", "cap5"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[COCO CSV] missing columns: {missing}\n"
            f"-> 현재 컬럼: {df.columns.tolist()}"
        )
    return df


# ======================
# 해시태그 정규화
# ======================

# [normalize_hashtags] 해시태그 내의 공백만 제거하여 '#game night' -> '#gamenight' 형태로 만드는 함수
def normalize_hashtags(tags: str) -> str:
    """
    해시태그 정규화: 공백 제거만
    예) "#game night" -> "#gamenight"
    """
    if not isinstance(tags, str):
        return ""
    s = tags.strip()
    if not s:
        return ""

    parts = s.split("#")
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append("#" + p.replace(" ", ""))
    return " ".join(out)


# [strip_hashtags_from_caption] 캡션에 실수로 섞인 '#...' 토큰을 제거해 캡션 텍스트를 깨끗하게 만드는 함수
def strip_hashtags_from_caption(caption: str) -> str:
    """caption 안에 실수로 들어간 해시태그 제거."""
    if not isinstance(caption, str):
        return ""
    return re.sub(r"#\S+", "", caption).strip()


# ======================
# few-shot 로드
# ======================

# [load_fewshot_examples] few-shot JSON에서 영어 캡션/해시태그만 뽑아 예시 리스트로 만드는 함수
def load_fewshot_examples() -> List[Dict[str, str]]:
    """few-shot JSON에서 en_caption/en_hashtags만 추출."""
    with open(FEWSHOT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"[Fewshot JSON] must be a list. got={type(data)}")

    examples: List[Dict[str, str]] = []
    skipped = 0

    for ex in data:
        if not isinstance(ex, dict):
            skipped += 1
            continue

        en_cap = ex.get("en_caption")
        en_tags = ex.get("en_hashtags")

        if isinstance(en_cap, str) and en_cap.strip() and isinstance(en_tags, str) and en_tags.strip():
            examples.append({
                "caption": en_cap.strip(),
                "hashtags": normalize_hashtags(en_tags),
            })
        else:
            skipped += 1

    if not examples:
        raise ValueError("[Fewshot JSON] no valid examples. check keys: en_caption, en_hashtags")

    log(f"✅ Few-shot loaded: {len(examples)} (skipped={skipped})")
    return examples


# ======================
# 프롬프트 빌더
# ======================

# [build_scene_desc] cap1~cap5를 "정확히 5줄"로 만들어 user prompt에 넣기 위한 입력 문자열을 만드는 함수
def build_scene_desc(row: pd.Series) -> str:
    """cap1~cap5를 5줄로 합쳐 user prompt 입력으로 사용."""
    caps = []
    for k in ["cap1", "cap2", "cap3", "cap4", "cap5"]:
        v = row.get(k, "")
        if isinstance(v, str) and v.strip() and v.lower() != "nan":
            caps.append(v.strip())
        else:
            caps.append("")
    return "\n".join(caps)


# [build_system_prompt] 모델 스타일을 고정하기 위한 system prompt를 구성(금지사항/출력형식/예시 포함)하는 함수
def build_system_prompt(fewshots: List[Dict[str, str]]) -> str:
    """few-shot을 system에 넣어 스타일 고정."""
    shots_txt = ""
    for i, ex in enumerate(fewshots, 1):
        shots_txt += (
            f"Example {i}:\n"
            f'{{"caption": "{ex["caption"]}", "hashtags": "{ex["hashtags"]}"}}\n'
        )

    return f"""You are NOT describing an image.
You are the person who posted this photo on Instagram.

Persona:
You are casually sharing a moment from your own daily life.
You are not explaining what is visible.
You are not analyzing a scene.
You are writing like a real Instagram user who lived this moment.

Writing mindset (very important):
- Write in first-person perspective implicitly (without saying "I" too much).
- Capture how the moment felt, not what was in the photo.
- Think: “Why did I feel like posting this?”
- The caption should feel personal, natural, and unforced.

Core style:
- Warm, cozy, lifestyle-focused.
- Emotions, atmosphere, quiet moments, shared time.
- Everyday feelings that people relate to.
- Use sensory language (light, warmth, calm, comfort, rhythm of the day).

Strong prohibitions (must NOT do):
- Do NOT describe the image like a report or dataset caption.
- Do NOT list objects, people, or actions factually.
- Do NOT summarize the scene.
- Do NOT mention the task, the input captions, or the generation process.
- Do NOT say things like “this image shows”, “based on the descriptions”, or similar.
- Do NOT include hashtags inside the caption text.

Output rules:
- Language: English only.
- Caption: 2–4 sentences.
- Emotional, personal, and Instagram-native.
- Hashtags: 5–7 hashtags, ONE separate line, all lowercase.
- Focus hashtags on lifestyle, mood, daily moments (not object names).

Output format (strict):
- Output exactly ONE JSON object.
- No explanations, no commentary, no extra text.
- Format:
  {{"caption":"...","hashtags":"#... #... #..."}}

Style references (few-shot examples):
{shots_txt}
"""


# [build_user_prompt] 원본 5줄 캡션(scene_desc)을 넣어 모델에게 "JSON만 출력"하도록 요청하는 user prompt를 만드는 함수
def build_user_prompt(scene_desc: str) -> str:
    """원본 5캡션 제공."""
    return f"""Here are 5 captions describing the same scene (exactly 5 lines):
{scene_desc}

Now produce the JSON output.
"""


# ======================
# JSON 파싱(더 견고하게)
# ======================

# [parse_json_from_text] 모델 출력에서 JSON만 최대한 안정적으로 추출해 dict로 변환하는 함수
def parse_json_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    1) 첫 '{'와 마지막 '}'를 기준으로 후보 JSON을 잡는다.
    2) json.loads 시도.
    3) 실패하면 마지막 '}'를 앞당기며 몇 번 더 시도.
    """
    if not isinstance(text, str):
        return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1].strip()

    # 여러 '}'가 있을 수 있어 점진적으로 줄이며 시도
    for _ in range(5):
        try:
            obj = json.loads(candidate)
            cap = str(obj.get("caption", "")).strip()
            tags = str(obj.get("hashtags", "")).strip()
            return {"caption": cap, "hashtags": tags}
        except Exception:
            end2 = candidate.rfind("}", 0, len(candidate) - 1)
            if end2 == -1:
                break
            candidate = candidate[:end2 + 1].strip()

    return None


# [validate_output] 파싱된 결과가 최소 품질 기준(문장/해시태그 개수 등)을 만족하는지 검사하는 함수
def validate_output(obj: Optional[Dict[str, str]]) -> bool:
    """기본 품질 검증."""
    if obj is None:
        return False
    cap = obj.get("caption", "")
    tags = obj.get("hashtags", "")
    if not cap or not tags:
        return False

    tag_list = [t for t in tags.split() if t.startswith("#")]
    if len(tag_list) < 3:
        return False

    sent_cnt = len(re.findall(r"[.!?]", cap))
    if sent_cnt < 1:
        return False

    return True


# ======================
# 모델 로드
# ======================

# [load_model_and_tokenizer] 토크나이저/모델을 로드하고, padding 설정과 pad_token_id를 안전하게 맞추는 함수
def load_model_and_tokenizer() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Qwen2.5 모델/토크나이저 로드 + padding 설정."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # decoder-only 모델에서는 left padding이 일반적으로 안전함
    tokenizer.padding_side = "left"

    # pad_token이 없으면 eos로 대체(패딩 관련 오류/경고 방지)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=DTYPE,
    )
    model.eval()

    # 모델의 pad_token_id가 비어 있으면 토크나이저 값으로 지정
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    log(f"✅ Loaded model: {MODEL_NAME}")
    return tokenizer, model


# ======================
# 배치 생성(샘플별 slice 안전 버전)
# ======================

# [generate_batch] 여러 샘플을 배치로 생성하고, 각 샘플의 실제 입력 길이 기준으로 생성 텍스트만 decode하는 함수
@torch.inference_mode()
def generate_batch(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    scene_desc_list: List[str],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """배치 추론 후, 샘플별 입력 길이 기준으로 decode."""
    prompts = []
    for scene_desc in scene_desc_list:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(scene_desc)},
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 샘플별 실제 입력 길이(=attention_mask 합)를 기준으로 slicing
    attn = inputs["attention_mask"]
    input_lens = attn.sum(dim=1).tolist()

    results = []
    for i in range(outputs.size(0)):
        gen_ids = outputs[i][int(input_lens[i]):]
        results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    return results


# ======================
# 저장 유틸
# ======================

# [append_fail_rows] 실패 샘플 로그를 CSV에 누적 저장하는 함수(중단/재개 시에도 기록 유지)
def append_fail_rows(fail_rows: List[Dict], path: Path) -> None:
    """실패 로그를 CSV에 append 저장."""
    if not fail_rows:
        return
    df_new = pd.DataFrame(fail_rows)
    if path.exists():
        df_prev = pd.read_csv(path)
        pd.concat([df_prev, df_new], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(path, index=False, encoding="utf-8-sig")


# [save_checkpoint] 결과 CSV를 저장하고, 실패 로그도 함께 반영하는 체크포인트 저장 함수
def save_checkpoint(rows_out: List[Dict], fail_buffer: List[Dict]) -> None:
    """결과/실패 로그 저장."""
    pd.DataFrame(rows_out).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    append_fail_rows(fail_buffer, FAIL_PATH)


# [make_unique_key] resume(이어하기) 중 중복 생성을 막기 위해 split+image_id를 유니크 키로 만드는 함수
def make_unique_key(row: pd.Series) -> str:
    """resume 안전성 위해 split+image_id를 키로 사용."""
    return f"{row.get('split','')}_{int(row.get('image_id'))}"


# ======================
# 메인 파이프라인
# ======================

# [run_generation] 전체 파이프라인 실행(데이터 로드 → 프롬프트/모델 준비 → 생성 루프 → 저장/재개)
def run_generation() -> None:
    ensure_paths()

    coco_df = load_coco_dataframe()
    fewshot_examples = load_fewshot_examples()
    system_prompt = build_system_prompt(fewshot_examples[:5])

    tokenizer, model = load_model_and_tokenizer()

    # train+val 모두 대상으로 생성
    gen_df = coco_df[coco_df["split"].isin(TARGET_SPLITS)].copy().reset_index(drop=True)
    log(f"✅ Target splits: {sorted(TARGET_SPLITS)}")
    log(f"✅ Total rows to generate: {len(gen_df)}")

    # resume 준비
    rows_out: List[Dict] = []
    done_keys = set()

    if OUT_PATH.exists():
        prev = pd.read_csv(OUT_PATH)
        rows_out = prev.to_dict("records")
        if "split" in prev.columns and "image_id" in prev.columns:
            done_keys = set(
                (prev["split"].astype(str) + "_" + prev["image_id"].astype(int).astype(str)).tolist()
            )
        log(f"🔁 Resume mode: already generated {len(done_keys)} rows.")

    cnt_total = cnt_success = cnt_retry = cnt_fail = 0
    last_saved = len(rows_out)
    fail_buffer: List[Dict] = []

    pbar = tqdm(range(0, len(gen_df), BATCH_SIZE), desc="Generating train+val")

    for start in pbar:
        batch = gen_df.iloc[start:start + BATCH_SIZE]

        # 이미 생성된 키는 제외(중복 방지)
        batch = batch[~batch.apply(make_unique_key, axis=1).isin(done_keys)]
        if len(batch) == 0:
            continue

        scene_desc_list = [build_scene_desc(r) for _, r in batch.iterrows()]

        # 배치 생성 (OOM 등 런타임 에러 가능)
        try:
            decoded_list = generate_batch(
                tokenizer, model, scene_desc_list, system_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMP_MAIN,
                top_p=TOP_P_MAIN,
            )
        except RuntimeError as e:
            log("\n❌ RuntimeError during generation (likely OOM).")
            log("   -> Reduce BATCH_SIZE or MAX_NEW_TOKENS.")
            log(f"   Error: {e}")
            raise

        # 샘플별 파싱/검증/재시도
        for (_, r), decoded in zip(batch.iterrows(), decoded_list):
            cnt_total += 1
            attempts = 1

            obj = parse_json_from_text(decoded)

            # 1회 재시도(temperature 낮춰 안정화)
            if not validate_output(obj):
                cnt_retry += 1
                attempts = 2
                decoded_retry = generate_batch(
                    tokenizer, model, [build_scene_desc(r)], system_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMP_RETRY,
                    top_p=TOP_P_RETRY,
                )[0]
                obj_retry = parse_json_from_text(decoded_retry)
                if validate_output(obj_retry):
                    obj = obj_retry
                    decoded = decoded_retry
                else:
                    cnt_fail += 1

            ok = validate_output(obj)

            # 원본 row의 모든 컬럼 보존 + 생성 메타정보 추가
            out_row = r.to_dict()
            out_row.update({
                "gen_model": MODEL_NAME,
                "gen_ok": bool(ok),
                "gen_attempts": int(attempts),
                "gen_max_new_tokens": int(MAX_NEW_TOKENS),
                "gen_temperature": float(TEMP_MAIN if attempts == 1 else TEMP_RETRY),
                "gen_top_p": float(TOP_P_MAIN if attempts == 1 else TOP_P_RETRY),
                "raw_output": (decoded or "")[:5000],  # 디버깅용 raw 저장
            })

            if obj:
                cap = strip_hashtags_from_caption(obj.get("caption", ""))
                tags = normalize_hashtags(obj.get("hashtags", ""))
            else:
                cap, tags = "", ""

            out_row["en_caption"] = cap if ok else ""
            out_row["en_hashtags"] = tags if ok else ""

            rows_out.append(out_row)
            done_keys.add(make_unique_key(r))

            if ok:
                cnt_success += 1
            else:
                fail_buffer.append({
                    **r.to_dict(),
                    "gen_model": MODEL_NAME,
                    "raw_output": (decoded or "")[:5000],
                })

            pbar.set_postfix({
                "done": cnt_total,
                "success": cnt_success,
                "retry": cnt_retry,
                "fail": cnt_fail,
                "saved_total": len(rows_out),
            })

        # 주기 저장(중단 대비)
        if (len(rows_out) - last_saved) >= SAVE_EVERY:
            save_checkpoint(rows_out, fail_buffer)
            last_saved = len(rows_out)
            fail_buffer = []
            log(f"\n[Checkpoint] saved_total={len(rows_out)} | done={cnt_total} | "
                f"success={cnt_success} | retry={cnt_retry} | fail={cnt_fail}")

    # 최종 저장
    save_checkpoint(rows_out, fail_buffer)

    log("\n✅ Generation finished")
    log(f"Processed={cnt_total}, Success={cnt_success}, Retry={cnt_retry}, Fail={cnt_fail}")
    log(f"Saved to: {OUT_PATH}")
    log(f"Failed log saved to: {FAIL_PATH}")


# [main] 전체 실행을 try/except로 감싸 에러 메시지를 보기 좋게 출력하는 엔트리 함수
def main() -> None:
    try:
        run_generation()
    except Exception as e:
        log("\n====================")
        log("❌ Pipeline crashed")
        log("====================")
        log(f"Error type: {type(e).__name__}")
        log(f"Message: {e}")
        log("Tip:")
        log(" - data/ 폴더에 coco_bottle_bowl_5caps.csv, insta_caption_5_en_kr.json 존재?")
        log(" - venv 활성화 상태? (.venv)")
        log(" - OOM이면 BATCH_SIZE=2, MAX_NEW_TOKENS=80로 낮추기")
        raise


if __name__ == "__main__":
    main()

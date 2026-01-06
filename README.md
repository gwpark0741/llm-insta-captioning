# LLM Instagram Captioning  
### COCO Captions → Instagram-style English Captions with Qwen2.5

본 프로젝트는 **COCO 데이터셋의 원본 캡션(동일 이미지에 대한 5개 캡션)** 을 입력으로 받아,  
오픈소스 LLM **Qwen2.5-3B-Instruct**를 활용해  
**Instagram 감성의 영어 캡션과 해시태그(JSON 형식)** 를 생성하는 파이프라인을 구현한 프로젝트입니다.

> 🎯 **목적**  
> 학부연구생(Undergraduate Research Assistant) 지원을 준비하며  
> **LLM 추론 파이프라인 설계, 프롬프트 엔지니어링, 데이터 관리, 재현성(resume), 에러 핸들링**을  
> 실제 코드 수준에서 학습·정리하고 이를 포트폴리오로 남기기 위함입니다.

---

## ✨ Key Features

- **Few-shot 기반 스타일 고정**
  - Instagram 감성의 실제 캡션 예시를 system prompt에 포함
  - 매 샘플마다 스타일 편차를 줄이고 일관성 유지

- **System / User Prompt 분리 설계**
  - System: 역할, 금지 규칙, 출력 형식(JSON)
  - User: 입력 데이터(cap1~cap5)
  - → 구조적 안정성 및 디버깅 용이

- **Robust Generation Pipeline**
  - JSON 파싱 + 품질 검증(문장 수, 해시태그 수)
  - 실패 시 temperature/top-p를 낮춰 1회 재시도

- **Train + Validation 혼합 데이터 처리**
  - `split` 컬럼 보존
  - resume 키를 `split + image_id`로 구성해 중복 생성 방지

- **Reproducibility & Debugging**
  - 결과 CSV에 원본 데이터 컬럼 전체 보존
  - 생성 메타정보(`gen_*`) + `raw_output` 저장
  - 실패 샘플은 별도 CSV로 관리

- **Local GPU Inference**
  - WSL + VS Code + NVIDIA GPU 환경
  - venv 기반 패키지 관리

---

## 🧱 Project Structure

```text
llm-insta-captioning/
├─ data/
│  ├─ coco_bottle_bowl_5caps.csv
│  └─ insta_caption_5_en_kr.json
├─ scripts/
│  └─ generate_trainval.py
├─ outputs/
│  ├─ insta_en_generated_trainval.csv
│  └─ insta_en_failed_trainval.csv
├─ .venv/                 # (git ignored) Python virtual environment
└─ README.md
```

---

## 🧠 What I Learned (학습 포인트)

### 1️⃣ Prompt Engineering (System / User 분리)

**System Prompt**
- 모델의 역할 정의
- "이미지 설명 금지", "JSON만 출력" 등 강한 제약
- Few-shot 예시를 system에 포함하여 스타일 고정

**User Prompt**
- 동일 이미지에 대한 5개의 원본 캡션만 제공
- 입력과 규칙을 분리함으로써 프롬프트 구조 단순화

### 2️⃣ Output Validation & Retry Strategy

**LLM 출력에서 자주 발생하는 문제:**
- JSON 외 텍스트가 섞임
- 해시태그 개수 부족
- 문장이 지나치게 짧음

**해결 방법:**
- JSON 파싱 로직을 견고하게 구현
- 품질 기준 미달 시:
  - temperature / top-p 감소
  - 1회 재시도 후에도 실패하면 로그로 분리

### 3️⃣ Data Management & Resume Design

**Train / Validation 데이터가 섞인 CSV에서도:**
- `split` 컬럼을 그대로 유지
- resume 키를 `split + image_id`로 구성

**중간 중단 후 재실행 시:**
- 이미 처리된 샘플은 자동 skip

### 4️⃣ Practical GPU Inference (Local)

- WSL + CUDA 환경에서 직접 추론
- **GPU OOM 대응 전략:**
  - BATCH_SIZE, MAX_NEW_TOKENS 조절
- venv 기반 환경 분리로 시스템 Python 보호

---

## ⚙️ Environment

- **OS:** Windows + WSL (Ubuntu)
- **IDE:** VS Code (Remote - WSL)
- **GPU:** NVIDIA RTX 2070 (8GB)
- **Python:** 3.12 (venv)
- **Frameworks:**
  - PyTorch (CUDA enabled)
  - HuggingFace Transformers

**GPU 확인:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 🛠 Setup

### 1️⃣ Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

### 2️⃣ Install Dependencies

```bash
pip install torch transformers pandas tqdm
```

### 3️⃣ Prepare Data

`data/` 폴더에 다음 파일 배치:
- `coco_bottle_bowl_5caps.csv`
- `insta_caption_5_en_kr.json`

---

## ▶️ Run

```bash
python3 scripts/generate_trainval.py
```

---

## 📦 Output Files

### `insta_en_generated_trainval.csv`

**원본 컬럼 유지**
- `split`, `image_id`, `file_name`, `cap1~cap5`, 기타 이미지 메타데이터

**생성 결과**
- `en_caption`
- `en_hashtags`

**생성 메타정보**
- `gen_ok`
- `gen_attempts`
- `gen_temperature`
- `gen_top_p`
- `gen_max_new_tokens`

**디버깅**
- `raw_output` (모델 원문 일부)

### `insta_en_failed_trainval.csv`

- 생성 실패 샘플만 별도 저장
- 추후 재생성 / 오류 분석 용도

---

## 🧩 Design Choices

### 왜 `raw_output`을 저장했는가?

- JSON 파싱 실패 원인을 즉시 확인하기 위함
- 모델이 코드블록, 설명 텍스트를 섞는 패턴 분석 가능

### 왜 원본 컬럼을 전부 보존했는가?

- 이미지 파일 매칭
- downstream 학습(train/val 분리)
- 추가 분석을 쉽게 하기 위함

---

## 🚀 Next Steps

- [ ] 실패 샘플만 재생성하는 `retry_failed.py`
- [ ] 생성 결과를 활용한 SFT(지도 파인튜닝) 데이터셋 구축
- [ ] 캡션/해시태그 품질 평가 지표 설계
- [ ] 2B 이하 모델 + 4bit 양자화로 추론 효율 개선

---

## 🙋‍♂️ About This Project

본 프로젝트는  
**LLM 기반 생성 파이프라인을 처음부터 끝까지 직접 설계·구현**하고,  
**재현성과 디버깅을 고려한 실전 코드**로 정리하는 것을 목표로 했습니다.

**학부연구생 지원을 준비하며**
- LLM 구조 이해
- 프롬프트 설계
- 데이터 처리 및 실험 관리

를 **실제 코드로 증명하기 위한 학습 기록**입니다.

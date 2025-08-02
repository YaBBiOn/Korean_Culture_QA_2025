# 국립국어원 주관 한국어 문화 질의응답 경진대회 (가 유형) 결과물
- 본 프로젝트는 다음 베이스라인 코드를 참고하여 개발되었습니다.  
- 출처: https://github.com/teddysum/Korean_Culture_QA_2025

## 파일 구조

```
run/
├─ before_submit.py   # 제주 전 후처리 스크립트
├─ qlora.py           # QLoRA 튜닉 스크립트
├─ test.py            # 8bit AWQ 양자화 추론 스크립트
└─ test_lora.py       # QLoRA 튜닉된 모델 추론 스크립트

src/
└─ data.py            # 프론프트 및 데이터셋 클래스 정의
```

## 대회 제약 사항

* 단일 GPU (NVIDIA RTX 4090)만 사용 가능
* 외부 데이터 증가 금지 (가 유형)
* 상업적 사용 시 저작권 문제가 없어야 함

## 작동 환경

* 로커룸 RTX 3070 (8GB VRAM)

## 사용 모델

### 최종 사용 모델

* 모델명: `K-intelligence/Midm-2.0-Base-Instruct`
* 양자화: 8bit AWQ
* 라이센스 정보:

```text
Model: K-intelligence/Midm-2.0-Base-Instruct
License: MIT License
Source: https://huggingface.co/K-intelligence/Midm-2.0-Base-Instruct
```

### 비교 모델

* 모델명: `SKT/A.X-4.0-Light`
* 양자화: 4bit QLoRA
* 라이센스 정보:

```text
Model: SKT/A.X-4.0-Light
License: Apache 2.0
Source: https://huggingface.co/skt/A.X-4.0-Light
```

## 작동 방법

### 공통 준비 사항

* 가상환경을 설치합니다. (Python 3.9 이상 권장)
* CUDA 건설이 가능하지 않은 경우, 해당 환경에 맞게 CUDA를 설치해야 합니다.
* 다음 명령을 입력해 특정 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 8bit AWQ 추론을 진행할 경우

* 사용 파일: `test.py`, `before_submit.py`
* 이전에 QLoRA를 진행했다면, `/src/data.py` 의 128\~137번줄 각주를 다음과 같이 변경합니다.

```python
# QLoRA를 사용하는 경우
# def __getitem__(self, idx):
#     return {
#         "input_ids": self.inp[idx],
#         "labels": self.label[idx]
#     }

# QLoRA를 사용하지 않는 경우
def __getitem__(self, idx):
    return self.inp[idx]
```

* `/run` 디렉토리로 이동한 후 다음 명령을 입력합니다.

```bash
python test.py \
  --input input_file.json \
  --output output_file.json \
  --model_id huggingface_model_name \
  --tokenizer huggingface_model_name \
  --device cuda \
  --use_auth_token hf_xxxxxxxx (optional)
```

* `before_submit.py`의 32\~33번줄을 다음과 같이 변경합니다.

```python
input_path = "B_prompt3_raw_test_submit.json"
output_path = "B_prompt3_raw_cleaned_submit.json"
```

* 다음 명령으로 후처리를 진행합니다.

```bash
python before_submit.py
```

### 4bit QLoRA 튜닉을 진행할 경우

* 사용 파일: `qlora.py`, `test_lora.py`, `before_submit.py`
* `/src` 파일로 이동한 후 `qlora.py` 내 `main()` 함수 중 단변을 변경합니다.

```python
model_id = "skt/A.X-4.0-Light"
train_file = os.path.join(os.path.dirname(__file__), "temp_train.json")
output_dir = os.path.join(os.path.dirname(__file__), "qlora_output")
```

* `/run`로 이동해 다음 명령을 입력합니다.

```bash
python qlora.py
```

* `/src/data.py`의 128\~137번줄을 다음과 같이 변경합니다.

```python
# QLoRA를 사용할 경우
def __getitem__(self, idx):
    return {
        "input_ids": self.inp[idx],
        "labels": self.label[idx]
    }

# 추론만 진행할 경우
# def __getitem__(self, idx):
#     return self.inp[idx]
```

* 다음 명령을 입력해 튜닉된 모델을 통해 추론를 진행합니다.

```bash
python test_lora.py \
  --input input_file.json \
  --output output_file.json \
  --model_id skt/A.X-4.0-Light \
  --adapter ./qlora_output \
  --device cuda
```

* `before_submit.py`의 인터페이스를 다음과 같이 변경합니다.

```python
input_path = "B_prompt3_raw_test_submit.json"
output_path = "B_prompt3_raw_cleaned_submit.json"
```

```bash
python before_submit.py
```

## 추론 데이터 예시

```json
[
  {
    "id": "6",
    "input": {
      "category": "문화 지식",
      "domain": "풍식/문화유산",
      "question_type": "단단형",
      "topic_keyword": "금줄",
      "question": "한국의 금줄은 무엇을 막기 위해 문이나 길 어귀에 건너집마 메나요?"
    },
    "output": {
      "answer": "부정"
    }
  },
  {
    "id": "2",
    "input": {
      "category": "문화 지식",
      "domain": "풍식/문화유산",
      "question_type": "선다형",
      "topic_keyword": "진도아리랑",
      "question": "한국의 진도아리랑은 어느 타령에 영향을 받아 만들어져나요? \n 1\t각설이타령   2\t도라지타령    3\t용강타령    4\t 산아지타령"
    },
    "output": {
      "answer": "4"
    }
  },
  {
    "id": "3",
    "input": {
      "category": "문화 지식",
      "domain": "사회",
      "question_type": "서술형",
      "topic_keyword": "행정구역",
      "question": "대한민국의 행정구역 체계를 서술하세요."
    },
    "output": {
      "answer": "대한민국의 행정구역은 ..."
    }
  }
]
```

> 위 예시는 대회에서 제공한 train 데이터 중 일부를 등록한 것으로, 결과 형식을 설명하기 위해 발체했습니다. 실제 제켜할 평가 데이터는 제공되지 않으며, 각 예시는 설명 목적으로만 사용됩니다.

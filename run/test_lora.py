import argparse
import json
import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import CustomDataset

import time

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--model_id", type=str, required=True)
parser.add_argument("--adapter", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
adapter_path = os.path.abspath(args.adapter)


def main():
    t1 = time.time()

    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # qlora.py를 통해 튜닝된 모델 불러오기
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map=args.device
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    # 토크나이저 설정
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") or tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    # 데이터셋 준비
    dataset = CustomDataset(args.input, tokenizer)

    with open(args.input, "r") as f:
        result = json.load(f)

    # 모델 추론
    print("=== LoRA 튜닝 모델로 문제 추론을 시작합니다 ===")
    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]

        with torch.no_grad():
            output_ids = model.generate(
                inp.to(args.device).unsqueeze(0),
                max_new_tokens=1024,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
                temperature=0.7,
                top_p=0.8
            )

        output_text = tokenizer.decode(output_ids[0][inp.shape[-1]:], skip_special_tokens=True)
        # 출력에서 "답변: " 접두어 제거
        if output_text.startswith("답변: "):
            output_text = output_text[4:]
        elif output_text.startswith("답변:"):
            output_text = output_text[3:]

        result[idx]["output"] = {"answer": output_text}

    # 결과물 저장
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

    print("=== 저장이 완료되었습니다 ===")
    t2 = time.time()
    print(f'=== 총 {t2-t1}초 소모했습니다 ===')



if __name__ == "__main__":
    main()
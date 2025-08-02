import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # 마지막꺼는 양자화용

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import CustomDataset

import time



# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")


g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, required=True, help="input filename")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
# fmt: on


def main(args):
    # 코드 추가 : 8b 양자화 
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": args.device
    }
    
    t1 = time.time()

    # 토크나이저 설정
    if args.use_auth_token:
        model_kwargs["use_auth_token"] = args.use_auth_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_kwargs
    )
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    
    tokenizer_kwargs = {}
    if args.use_auth_token:
        tokenizer_kwargs["use_auth_token"] = args.use_auth_token
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    file_test = args.input

    # 데이터셋 설정
    dataset = CustomDataset(file_test, tokenizer)

    with open(file_test, "r") as f:
        result = json.load(f)

    # 모델 추론
    print('=== 문제 추론을 진행합니다 ===')
    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]

        with torch.no_grad():
            outputs = model.generate(
                inp.to(args.device).unsqueeze(0),
                max_new_tokens=1024, 
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
                temperature=0.7, 
                top_p=0.8,
                # do_sample=False,
            )

        output_text = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
        # 출력에서 "답변: " 접두어 제거
        if output_text.startswith("답변: "):
            output_text = output_text[4:]
        elif output_text.startswith("답변:"):
            output_text = output_text[3:]
            
        result[idx]["output"] = {"answer": output_text}

    print('=== 모든 문제 정리가 끝났습니다. 해당 내용을 저장하겠습니다. ===')
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
    print('=== 저장이 완료되었습니다 ===')

    t2 = time.time()
    print(f'추론하는데 {t2-t1:.5f} sec가 소모되었습니다.')
    print(f'추론하는데 약 {round((t2-t1)/3600)}시간 정도 소요되었습니다.')

if __name__ == "__main__":
    
    exit(main(parser.parse_args()))
    
    


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/data.py 인식
from src.data import CustomDataset, DataCollatorForSupervisedDataset


def load_model(model_id: str) -> torch.nn.Module:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    return model


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_dataset(train_file: str, tokenizer) -> tuple:
    dataset = CustomDataset(train_file, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    return dataset, data_collator


def create_training_args(output_dir: str) -> TrainingArguments:
    bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere 이상
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=bf16_supported,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        evaluation_strategy="no",
        report_to="none",
        save_strategy="steps"
    )


def main():
    t1 = time.time()

    model_id = "skt/A.X-4.0-Light" # 이곳에 사용할 모델명을 입력해주세요
    train_file = os.path.join(os.path.dirname(__file__), "temp_train.json") # 이곳에 추론할 데이터명을 입력해주세요
    output_dir = os.path.join(os.path.dirname(__file__), "qlora_output") # 이곳에 qlora로 튜닝된 모델명 (=아웃풋)을 입력하세요

    print("모델을 로드합니다.")
    model = load_model(model_id)

    print("토크나이저를 로드합니다.")
    tokenizer = load_tokenizer(model_id)

    print("데이터셋을 준비합니다.")
    dataset, data_collator = load_dataset(train_file, tokenizer)

    print("학습에 필요한 설정을 진행합니다.")
    training_args = create_training_args(output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(" 학습 시작!")
    trainer.train()

    t2 = time.time()
    print("학습이 종료되었습니다.")
    print(f"학습 시간은 {t2 - t1:.2f}초 입니다.")


if __name__ == "__main__":
    main()

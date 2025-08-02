import json

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = """You are a helpful AI assistant.
                당신은 한국의 전통 문화, 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 해박하고 유능한 AI 어시스턴트입니다.
                답변은 최대한 정확하고 일관되게 작성하십시오.
                사용자의 질문에 대해 친절하게 답변하되, 반드시 질문 유형에 따른 답변 형식을 지키십시오.
                또한 문제의 문장을 그대로 복사하거나 재인용하지 마시오.
                """

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            # 문제 유형별 영어 지침 + 한국어 예시
            type_instructions = {
                "선다형": (
                    "[Instructions]\n"
                    "- Read and understand the question carefully.\n"
                    "- Respond **only** with the number of the correct choice and nothing else.\n"
                    "- Do not include any explanations or extra information.\n"
                    "- Do not copy or quote the question text.\n\n"
                    "[Example]\n"
                    "질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?\n"
                    "1) 주사위 놀이\n"
                    "2) 검무\n"
                    "3) 격구\n"
                    "4) 영고\n"
                    "5) 무애무\n"
                    "답변: 3"
                ),
                "서술형": (
                    "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
                    "[예시]\n"
                    "질문: 대한민국의 행정구역 체계를 서술하세요.\n"
                    "답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다."
                ),
                "단답형": (
                    "[Instructions]\n"
                    "- Respond in **no more than two words**.\n"
                    "- Do not include any explanations or extra information.\n"
                    "- Do not copy or quote the question text.\n\n"
                    "[Example]\n"
                    "질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?\n"
                    "답변: 정약용"
                )
            }

            # 문제 유형에 따른 지침 가져오기
            instruction = type_instructions.get(inp['question_type'], "")

            # 문제 맥락 정보 (한국어)
            context_info = (
                "[문제 맥락]\n"
                f"- 카테고리: {inp.get('category', '')}\n"
                f"- 도메인: {inp.get('domain', '')}\n"
                f"- 키워드: {inp.get('topic_keyword', '')}\n"
            )

            # 최종 프롬프트 조합
            chat = "\n\n".join([
                instruction,
                context_info,
                "[질문]",
                inp["question"]
            ])

            return chat

        for example in data:
            user_prompt = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if False: print(f'[DBG] message: {message}')
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )

            output = example.get("output", None)


            # 추가 내용 1 : 데이터셋의 상황에 따라 분기 처리
            if output is None or output == "":
                # test set이라 정답이 없음 => 빈 target
                target_text = ""
            else:
                # valid set이라 정답 dict
                target_text = output["answer"]

            if target_text != "":
                target_text += tokenizer.eos_token

            target = tokenizer(
                target_text,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )

            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)


    # 추가 코드 2 : qlora로 튜닝중일 때에는 이 코드를 사용
    # def __getitem__(self, idx):
    #     return {
    #         "input_ids": self.inp[idx],
    #         "labels": self.label[idx]
    #     }

    # 비교 코드 : qlora를 안 쓰고 추론만 진행할 때엔 이 코드 사용
    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

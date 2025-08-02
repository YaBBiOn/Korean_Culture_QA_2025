import json
import re

# 한글 초성 집합
chosung_set = set('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')

# 전처리 함수
def preprocess_answers(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        answer = item["output"]["answer"]

        # 1. 맨 앞 공백 제거
        answer = answer.lstrip()

        # 2. 단답형 + 초성 포함 + '-'로 적절히 구분 안 된 경우 처리
        if item["input"].get("question_type") == "단답형":
            has_chosung = any(char in chosung_set for char in answer)
            if has_chosung and not re.fullmatch(r"(?:[ㄱ-ㅎ]-?)+", answer):
                # 초성 문자만 골라서 '-'로 연결
                answer = '-'.join([char for char in answer if char in chosung_set])

        item["output"]["answer"] = answer

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 사용 예시
input_path = "B_prompt3_raw_test_submit.json"
output_path = "B_prompt3_raw_cleaned_submit.json"
preprocess_answers(input_path, output_path)
print('후처리 성공')
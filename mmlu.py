import re


def parse_mmlu_response(mmlu_example, model_output) :
    match = re.search(r"(?i:correct answer is)\s*([^.]+)", model_output)
    if not match:
        return None
    answer = match.group(1).strip()
    if re.fullmatch(r"[A-D]|\(\s*[A-D]\s*\)", answer):
        return answer.strip("() \t\n")
    return None

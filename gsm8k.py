import re


def parse_gsm8k_response(model_output) :
    # last number in the output ("She sold 15 clips." -> "15"); commas dropped, "72." -> "72"
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", model_output)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")

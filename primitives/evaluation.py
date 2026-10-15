from difflib import SequenceMatcher
import re


def parse_longhealth_answer(
    text: str,
    options: list[str],
) -> str | None:
    # <answer> 태그 안의 답과 가장 비슷한 선택지를 답으로 봄.
    # 태그를 닫지 못하면 답이 없는 것으로 봄.
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match is None:
        return None
    options = [option.strip().lower() for option in options]
    prediction = match.group(1).strip().lower()
    return max(options, key=lambda option: SequenceMatcher(None, prediction, option).ratio())

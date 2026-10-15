from dataclasses import dataclass
import math
import time

from simple_parsing import parse
from transformers import set_seed

from primitives.dataset import load_longhealth_corpus, load_longhealth_questions
from primitives.evaluation import parse_longhealth_answer
from primitives.hub import get_pending_targets, load_tensors, save_result
from primitives.model import generate, generate_vllm, load_model, load_vllm
from primitives.prompts import LLAMA_USER_TURN
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"
    batch_size: int = 32
    max_generation_len: int = 1024
    temperature: float = 0.3
    seed: int = 0


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    targets = get_pending_targets("longhealth", config.model_name, file_name)
    samples = load_longhealth_questions()

    # icl은 기록 전체가 프롬프트에 들어가 길어서 vllm으로 따로 돌림.
    if f"icl/{config.model_name}" in targets:
        corpus = load_longhealth_corpus()
        model, _ = load_vllm(config.model_name, max_sequence_len=131072)
        chats = [(corpus, question) for question, _, _ in samples]
        outputs = generate_vllm(model, chats, config.max_generation_len, config.temperature, seed=config.seed)
        texts = [output.text for output in outputs]
        corrects = [parse_longhealth_answer(text, options) == answer for text, (_, options, answer) in zip(texts, samples)]
        accuracy = sum(corrects) / len(samples)
        print(f"icl/{config.model_name} accuracy {accuracy:.2%}")
        save_result(f"longhealth/icl/{config.model_name}", file_name, {"accuracy": accuracy, "corrects": corrects, "texts": texts})
        model.llm_engine.engine_core.shutdown()

    model, tokenizer = load_model(config.model_name)
    turns = [LLAMA_USER_TURN.format(content=question) for question, _, _ in samples]
    question_ids = [tokenizer(turn, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for turn in turns]
    num_batches = math.ceil(len(samples) / config.batch_size)
    for target in targets:
        method = target.split("/")[0]
        if method == "icl":
            continue
        sink = None
        cartridge = None
        if method == "cartridge":
            tensors = load_tensors(target, "cartridge")
            sink = tensors["sink"].to(model.device)
            cartridge = tensors["cartridge"].to(model.device)

        set_seed(config.seed)
        texts = []
        corrects = []
        log_time = time.perf_counter()
        for step, start in enumerate(range(0, len(samples), config.batch_size)):
            batch = samples[start : start + config.batch_size]
            batch_question_ids = question_ids[start : start + config.batch_size]
            output_ids = generate(model, tokenizer, batch_question_ids, config.max_generation_len, config.temperature, sink=sink, cartridge=cartridge)
            batch_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            texts.extend(batch_texts)
            corrects.extend(parse_longhealth_answer(text, options) == answer for text, (_, options, answer) in zip(batch_texts, batch))
            print(f"batch {step + 1}/{num_batches} accuracy {sum(corrects) / len(corrects):.2%} time {format_duration(time.perf_counter() - log_time)}")
            log_time = time.perf_counter()

        accuracy = sum(corrects) / len(samples)
        print(f"{target} accuracy {accuracy:.2%}")
        save_result(f"longhealth/{target}", file_name, {"accuracy": accuracy, "corrects": corrects, "texts": texts})


if __name__ == "__main__":
    main()

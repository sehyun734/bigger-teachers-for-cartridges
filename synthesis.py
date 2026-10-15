from collections import defaultdict
from dataclasses import dataclass
import math
import random
import time

from simple_parsing import parse
import torch

from primitives.dataset import load_longhealth_contexts
from primitives.hub import make_synthesis_path, save_synthesis
from primitives.model import generate_vllm, load_vllm
from primitives.prompts import COT_PROMPTS, FORMATS, SEED_PROMPTS, SYSTEM_PROMPT
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"
    num_samples: int = 16384
    batch_size: int = 1024
    max_question_len: int = 512
    question_temperature: float = 0.6
    max_answer_len: int = 1024
    answer_temperature: float = 0.0
    cot_ratio: float = 0.75
    topk: int = 20


def main() -> None:
    config = parse(Config)
    contexts = load_longhealth_contexts()
    model, tokenizer = load_vllm(config.model_name)

    prompts = []
    for _ in range(config.num_samples):
        context = random.choice(contexts)
        seed_prompt = random.choice(random.choice(SEED_PROMPTS)).format(format=random.choice(FORMATS))
        cot_prompt = random.choice(COT_PROMPTS) if random.random() < config.cot_ratio else None
        prompts.append((context, seed_prompt, cot_prompt))

    rows = []
    tensors = defaultdict(list)
    num_batches = math.ceil(len(prompts) / config.batch_size)
    log_time = time.perf_counter()
    for step, start in enumerate(range(0, len(prompts), config.batch_size)):
        batch = prompts[start : start + config.batch_size]

        chats = [(SYSTEM_PROMPT.format(context=context), seed_prompt) for context, seed_prompt, _ in batch]
        question_outputs = generate_vllm(model, chats, config.max_question_len, config.question_temperature)
        questions = [output.text for output in question_outputs]

        contents = [f"{question}\n\n{cot_prompt}" if cot_prompt else question for (_, _, cot_prompt), question in zip(batch, questions)]
        chats = [(SYSTEM_PROMPT.format(context=context), content) for (context, _, _), content in zip(batch, contents)]
        answer_outputs = generate_vllm(model, chats, config.max_answer_len, config.answer_temperature, topk=config.topk)
        answers = [tokenizer.decode(output.token_ids) for output in answer_outputs]

        rows.extend({"context": context, "question": question, "cot": cot_prompt, "answer": answer} for (context, _, cot_prompt), question, answer in zip(batch, questions, answers))
        tensors["answer_ids"].append(torch.cat([torch.tensor(output.token_ids) for output in answer_outputs]))
        # vllm은 샘플링한 토큰의 logprob을 top-k 앞에 붙이므로 빼고 저장.
        tensors["topk_ids"].append(torch.cat([torch.tensor(output.logprobs.token_ids).view(-1, config.topk + 1)[:, 1:] for output in answer_outputs]))
        tensors["topk_logprobs"].append(torch.cat([torch.tensor(output.logprobs.logprobs).view(-1, config.topk + 1)[:, 1:] for output in answer_outputs]))
        tensors["answer_lens"].append(torch.tensor([len(output.token_ids) for output in answer_outputs]))

        print(f"batch {step + 1}/{num_batches} time {format_duration(time.perf_counter() - log_time)}")
        log_time = time.perf_counter()

    tensors = {name: torch.cat(values) for name, values in tensors.items()}
    save_synthesis(make_synthesis_path(config.model_name, config.num_samples, config.cot_ratio), rows, tensors)


if __name__ == "__main__":
    main()

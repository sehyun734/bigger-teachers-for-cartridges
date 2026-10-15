from dataclasses import dataclass
import math
import random
import time

from simple_parsing import parse
import torch
from torch.nn import Parameter
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import DynamicCache, set_seed

from primitives.dataset import load_synthesis, load_wikitext
from primitives.hub import make_cartridge_path, make_synthesis_path, save_tensors
from primitives.model import load_model, make_cartridge_cache
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"
    teacher_name: str = "Llama-3.1-8B-Instruct"
    cartridge_len: int = 2048
    num_samples: int = 16384
    cot_ratio: float = 0.75
    num_epochs: int = 2
    batch_size: int = 64
    micro_batch_size: int = 4
    learning_rate: float = 2e-2
    seed: int = 0


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    model, tokenizer = load_model(config.model_name)
    samples = load_synthesis(make_synthesis_path(config.teacher_name, config.num_samples, config.cot_ratio), tokenizer)

    # cartridge는 문서와 상관없는 텍스트의 kv cache로 초기화함.
    # 맨 앞 bos는 sink로 고정하고 나머지만 학습.
    cartridge_ids = load_wikitext(tokenizer, config.cartridge_len)
    cache = DynamicCache()
    model.model(input_ids=cartridge_ids.unsqueeze(0).to(model.device), past_key_values=cache)
    key_value = torch.stack([torch.stack([layer.keys, layer.values]) for layer in cache.layers])
    sink = key_value[..., :1, :]
    cartridge = Parameter(key_value[..., 1:, :].float())
    optimizer = AdamW([cartridge], lr=config.learning_rate, weight_decay=0)

    num_batches = math.ceil(len(samples) / config.batch_size)
    losses = []
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config.num_epochs):
        random.shuffle(samples)
        for step, start in enumerate(range(0, len(samples), config.batch_size)):
            batch = samples[start : start + config.batch_size]
            num_answer_tokens = sum(len(answer_ids) for _, answer_ids, _, _ in batch)
            batch_loss = 0.0

            for micro_start in range(0, len(batch), config.micro_batch_size):
                micro_batch = batch[micro_start : micro_start + config.micro_batch_size]
                input_ids = [torch.cat([question_ids, answer_ids]) for question_ids, answer_ids, _, _ in micro_batch]
                input_ids = pad_sequence(input_ids, batch_first=True).to(model.device)
                cache = make_cartridge_cache(sink, cartridge, len(micro_batch))
                hidden = model.model(input_ids=input_ids, past_key_values=cache).last_hidden_state
                answer_hiddens = [hidden[index, len(question_ids) - 1 : len(question_ids) + len(answer_ids) - 1] for index, (question_ids, answer_ids, _, _) in enumerate(micro_batch)]
                logprobs = log_softmax(model.lm_head(torch.cat(answer_hiddens)).float(), dim=-1)
                topk_ids = torch.cat([sample_topk_ids for _, _, sample_topk_ids, _ in micro_batch]).to(model.device)
                topk_logprobs = torch.cat([sample_topk_logprobs for _, _, _, sample_topk_logprobs in micro_batch]).to(model.device)
                # 답 토큰마다 teacher top-k 분포와의 cross entropy.
                loss = -(topk_logprobs.exp() * logprobs.gather(-1, topk_ids)).sum() / num_answer_tokens
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            losses.append(batch_loss)
            if (step + 1) % 16 == 0:
                print(f"epoch {epoch + 1}/{config.num_epochs} batch {step + 1}/{num_batches} loss {sum(losses) / len(losses):.4f} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                losses = []
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

    path = make_cartridge_path(config.model_name, config.teacher_name, config.num_samples, config.cot_ratio, config.cartridge_len, config.learning_rate, config.batch_size, config.num_epochs, config.seed)
    save_tensors(path, "cartridge", {"sink": sink, "cartridge": cartridge})


if __name__ == "__main__":
    main()

import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from primitives.prompts import DATE_STRING


def load_model(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # llama tokenizer는 pad 토큰이 없어 generate에서 오류가 남.
    # 따라서 eos를 pad로 씀.
    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}", dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model, tokenizer


def load_vllm(
    model_name: str,
    max_sequence_len: int = 16384,
) -> tuple[LLM, TokenizerLike]:
    model = LLM(f"meta-llama/{model_name}", dtype="bfloat16", max_model_len=max_sequence_len)
    return model, model.get_tokenizer()


def make_cartridge_cache(
    sink: Tensor,
    cartridge: Tensor,
    batch_size: int,
) -> DynamicCache:
    key_value = torch.cat([sink, cartridge.to(sink.dtype)], dim=-2)
    return DynamicCache(ddp_cache_data=key_value.repeat_interleave(batch_size, dim=2))


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_ids: list[Tensor],
    max_generation_len: int,
    temperature: float,
    sink: Tensor | None = None,
    cartridge: Tensor | None = None,
) -> Tensor:
    cache = make_cartridge_cache(sink, cartridge, len(question_ids)) if cartridge is not None else None
    cache_len = cache.get_seq_length() if cache is not None else 0
    # cartridge는 맨 앞에 bos를 sink로 가지고 있으므로 cartridge가 없을 때만 bos를 붙임.
    bos = [torch.tensor([tokenizer.bos_token_id])] if cartridge is None else []
    question_ids = [torch.cat([*bos, sample_question_ids]) for sample_question_ids in question_ids]
    input_ids = pad_sequence(question_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    attention_mask = pad_sequence([torch.ones_like(sample_question_ids) for sample_question_ids in question_ids], batch_first=True, padding_side="left")
    # generate는 input_ids 앞에서 cache 길이만큼을 이미 처리한 토큰으로 보고 건너뜀.
    # 따라서 cartridge 길이만큼 앞을 채우고, attention_mask는 cartridge를 보도록 1로 채움.
    input_ids = pad(input_ids, (cache_len, 0), value=tokenizer.pad_token_id)
    attention_mask = pad(attention_mask, (cache_len, 0), value=1)

    output_ids = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        past_key_values=cache,
        max_new_tokens=max_generation_len,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id,
    )
    return output_ids[:, input_ids.shape[1] :]


def generate_vllm(
    model: LLM,
    chats: list[tuple[str, str]],
    max_generation_len: int,
    temperature: float,
    topk: int | None = None,
    seed: int | None = None,
) -> list[CompletionOutput]:
    messages = [[{"role": "system", "content": system}, {"role": "user", "content": user}] for system, user in chats]
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_generation_len, logprobs=topk, flat_logprobs=True, seed=seed)
    outputs = model.chat(messages, sampling_params, use_tqdm=False, chat_template_kwargs={"date_string": DATE_STRING})
    return [output.outputs[0] for output in outputs]

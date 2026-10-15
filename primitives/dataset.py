import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import requests
from safetensors.torch import load_file
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from primitives.hub import get_repo_name
from primitives.prompts import CORPUS, DATE_STRING, LLAMA_USER_TURN, LONGHEALTH_NO_COT_QUESTION, LONGHEALTH_QUESTION, NOTE, PATIENT, RECORD

LONGHEALTH_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"


def load_longhealth_patients(
    num_patients: int,
) -> list[tuple[str, dict]]:
    return sorted(requests.get(LONGHEALTH_URL).json().items())[:num_patients]


def load_longhealth_corpus(
    num_patients: int = 10,
) -> str:
    patients = load_longhealth_patients(num_patients)
    records = []
    for patient_id, patient in patients:
        notes = [NOTE.format(note_id=note_id, note=note) for note_id, note in patient["texts"].items()]
        records.append(PATIENT.format(patient_id=patient_id, num_notes=len(notes), notes="\n".join(notes), **patient))
    return CORPUS.format(records="\n\n".join(records))


def load_longhealth_contexts(
    num_patients: int = 10,
) -> list[str]:
    # 환자 정보와 노트 하나씩을 묶어 context로 씀.
    patients = load_longhealth_patients(num_patients)
    contexts = []
    for patient_id, patient in patients:
        notes = [NOTE.format(note_id=note_id, note=note) for note_id, note in patient["texts"].items()]
        contexts.extend(RECORD.format(patient_id=patient_id, num_notes=len(notes), notes=note, **patient) for note in notes)
    return contexts


def load_longhealth_questions(
    num_patients: int = 10,
    use_cot: bool = True,
) -> list[tuple[str, list[str], str]]:
    patients = load_longhealth_patients(num_patients)
    template = LONGHEALTH_QUESTION if use_cot else LONGHEALTH_NO_COT_QUESTION
    samples = []
    for patient_id, patient in patients:
        for row in patient["questions"]:
            options = [row[f"answer_{option}"] for option in "abcde"]
            question = template.format(patient_id=patient_id, question=row["question"], options="\n".join(options), **patient)
            samples.append((question, options, row["correct"].strip().lower()))
    return samples


def load_wikitext(
    tokenizer: PreTrainedTokenizerBase,
    cartridge_len: int,
) -> Tensor:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = ""
    for row in dataset:
        text += row["text"]
        if len(tokenizer(text)["input_ids"]) > cartridge_len:
            break

    # cartridge_len개로 자르되 system turn을 닫는 마지막 토큰은 남김.
    input_ids = tokenizer.apply_chat_template([{"role": "system", "content": text}], date_string=DATE_STRING, return_tensors="pt")["input_ids"][0]
    return torch.cat([input_ids[:cartridge_len], input_ids[-1:]])


def load_synthesis(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    repo_name = get_repo_name()
    rows = json.loads(Path(hf_hub_download(repo_name, f"{path}/rows.json")).read_text())
    tensors = load_file(hf_hub_download(repo_name, f"{path}/logprobs.safetensors"))
    answer_lens = tensors["answer_lens"].tolist()
    answer_ids = tensors["answer_ids"].split(answer_lens)
    topk_ids = tensors["topk_ids"].split(answer_lens)
    topk_logprobs = tensors["topk_logprobs"].split(answer_lens)

    contents = [f"{row['question']}\n\n{row['cot']}" if row["cot"] else row["question"] for row in rows]
    turns = [LLAMA_USER_TURN.format(content=content.strip()) for content in contents]
    question_ids = [tokenizer(turn, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for turn in turns]
    return list(zip(question_ids, answer_ids, topk_ids, topk_logprobs))

import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from safetensors.torch import load_file, save
from torch import Tensor


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/bigger-teachers-for-cartridges"


def make_synthesis_path(
    teacher_name: str,
    num_samples: int,
    cot_ratio: float,
) -> str:
    return f"synthesis/{teacher_name}/num_samples={num_samples},cot_ratio={cot_ratio:g}"


def make_cartridge_path(
    model_name: str,
    teacher_name: str,
    num_samples: int,
    cot_ratio: float,
    cartridge_len: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    seed: int,
) -> str:
    return f"cartridge/{model_name}/teacher={teacher_name},num_samples={num_samples},cot_ratio={cot_ratio:g},cartridge_len={cartridge_len},learning_rate={learning_rate:g},batch_size={batch_size},num_epochs={num_epochs},seed={seed}"


def save_synthesis(
    path: str,
    rows: list[dict],
    tensors: dict[str, Tensor],
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=f"{path}/rows.json", path_or_fileobj=json.dumps(rows, indent=2).encode()),
        CommitOperationAdd(path_in_repo=f"{path}/logprobs.safetensors", path_or_fileobj=save(tensors)),
    ]
    api.create_commit(repo_name, operations=operations, commit_message=f"Save {path}")
    print(f"saved {path}/rows.json, {path}/logprobs.safetensors")


def get_pending_targets(
    entry_name: str,
    model_name: str,
    file_name: str,
) -> list[str]:
    files = set(HfApi().list_repo_files(get_repo_name()))
    targets = [f"base/{model_name}", f"icl/{model_name}"]
    for file in sorted(files):
        if file.startswith(f"cartridge/{model_name}/") and file.endswith("/cartridge.safetensors"):
            targets.append(str(Path(file).parent))
    targets = [target for target in targets if f"{entry_name}/{target}/{file_name}.json" not in files]
    print(f"{entry_name} pending {len(targets)}", *targets, sep="\n  ")
    return targets


def save_result(
    path: str,
    file_name: str,
    result: dict,
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(), path_in_repo=f"{path}/{file_name}.json", repo_id=repo_name, commit_message=f"Save {path}/{file_name}.json")
    print(f"saved {path}/{file_name}.json")


def save_tensors(
    path: str,
    file_name: str,
    tensors: dict[str, Tensor],
) -> None:
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    data = save({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()})
    api.upload_file(path_or_fileobj=data, path_in_repo=f"{path}/{file_name}.safetensors", repo_id=repo_name, commit_message=f"Save {path}/{file_name}.safetensors")
    print(f"saved {path}/{file_name}.safetensors")


def load_tensors(
    path: str,
    file_name: str,
) -> dict[str, Tensor]:
    return load_file(hf_hub_download(get_repo_name(), f"{path}/{file_name}.safetensors"))

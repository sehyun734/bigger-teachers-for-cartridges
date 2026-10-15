from collections import defaultdict
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from primitives.hub import get_repo_name

METHOD_COLORS = {"base": "tab:gray", "self": "tab:blue", "cross": "tab:orange", "icl": "tab:green"}
GRID_COLOR = "#e5e4df"


def get_method(
    target: str,
    model_name: str,
) -> str:
    # cartridge는 teacher가 student 자신이면 self, 아니면 cross로 봄.
    method, _, *settings = target.split("/")
    if method == "cartridge":
        return "self" if settings[0].startswith(f"teacher={model_name},") else "cross"
    return method


def get_labels(
    targets: list[str],
) -> list[str]:
    settings = {target: target.split("/")[2].split(",") for target in targets if target.startswith("cartridge/")}
    shared = set.intersection(*[set(target_settings) for target_settings in settings.values()]) if len(settings) > 1 else set()
    return ["\n".join(setting for setting in settings[target] if setting not in shared) if target in settings else target.split("/")[0] for target in targets]


def load_results(
    entry_name: str,
    model_name: str,
    file_name: str,
) -> dict[str, list[dict]]:
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=[f"{entry_name}/*/{model_name}/*{file_name}.json"])) / entry_name
    results = defaultdict(list)
    for file in sorted(result_dir.glob(f"*/{model_name}/**/{file_name}.json")):
        results[str(file.parent.relative_to(result_dir))].append(json.loads(file.read_text()))
    return dict(sorted(results.items(), key=lambda item: (list(METHOD_COLORS).index(get_method(item[0], model_name)), item[0])))


def style_axis(
    ax: Axes,
    targets: list[str],
    ylabel: str,
) -> None:
    ax.set_xticks(range(len(targets)), get_labels(targets), fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(
    fig: Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")

from dataclasses import dataclass
from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.plot import METHOD_COLORS, get_method, load_results, save_figure, style_axis


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"


def main() -> None:
    config = parse(Config)
    results = load_results("longhealth_logprob", config.model_name, "logprob")
    targets = list(results)

    fig, ax = plt.subplots(figsize=(2 + 2 * len(targets), 5), layout="constrained")
    for x, target in enumerate(targets):
        accuracy = results[target][0]["accuracy"]
        ax.bar(x, accuracy, width=0.6, color=METHOD_COLORS[get_method(target, config.model_name)], zorder=3)
        ax.annotate(f"{accuracy:.1%}", (x, accuracy), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    style_axis(ax, targets, "accuracy")
    handles = [Patch(color=color, label=method) for method, color in METHOD_COLORS.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} longhealth_logprob")
    save_figure(fig, Path(f"figures/{config.model_name}/longhealth_logprob.png"))


if __name__ == "__main__":
    main()

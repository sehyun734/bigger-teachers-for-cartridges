from dataclasses import dataclass
from pathlib import Path
import re
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.plot import GRID_COLOR, METHOD_COLORS, get_method, load_results, save_figure, style_axis


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"


def main() -> None:
    config = parse(Config)
    results = load_results("longhealth", config.model_name, "seed=*")
    targets = list(results)

    fig, ax = plt.subplots(figsize=(2 + 2.5 * len(targets), 6), layout="constrained")
    for x, target in enumerate(targets):
        color = METHOD_COLORS[get_method(target, config.model_name)]
        accuracies = [result["accuracy"] for result in results[target]]
        tagged_ratios = [mean(re.search(r"<answer>(.*?)</answer>", text, re.DOTALL) is not None for text in result["texts"]) for result in results[target]]
        accuracy = mean(accuracies)
        tagged_ratio = mean(tagged_ratios)
        ax.bar(x, accuracy, width=0.6, color=color, zorder=3)
        ax.bar(x, tagged_ratio - accuracy, width=0.6, bottom=accuracy, color=color, alpha=0.3, zorder=3)
        ax.bar(x, 1 - tagged_ratio, width=0.6, bottom=tagged_ratio, color="white", edgecolor=GRID_COLOR, hatch="//", zorder=3)
        ax.scatter([x] * len(accuracies), accuracies, color="black", s=12, zorder=4)
        ax.annotate(f"{accuracy:.1%}\ntagged {accuracy / tagged_ratio:.1%}", (x, 1), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    style_axis(ax, targets, "accuracy")
    ax.set_ylim(0, 1)

    handles = [Patch(color=color, label=method) for method, color in METHOD_COLORS.items()]
    handles += [Patch(color="gray", alpha=0.3, label="wrong"), Patch(facecolor="white", edgecolor="gray", hatch="//", label="no answer tag")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} longhealth")
    save_figure(fig, Path(f"figures/{config.model_name}/longhealth.png"))


if __name__ == "__main__":
    main()

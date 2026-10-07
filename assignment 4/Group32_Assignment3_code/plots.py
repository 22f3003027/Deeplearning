"""Required optimizer loss curves and selected train/test confusion matrices."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results,destination):
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    for architecture,hidden in results["config"]["architectures"].items():
        fig,axes=plt.subplots(1,2,figsize=(9.5,3.8))
        for row in results["runs"]:
            if row["architecture"]!=architecture:
                continue
            for ax in axes:
                ax.plot([h["epoch"] for h in row["history"]],
                        [h["train_loss"] for h in row["history"]],label=row["optimizer"],linewidth=1.4)
        axes[0].set_title("Full loss range"); axes[1].set_title("Low-loss detail (log epochs)")
        axes[1].set_xscale("log")
        axes[1].set_ylim(0,.5)
        for ax in axes:
            ax.set_xlabel("Epoch"); ax.set_ylabel("Mean training cross-entropy"); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle("Hidden layers: "+" -> ".join(map(str,hidden)))
        fig.tight_layout(); fig.savefig(destination/f"loss_{architecture}.png",dpi=190); plt.close(fig)
    selected=results["selected"]
    fig,axes=plt.subplots(1,2,figsize=(9,4.0))
    for ax,split in zip(axes,("train","test")):
        matrix=np.asarray(selected[split]["confusion_matrix"])
        ax.imshow(matrix,cmap="Blues",vmin=0,vmax=matrix.max())
        for (i,j),value in np.ndenumerate(matrix):
            ax.text(j,i,str(value),ha="center",va="center",color="white" if value>matrix.max()/2 else "#223044")
        ax.set_xticks(range(5),results["classes"]); ax.set_yticks(range(5),results["classes"])
        ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
        ax.set_title(f"{split.capitalize()} accuracy {selected[split]['accuracy']*100:.2f}%")
    fig.suptitle(f"Validation-selected model: {selected['hidden']}, {selected['optimizer']}")
    fig.tight_layout(); fig.savefig(destination/"confusion_selected.png",dpi=190); plt.close(fig)

"""Headless plots for all six assignment tasks."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def examples(data, classes):
    return {s: [int(np.flatnonzero(y == i)[0]) for i in range(len(classes))]
            for s, (_, y) in data.items()}


def reconstruction_grid(data, classes, models, output: Path, noisy_inputs=None):
    chosen = examples(data, classes)
    for split, (x, _) in data.items():
        indices = chosen[split]
        rows = [("Original", x[indices])]
        with torch.no_grad():
            for name, model in models.items():
                inputs = x if noisy_inputs is None else noisy_inputs[name][split]
                if noisy_inputs is not None:
                    rows.append((f"{name} input", inputs[indices]))
                model.eval()
                rows.append((name, model(torch.from_numpy(inputs[indices])).numpy()))
        fig, axes = plt.subplots(len(rows), len(classes), figsize=(8.3, len(rows)*0.85), squeeze=False)
        for r, (name, pixels) in enumerate(rows):
            for c, digit in enumerate(classes):
                ax = axes[r, c]
                ax.imshow(pixels[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if r == 0:
                    ax.set_title(f"Digit {digit}")
                if c == 0:
                    ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=7, fontsize=8)
        fig.suptitle(f"{'Denoising' if noisy_inputs else 'Autoencoder'} reconstructions - {split}", y=0.998)
        fig.tight_layout(rect=(0.12, 0, 1, 0.98))
        fig.savefig(output / f"{'denoising' if noisy_inputs else 'reconstructions'}_{split}.png", dpi=190)
        plt.close(fig)
    return chosen


def confusion_grid(selected, classes, title, destination):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.1))
    for ax, row in zip(axes.flat, selected):
        matrix = np.asarray(row["test"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.sum(axis=1)))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max()/2 else "#223044", fontsize=9)
        ax.set_xticks(range(5), classes); ax.set_yticks(range(5), classes)
        ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
        ax.set_title(f'd={row["dimension"]}, hidden={row["architecture"]}\nTest {row["test"]["accuracy"]*100:.2f}%')
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=180); plt.close(fig)


def weight_grids(models, destination):
    weights = {name: model.encoder[0].weight.detach().numpy() for name, model in models.items()}
    limit = float(np.percentile(np.abs(np.concatenate(list(weights.values()))), 99))
    stats = {}
    for name, array in weights.items():
        n = len(array); columns = min(16, math.ceil(math.sqrt(n)))
        rows = math.ceil(n/columns)
        fig, axes = plt.subplots(rows, columns, figsize=(8.6, rows*0.55+0.5), squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.set_axis_off()
            if i < n:
                ax.imshow(array[i].reshape(28,28), cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set_title(str(i+1), fontsize=5, pad=1)
        fig.suptitle(f"{name}: all {n} encoder weight images (shared scale +/-{limit:.3f})", fontsize=10)
        fig.tight_layout(pad=0.3, rect=(0,0,1,0.96))
        fig.savefig(destination / f"weights_{name}.png", dpi=220); plt.close(fig)
        pixels = array.reshape(n,28,28)
        tv = (np.abs(np.diff(pixels, axis=1)).mean() + np.abs(np.diff(pixels,axis=2)).mean()) / 2
        rms = np.sqrt(np.mean(array**2))
        stats[name] = {"weight_rms": float(rms), "neighbor_absolute_difference": float(tv),
                       "normalized_neighbor_difference": float(tv/rms), "shared_limit": limit}
    return stats


def overview_figures(results, destination):
    fig, ax = plt.subplots(figsize=(8,4.1))
    for method, selected in results["selected"].items():
        if method in ("pca", "shallow", "deep"):
            ax.plot([r["dimension"] for r in selected], [r["test"]["accuracy"]*100 for r in selected],
                    marker="o", label=method)
    ax.set_xlabel("Representation dimension"); ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(results["config"]["dimensions"]); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(destination / "accuracy_comparison.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,3.7))
    for name, row in results["autoencoders"].items():
        ax = axes[1 if row["deep"] else 0]
        ax.plot([h["epoch"] for h in row["history"]], [h["val_mse"] for h in row["history"]], label=name)
    for ax, title in zip(axes, ("Shallow / denoising", "Deep autoencoders")):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MSE"); ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(destination / "ae_learning_curves.png", dpi=180); plt.close(fig)

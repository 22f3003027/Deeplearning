"""Run the complete six-task study. See README.md for commands and conventions."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time

# Bound BLAS threads before importing numerical packages.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from data import load_data, fit_pca
from models import ARCHITECTURES, Autoencoder
from plots import reconstruction_grid, confusion_grid, weight_grids, overview_figures
from training import train_autoencoder, train_classifier, encode, evaluate_checkpoint, classify


def save_json(path, value):
    # Atomic progress snapshots make interrupted runs resumable.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_tables(results, output):
    def table(name, rows):
        if not rows:
            return
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    table("classification_validation.csv", [
        {"method": method, "dimension": r["dimension"], "architecture": r["architecture"],
         "validation_accuracy": r["val_accuracy"], "selected_epoch": r["best_epoch"]}
        for method, rows in results["classifiers"].items() for r in rows])
    table("classification_selected.csv", [
        {"method": method, "dimension": r["dimension"], "architecture": r["architecture"],
         "train_accuracy": r["train"]["accuracy"], "validation_accuracy": r["val_accuracy"],
         "test_accuracy": r["test"]["accuracy"]}
        for method, rows in results["selected"].items() for r in rows])
    table("reconstruction_errors.csv", [
        {"model": name, "dimension": r["dimension"], "noise": r["noise"],
         "train_mse": r["mse"]["train"], "validation_mse": r["mse"]["val"],
         "test_mse": r["mse"]["test"], "selected_epoch": r["best_epoch"]}
        for name, r in results["autoencoders"].items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "Group_32.zip")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--ae-epochs", type=int, default=80)
    parser.add_argument("--classifier-epochs", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    bundled_a3 = Path(__file__).resolve().parent / "assignment3_reference" / "assignment3_best.json"
    parser.add_argument("--assignment3-results", type=Path, default=bundled_a3 if bundled_a3.exists() else None,
                        help='JSON with architecture, validation_accuracy, test_accuracy (fractions).')
    args = parser.parse_args()
    if min(args.ae_epochs, args.classifier_epochs, args.threads) < 1:
        parser.error("Epoch counts and thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    threadpool_limits(limits=args.threads)
    output = args.output.resolve()
    for folder in (output, output/"figures", output/"checkpoints", output/"representations"):
        folder.mkdir(parents=True, exist_ok=True)
    config = {"dimensions": [32,64,128,256], "architectures": {k:list(v) for k,v in ARCHITECTURES.items()},
              "ae_epochs": args.ae_epochs, "classifier_epochs": args.classifier_epochs,
              "ae_patience": 12, "classifier_patience": 8, "batch_size": 256,
              "ae_lr": .001, "classifier_lr": .001, "seed": args.seed, "threads": args.threads}
    data, manifest = load_data(args.data.resolve(), output)
    labels = {s:y for s,(_,y) in data.items()}
    arrays = {s:x for s,(x,_) in data.items()}
    metrics = output/"metrics.json"
    if args.resume and metrics.exists():
        results = json.loads(metrics.read_text())
        if results["config"] != config or results["dataset_sha256"] != manifest["archive_sha256"]:
            raise ValueError("Resume configuration or dataset differs; use a new output directory")
    else:
        if metrics.exists():
            raise ValueError("Results already exist: use --resume or a different --output")
        results = {"config": config, "dataset_sha256": manifest["archive_sha256"],
                   "classes": manifest["classes"], "counts": manifest["counts"],
                   "environment": {"python":platform.python_version(), "torch":torch.__version__,
                                   "numpy":np.__version__, "device":"cpu"},
                   "autoencoders": {}, "classifiers": {}, "selected": {}, "status":"running"}
    if args.assignment3_results:
        a3 = json.loads(args.assignment3_results.read_text())
        if a3.get("dataset_sha256", manifest["archive_sha256"]) != manifest["archive_sha256"]:
            raise ValueError("Assignment 3 comparison uses a different dataset archive")
        for key in ("validation_accuracy", "test_accuracy"):
            if not 0 <= a3[key] <= 1:
                raise ValueError("Assignment 3 accuracies must be fractions between 0 and 1")
        results["assignment3"] = a3
    results.setdefault("assignment3", None)
    save_json(metrics, results)
    started = time.perf_counter()
    print(f"Dataset: classes={manifest['classes']}, counts={manifest['counts']}", flush=True)

    def study(method, dimension, features, architectures=ARCHITECTURES):
        rows = results["classifiers"].setdefault(method, [])
        for j, (name, hidden) in enumerate(architectures.items()):
            existing = next((r for r in rows if r["dimension"] == dimension and r["architecture"] == name), None)
            checkpoint = output/"checkpoints"/f"classifier_{method}_{dimension}_{name}.pt"
            if existing is not None:
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                continue
            seed = args.seed + dimension * 10 + j  # matched FCNN seeds across feature methods
            print(f"Classifier {method}, d={dimension}, hidden={name}", flush=True)
            model, row, inputs, targets = train_classifier(features, labels, hidden, config, seed, checkpoint)
            row.update({"dimension": dimension, "architecture": name, "checkpoint":checkpoint.name})
            rows.append(row); save_json(metrics, results)
            print(f"  validation accuracy {row['val_accuracy']:.4%}, epoch {row['best_epoch']}", flush=True)
        candidates = [r for r in rows if r["dimension"] == dimension]
        best = max(candidates, key=lambda r:(r["val_accuracy"], -r["val_loss"], -list(architectures).index(r["architecture"])))
        evaluated = evaluate_checkpoint(output/"checkpoints"/best["checkpoint"], features, labels)
        selected = {k:v for k,v in best.items() if k != "history"}
        selected.update(evaluated)
        selected_rows = results["selected"].setdefault(method, [])
        selected_rows[:] = [r for r in selected_rows if r["dimension"] != dimension] + [selected]
        selected_rows.sort(key=lambda r:r["dimension"])
        save_json(metrics, results)
        return selected

    print("Task 1: exact training-only PCA", flush=True)
    pca, variance = fit_pca(data, config["dimensions"], output/"checkpoints"/"pca.npz")
    results["pca_explained_variance"] = variance
    for dimension in config["dimensions"]:
        features = {s:x[:,:dimension].copy() for s,x in pca.items()}
        np.savez_compressed(output/"representations"/f"pca_{dimension}.npz", **features)
        study("pca", dimension, features)

    ae_models = {}
    def autoencoder(name, dimension, deep=False, noise=0):
        checkpoint = output/"checkpoints"/f"{name}.pt"
        # Fixed corruption is regenerated when resuming, so reconstruction figures are identical.
        seed = args.seed + dimension + (10000 if deep else 0) + int(noise*10000)
        if name in results["autoencoders"]:
            from models import corrupt
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = Autoencoder(dimension, deep); model.load_state_dict(saved["state_dict"])
            evaluation = {s:corrupt(torch.from_numpy(x), noise, torch.Generator().manual_seed(seed+1000+i)).numpy()
                          for i,(s,x) in enumerate(arrays.items())}
        else:
            print(f"Autoencoder {name}: d={dimension}, deep={deep}, noise={noise}", flush=True)
            model, metadata, evaluation = train_autoencoder(data, dimension, deep, noise, config, seed, checkpoint)
            results["autoencoders"][name] = metadata
            save_json(metrics, results)
            print(f"  reconstruction MSE {metadata['mse']}", flush=True)
        model.eval()
        features = encode(model, arrays)
        np.savez_compressed(output/"representations"/f"{name}.npz", **features)
        return model, features, evaluation

    for deep, method in ((False,"shallow"),(True,"deep")):
        for dimension in config["dimensions"]:
            name = f"{method}_{dimension}"
            model, features, _ = autoencoder(name, dimension, deep)
            ae_models[f"{'Deep' if deep else 'Shallow'} {dimension}"] = model
            study(method, dimension, features)
    results["example_indices"] = reconstruction_grid(data, manifest["classes"], ae_models, output/"figures")

    # The brief explicitly requires this test-based dimension choice for Task 5.
    best_shallow = max(results["selected"]["shallow"], key=lambda r:(r["test"]["accuracy"], -r["dimension"]))
    dimension = best_shallow["dimension"]
    results["task5_choice"] = {"dimension":dimension, "architecture":best_shallow["architecture"],
                               "selection_rule":"highest Task 3 test accuracy; smaller dimension on ties (brief requirement)"}
    dae_models, dae_inputs = {}, {}
    weight_models = {"shallow": ae_models[f"Shallow {dimension}"]}
    for noise in (.2,.4):
        name = f"denoising_{int(noise*100)}"
        model, features, evaluation = autoencoder(name, dimension, noise=noise)
        dae_models[f"DAE {int(noise*100)}%"] = model
        dae_inputs[f"DAE {int(noise*100)}%"] = evaluation
        weight_models[name] = model
        hidden = ARCHITECTURES[best_shallow["architecture"]]
        selected = study(name, dimension, features, {best_shallow["architecture"]:hidden})
        noisy_features = encode(model, evaluation)
        selected["noisy_input_evaluation"] = evaluate_checkpoint(output/"checkpoints"/selected["checkpoint"], noisy_features, labels)
        np.savez_compressed(output/"representations"/f"{name}_noisy.npz", **noisy_features)
        save_json(metrics, results)
    reconstruction_grid(data, manifest["classes"], dae_models, output/"figures", dae_inputs)
    results["weight_statistics"] = weight_grids(weight_models, output/"figures")

    if results["assignment3"] is None:
        print("Additional reference: FCNN on original 784 pixels (not an Assignment 3 result)", flush=True)
        study("raw_reference", 784, arrays)
    for method in ("pca","shallow","deep"):
        confusion_grid(results["selected"][method], manifest["classes"], f"{method.capitalize()} test confusion matrices", output/"figures"/f"confusion_{method}.png")
    overview_figures(results, output/"figures")
    results["status"] = "complete"
    results["last_invocation_seconds"] = time.perf_counter() - started
    save_json(metrics, results)
    write_tables(results, output)
    print(f"Completed all six tasks. Results: {output}", flush=True)
    print("Generate report: python report.py --results " + str(output), flush=True)


if __name__ == "__main__":
    main()

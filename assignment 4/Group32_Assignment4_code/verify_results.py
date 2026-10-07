"""Audit saved outputs by independently recomputing metrics from checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

from models import Autoencoder, ARCHITECTURES, corrupt
from training import encode, evaluate_checkpoint, reconstruction_mse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent/"results")
    args = parser.parse_args()
    root = args.results.resolve()
    r = json.loads((root/"metrics.json").read_text())
    manifest = json.loads((root/"dataset_manifest.json").read_text())
    assert r["status"] == "complete", "The study has not completed"
    torch.set_num_threads(r["config"]["threads"])
    threadpool_limits(r["config"]["threads"])
    with np.load(root/"dataset.npz") as saved:
        arrays = {s:saved[f"{s}_x"] for s in ("train","val","test")}
        labels = {s:saved[f"{s}_y"] for s in arrays}
    assert r["classes"] == manifest["classes"]
    for split,x in arrays.items():
        assert x.shape == (sum(manifest["counts"][split].values()),784)
        assert x.dtype == np.float32 and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
    with np.load(root/"checkpoints"/"pca.npz") as pca:
        assert np.allclose(pca["mean"],arrays["train"].mean(axis=0,dtype=np.float64))
        assert np.allclose(pca["components"] @ pca["components"].T, np.eye(256),atol=1e-8)
        for dimension in r["config"]["dimensions"]:
            with np.load(root/"representations"/f"pca_{dimension}.npz") as saved:
                for split,x in arrays.items():
                    assert saved[split].shape==(len(x),dimension)
                    projected=(x[:11].astype(np.float64)-pca["mean"]) @ pca["components"][:dimension].T
                    assert np.allclose(saved[split][:11],projected,atol=1e-5)
    for name,row in r["autoencoders"].items():
        checkpoint=torch.load(root/"checkpoints"/f"{name}.pt",map_location="cpu",weights_only=True)
        model=Autoencoder(row["dimension"],row["deep"]); model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for i,(split,x) in enumerate(arrays.items()):
            targets=torch.from_numpy(x)
            inputs=corrupt(targets,row["noise"],torch.Generator().manual_seed(row["seed"]+1000+i))
            measured=reconstruction_mse(model,inputs,targets)
            assert abs(measured-row["mse"][split])<1e-7,(name,split,"MSE mismatch")
            codes=encode(model,{split:x[:11]})[split]
            with np.load(root/"representations"/f"{name}.npz") as saved:
                assert saved[split].shape==(len(x),row["dimension"])
                assert np.allclose(saved[split][:11],codes,atol=2e-6)
        assert row["best_epoch"] == min(row["history"],key=lambda h:h["val_mse"])["epoch"]
    for method, selected in r["selected"].items():
        for row in selected:
            if method == "raw_reference":
                features=arrays
            else:
                filename = f"{method}_{row['dimension']}" if method in ("pca","shallow","deep") else method
                with np.load(root/"representations"/f"{filename}.npz") as saved:
                    features={s:saved[s] for s in arrays}
            actual=evaluate_checkpoint(root/"checkpoints"/row["checkpoint"],features,labels)
            for split in arrays:
                assert abs(actual[split]["accuracy"]-row[split]["accuracy"])<1e-12
                assert actual[split]["confusion_matrix"]==row[split]["confusion_matrix"]
                matrix=np.asarray(row[split]["confusion_matrix"])
                assert matrix.shape==(5,5) and matrix.sum()==len(labels[split])
                assert np.array_equal(matrix.sum(axis=1),np.bincount(labels[split],minlength=5))
            candidates=[c for c in r["classifiers"][method] if c["dimension"]==row["dimension"]]
            chosen=max(candidates,key=lambda c:(c["val_accuracy"],-c["val_loss"]))
            assert row["architecture"]==chosen["architecture"]
            if "noisy_input_evaluation" in row:
                with np.load(root/"representations"/f"{method}_noisy.npz") as saved:
                    noisy=evaluate_checkpoint(root/"checkpoints"/row["checkpoint"],{s:saved[s] for s in arrays},labels)
                assert noisy==row["noisy_input_evaluation"]
    assert len(r["autoencoders"])==10
    for method in ("pca","shallow","deep"):
        assert len(r["classifiers"][method])==12
        assert len(r["selected"][method])==4
    choice=max(r["selected"]["shallow"],key=lambda x:(x["test"]["accuracy"],-x["dimension"]))
    assert r["task5_choice"]["dimension"]==choice["dimension"]
    assert r["task5_choice"]["architecture"]==choice["architecture"]
    for level in (20,40):
        assert r["selected"][f"denoising_{level}"][0]["architecture"]==choice["architecture"]
    for path in (root/"figures").glob("*.png"):
        with Image.open(path) as image:
            image.verify()
    assert len(list((root/"figures").glob("*.png")))==14
    for name in ("classification_validation.csv","classification_selected.csv","reconstruction_errors.csv"):
        assert (root/name).stat().st_size>100
    if r.get("assignment3") is not None:
        baseline=r["assignment3"]
        assert baseline["dataset_sha256"]==r["dataset_sha256"]
        assert 0<=baseline["validation_accuracy"]<=1 and 0<=baseline["test_accuracy"]<=1
        comparison=root/"assignment3_comparison.csv"
        if comparison.exists():
            with comparison.open(newline="",encoding="utf-8") as stream:
                rows=list(csv.DictReader(stream))
            assert len(rows)==6 and rows[0]["study"]=="assignment3"
            assert abs(float(rows[0]["test_accuracy"])-baseline["test_accuracy"])<1e-12
            for row in rows[1:]:
                best=max(r["selected"][row["study"]],key=lambda x:(x["test"]["accuracy"],-x["dimension"]))
                assert int(row["dimension"])==best["dimension"]
                assert abs(float(row["test_accuracy"])-best["test"]["accuracy"])<1e-12
                difference=(best["test"]["accuracy"]-baseline["test_accuracy"])*100
                assert abs(float(row["difference_from_assignment3_percentage_points"])-difference)<1e-10
    summary={"status":"passed", "autoencoders_checked":10,
             "selected_classifiers_checked":sum(map(len,r["selected"].values())),
             "figures_checked":14, "checks":["split shapes and counts", "training-only PCA mean",
             "PCA orthogonality and projections", "saved latent representations", "recomputed reconstruction MSE",
             "validation-only FCNN selection", "recomputed checkpoint accuracies and confusion matrices",
             "denoising architecture/dimension reuse", "noisy-code evaluation", "figure and CSV integrity"]}
    if r.get("assignment3") is not None:
        summary["checks"].append("Assignment 3 dataset provenance and measured comparison differences")
    (root/"verification.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()

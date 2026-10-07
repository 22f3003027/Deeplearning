"""Recompute checkpoint metrics and audit all prescribed Assignment 3 conditions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from network import ARCHITECTURES,OPTIMIZERS,load_model,evaluate,parameter_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path(__file__).resolve().parent/"results")
    args=parser.parse_args(); root=args.results.resolve(); threadpool_limits(1)
    r=json.loads((root/"metrics.json").read_text())
    assert r["status"]=="complete"
    assert len(r["runs"])==len(ARCHITECTURES)*len(OPTIMIZERS)==21
    with np.load(root/"dataset.npz") as saved:
        arrays={s:np.ascontiguousarray(saved[f"{s}_x"],dtype=np.float64) for s in ("train","val","test")}
        labels={s:saved[f"{s}_y"] for s in arrays}
    for architecture in ARCHITECTURES:
        initial_w,initial_b=load_model(root/"checkpoints"/f"initial_{architecture}.npz")
        digest=parameter_hash(initial_w,initial_b)
        runs=[row for row in r["runs"] if row["architecture"]==architecture]
        assert len(runs)==7 and {row["initial_parameter_sha256"] for row in runs}=={digest}
    for row in r["runs"]:
        assert 3<=len(row["hidden"])<=5 and tuple(row["hidden"])==ARCHITECTURES[row["architecture"]]
        settings=OPTIMIZERS[row["optimizer"]]
        expected_batch=len(labels["train"]) if settings["batch_size"]=="full" else 1
        assert row["batch_size"]==expected_batch
        epochs=row["convergence_epochs"]
        assert epochs==len(row["history"]) and epochs>=2
        assert row["optimizer_steps"]==epochs*(len(labels["train"]) if expected_batch==1 else 1)
        for index in range(1,epochs):
            current=row["history"][index]; previous=row["history"][index-1]
            difference=abs(current["train_loss"]-previous["train_loss"])
            assert abs(difference-current["loss_difference"])<1e-14
            assert (difference<1e-4) if index==epochs-1 else (difference>=1e-4)
        assert row["final_loss_difference"]<1e-4
        weights,biases=load_model(root/"checkpoints"/row["checkpoint"])
        for split in ("train","val"):
            actual=evaluate(arrays[split],labels[split],weights,biases)
            assert abs(actual["loss"]-row[split]["loss"])<1e-12
            assert actual["accuracy"]==row[split]["accuracy"]
            assert actual["confusion_matrix"]==row[split]["confusion_matrix"]
    winner=max(r["runs"],key=lambda row:(row["val"]["accuracy"],-row["val"]["loss"],-len(row["hidden"])))
    assert winner["name"]==r["selected"]["name"]
    weights,biases=load_model(root/"checkpoints"/winner["checkpoint"])
    assert evaluate(arrays["test"],labels["test"],weights,biases)==r["selected"]["test"]
    best=json.loads((root/"assignment3_best.json").read_text())
    assert best["test_accuracy"]==r["selected"]["test"]["accuracy"]
    assert best["dataset_sha256"]==r["dataset_sha256"]
    for path in (root/"figures").glob("*.png"):
        with Image.open(path) as image: image.verify()
    assert len(list((root/"figures").glob("*.png")))==4
    summary={"status":"passed","runs_checked":21,"optimizers":7,"architectures":3,
             "checks":["identical initialization within each architecture","prescribed online/full batch sizes",
                       "optimizer step counts","loss delta as sole stopping rule","first qualifying stopping epoch",
                       "recomputed train/validation checkpoint metrics","validation-only architecture/optimizer selection",
                       "selected test confusion matrix","comparison record provenance","all four figure files"]}
    (root/"verification.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()

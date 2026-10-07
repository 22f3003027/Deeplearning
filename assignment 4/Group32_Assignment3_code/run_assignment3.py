"""Run Assignment 3 exactly: seven optimizers, matched starts, loss-delta stopping."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from data import load_data
from network import (ARCHITECTURES,OPTIMIZERS,initialize,clone,buffers,parameter_hash,
                     online_epoch,full_batch_update,evaluate,save_model)


def should_stop(previous,current,threshold=1e-4):
    return previous is not None and abs(current-previous)<threshold


def snapshot(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path(__file__).resolve().parent.parent/"Group_32.zip")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--seed",type=int,default=32)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--only-architecture",choices=list(ARCHITECTURES),
                        help="Optional isolated worker; partial results must be merged before final test selection")
    parser.add_argument("--only-optimizers",nargs="+",choices=list(OPTIMIZERS),
                        help="Optional independent optimizer subset for an isolated worker")
    args=parser.parse_args()
    torch.set_num_threads(1)
    flush_denormals=bool(torch.set_flush_denormal(True))
    threadpool_limits(1)  # Small online matrix-vector products must not start BLAS thread teams.
    output=args.output.resolve()
    for directory in (output,output/"checkpoints",output/"figures"):
        directory.mkdir(parents=True,exist_ok=True)
    data,manifest=load_data(args.data.resolve(),output)
    arrays={s:np.ascontiguousarray(x,dtype=np.float64) for s,(x,_) in data.items()}
    labels={s:y for s,(_,y) in data.items()}
    config={"architectures":{k:list(v) for k,v in ARCHITECTURES.items()},"optimizers":OPTIMIZERS,
            "learning_rate":.001,"momentum":.9,"rmsprop_alpha":.99,"epsilon":1e-8,
            "adam_betas":[.9,.999],"threshold":1e-4,"seed":args.seed,
            "stopping_loss":"post-epoch mean training cross-entropy","activation":"ReLU",
            "dtype":"float64","epoch_limit":None}
    path=output/"metrics.json"
    if path.exists():
        if not args.resume:
            raise ValueError("Use --resume for existing results, or choose a new --output")
        r=json.loads(path.read_text())
        if r["config"]!=config or r["dataset_sha256"]!=manifest["archive_sha256"]:
            raise ValueError("Resume dataset/configuration mismatch")
    else:
        r={"config":config,"dataset_sha256":manifest["archive_sha256"],"classes":manifest["classes"],
           "counts":manifest["counts"],"runs":[],"status":"running",
           "environment":{"python":platform.python_version(),"numpy":np.__version__,
                          "backend":"NumPy BLAS + Numba exact online updates","device":"cpu"}}
    snapshot(path,r)
    r["environment"]["cpu_flush_denormals"]=flush_denormals
    for completed in r["runs"]:
        completed.setdefault("cpu_flush_denormals",False)
    for architecture,hidden in ARCHITECTURES.items():
        if args.only_architecture and architecture!=args.only_architecture:
            continue
        seed=args.seed+len(hidden)*100
        initial_w,initial_b=initialize(hidden,seed)
        initial_hash=parameter_hash(initial_w,initial_b)
        save_model(output/"checkpoints"/f"initial_{architecture}.npz",initial_w,initial_b)
        for optimizer,settings in OPTIMIZERS.items():
            if args.only_optimizers and optimizer not in args.only_optimizers:
                continue
            name=f"{architecture}_{optimizer}"
            if any(row["name"]==name for row in r["runs"]):
                continue
            weights,biases=clone(initial_w),clone(initial_b)
            first_w,first_b=buffers(weights,biases)
            second_w,second_b=buffers(weights,biases)
            rng=np.random.default_rng(seed+1000)
            previous=None; epoch=0; step=0; history=[]; started=time.perf_counter()
            batch_size=len(arrays["train"]) if settings["batch_size"]=="full" else 1
            print(f"{name}: hidden={hidden}, batch={batch_size}; stopping only at loss delta <1e-4",flush=True)
            while True:
                epoch+=1
                if batch_size==1:
                    indices=rng.permutation(len(labels["train"]))
                    step=online_epoch(arrays["train"],labels["train"],indices,weights,biases,
                                      first_w,first_b,second_w,second_b,settings["kind"],step)
                else:
                    full_batch_update(arrays["train"],labels["train"],weights,biases,
                                      first_w,first_b,second_w,second_b,settings["kind"])
                    step+=1
                train=evaluate(arrays["train"],labels["train"],weights,biases)
                current=train["loss"]
                if not np.isfinite(current):
                    raise FloatingPointError(f"Nonfinite training loss in {name}, epoch {epoch}")
                difference=None if previous is None else abs(current-previous)
                history.append({"epoch":epoch,"train_loss":current,"loss_difference":difference})
                if epoch==1 or epoch%10==0:
                    print(f"  epoch {epoch}: CE={current:.6f}, delta={difference}, elapsed={time.perf_counter()-started:.1f}s",flush=True)
                if should_stop(previous,current,config["threshold"]):
                    break
                previous=current
            validation=evaluate(arrays["val"],labels["val"],weights,biases)
            checkpoint=output/"checkpoints"/f"{name}.npz"
            save_model(checkpoint,weights,biases)
            row={"name":name,"architecture":architecture,"hidden":list(hidden),"optimizer":optimizer,
                 "batch_size":batch_size,"seed":seed,"initial_parameter_sha256":initial_hash,
                 "convergence_epochs":epoch,"optimizer_steps":step,"final_loss_difference":difference,
                 "history":history,"train":train,"val":validation,"checkpoint":checkpoint.name,
                 "seconds":time.perf_counter()-started,"stopping_reason":"abs successive epoch mean training CE <1e-4"}
            row["cpu_flush_denormals"]=flush_denormals
            r["runs"].append(row); snapshot(path,r)
            print(f"  converged in {epoch} epochs: train={train['accuracy']:.4%}, val={validation['accuracy']:.4%}",flush=True)
    if len(r["runs"]) != len(ARCHITECTURES)*len(OPTIMIZERS):
        r["status"]="partial"; snapshot(path,r)
        print(f"Isolated worker complete: {len(r['runs'])} converged runs; no test selection was performed",flush=True)
        return
    selected=max(r["runs"],key=lambda row:(row["val"]["accuracy"],-row["val"]["loss"],-len(row["hidden"])))
    from network import load_model
    weights,biases=load_model(output/"checkpoints"/selected["checkpoint"])
    r["selected"]={k:v for k,v in selected.items() if k!="history"}
    r["selected"]["test"]=evaluate(arrays["test"],labels["test"],weights,biases)
    # Per-optimizer architecture winners are validation-only summaries; no further tests.
    r["best_architecture_per_optimizer"]={name:max((row for row in r["runs"] if row["optimizer"]==name),
                                                   key=lambda row:(row["val"]["accuracy"],-row["val"]["loss"]))["architecture"]
                                           for name in OPTIMIZERS}
    r["status"]="complete"
    snapshot(path,r)
    comparison={"architecture":"784 -> "+" -> ".join(map(str,selected["hidden"]))+" -> 5",
                "optimizer":selected["optimizer"],"validation_accuracy":selected["val"]["accuracy"],
                "test_accuracy":r["selected"]["test"]["accuracy"],"training_accuracy":selected["train"]["accuracy"],
                "convergence_epochs":selected["convergence_epochs"],"dataset_sha256":manifest["archive_sha256"],
                "source":"Assignment 3 reproduced from supplied PDF; newly measured on the original Group 32 splits",
                "metrics_file":"Group32_Assignment3_code/results/metrics.json"}
    snapshot(output/"assignment3_best.json",comparison)
    with (output/"optimizer_results.csv").open("w",newline="",encoding="utf-8") as stream:
        columns=["architecture","hidden","optimizer","batch_size","convergence_epochs","optimizer_steps",
                 "train_accuracy","validation_accuracy","final_training_loss","final_loss_difference"]
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
        for row in r["runs"]:
            writer.writerow({"architecture":row["architecture"],"hidden":str(row["hidden"]),"optimizer":row["optimizer"],
                             "batch_size":row["batch_size"],"convergence_epochs":row["convergence_epochs"],
                             "optimizer_steps":row["optimizer_steps"],"train_accuracy":row["train"]["accuracy"],
                             "validation_accuracy":row["val"]["accuracy"],"final_training_loss":row["train"]["loss"],
                             "final_loss_difference":row["final_loss_difference"]})
    from plots import plot_results
    plot_results(r,output/"figures")
    print("Assignment 3 completed. Best model:",comparison,flush=True)


if __name__=="__main__":
    main()

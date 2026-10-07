"""Merge independent completed optimizer workers, before global test selection.

Stop any writer to the destination before invoking. Dataset and experiment
configurations must match exactly. Duplicate converged runs must agree.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from network import load_model,parameter_hash
from run_assignment3 import snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--source",type=Path,action="append",required=True)
    args=parser.parse_args()
    destination=args.destination.resolve()
    target=json.loads((destination/"metrics.json").read_text())
    names={row["name"]:row for row in target["runs"]}
    for folder in args.source:
        source=json.loads((folder/"metrics.json").read_text())
        if source["status"] not in ("partial","complete"):
            raise ValueError(f"Worker is still running: {folder}")
        if source["config"]!=target["config"] or source["dataset_sha256"]!=target["dataset_sha256"]:
            raise ValueError(f"Mismatched worker: {folder}")
        for row in source["runs"]:
            if row["name"] in names:
                existing=names[row["name"]]
                assert existing["initial_parameter_sha256"]==row["initial_parameter_sha256"]
                assert existing["convergence_epochs"]==row["convergence_epochs"]
                assert abs(existing["train"]["loss"]-row["train"]["loss"])<1e-12
                assert existing["val"]["accuracy"]==row["val"]["accuracy"]
                continue
            shutil.copy2(folder/"checkpoints"/row["checkpoint"],destination/"checkpoints"/row["checkpoint"])
            initial=f"initial_{row['architecture']}.npz"
            source_w,source_b=load_model(folder/"checkpoints"/initial)
            assert parameter_hash(source_w,source_b)==row["initial_parameter_sha256"]
            target_initial=destination/"checkpoints"/initial
            if target_initial.exists():
                target_w,target_b=load_model(target_initial)
                assert parameter_hash(target_w,target_b)==row["initial_parameter_sha256"]
            else:
                shutil.copy2(folder/"checkpoints"/initial,target_initial)
            target["runs"].append(row); names[row["name"]]=row
    target["status"]="running"  # Final test evaluation belongs to the full resumed study.
    snapshot(destination/"metrics.json",target)
    print(f"Merged {len(target['runs'])} converged runs. Resume the main study for final selection, tables, and plots.")


if __name__=="__main__": main()

"""Deterministic CPU training and validation-only checkpoint selection."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from torch import nn

from models import Autoencoder, Classifier, corrupt


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def encode(model, arrays, batch_size=1024):
    model.eval()
    return {s: np.concatenate([model.encoder(torch.from_numpy(x[i:i+batch_size])).numpy()
                              for i in range(0, len(x), batch_size)]) for s, x in arrays.items()}


@torch.no_grad()
def reconstruction_mse(model, inputs, targets, batch_size=1024):
    model.eval()
    squared_error = 0.0
    for i in range(0, len(inputs), batch_size):
        prediction = model(inputs[i:i+batch_size])
        squared_error += (prediction - targets[i:i+batch_size]).square().sum().item()
    return squared_error / targets.numel()


def train_autoencoder(data, bottleneck, deep, noise, config, seed, checkpoint):
    seed_all(seed)
    model = Autoencoder(bottleneck, deep)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["ae_lr"])
    targets = {s: torch.from_numpy(x) for s, (x, _) in data.items()}
    # Fixed independent corruption for reporting/checkpoint selection; fresh noise during training.
    evaluation_inputs = {s: corrupt(x, noise, torch.Generator().manual_seed(seed + 1000 + i))
                         for i, (s, x) in enumerate(targets.items())}
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config["ae_epochs"] + 1):
        model.train()
        indices = torch.randperm(len(targets["train"]))
        total = 0.0
        for idx in indices.split(config["batch_size"]):
            clean = targets["train"][idx]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(corrupt(clean, noise)), clean)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        val_loss = reconstruction_mse(model, evaluation_inputs["val"], targets["val"])
        history.append({"epoch": epoch, "train_mse": total / len(indices), "val_mse": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch % 20 == 0:
            print(f"  AE epoch {epoch}: val MSE={val_loss:.6f}", flush=True)
        if epoch >= 20 and stale >= config["ae_patience"]:
            break
    model.load_state_dict(best_state)
    errors = {s: reconstruction_mse(model, evaluation_inputs[s], targets[s]) for s in targets}
    clean_errors = {s: reconstruction_mse(model, targets[s], targets[s]) for s in targets}
    torch.save({"state_dict": best_state, "bottleneck": bottleneck, "deep": deep,
                "noise_probability": noise, "seed": seed, "selected_epoch": best_epoch,
                "activation": "sigmoid"}, checkpoint)
    metadata = {"dimension": bottleneck, "deep": deep, "noise": noise, "seed": seed,
                "best_epoch": best_epoch, "epochs_run": len(history), "history": history,
                "mse": errors, "clean_input_mse": clean_errors,
                "seconds": time.perf_counter() - started}
    return model, metadata, {s: x.numpy() for s, x in evaluation_inputs.items()}


@torch.no_grad()
def classify(model, inputs, labels, classes=5, batch_size=1024):
    model.eval()
    predictions = torch.cat([model(inputs[i:i+batch_size]).argmax(1)
                             for i in range(0, len(inputs), batch_size)])
    matrix = torch.bincount(labels * classes + predictions, minlength=classes**2).reshape(classes, classes)
    return {"accuracy": float((predictions == labels).double().mean()),
            "confusion_matrix": matrix.tolist()}


def standardize(features):
    mean = features["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features["train"].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return {s: torch.from_numpy(np.ascontiguousarray((x - mean) / std)) for s, x in features.items()}, mean, std


def train_classifier(features, labels, hidden, config, seed, checkpoint):
    seed_all(seed)
    inputs, mean, std = standardize(features)
    targets = {s: torch.from_numpy(y) for s, y in labels.items()}
    model = Classifier(inputs["train"].shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["classifier_lr"])
    best_accuracy, best_loss, best_state, best_epoch, stale = -1.0, float("inf"), None, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config["classifier_epochs"] + 1):
        model.train()
        indices = torch.randperm(len(inputs["train"]))
        train_loss = 0.0
        for idx in indices.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs["train"][idx]), targets["train"][idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(inputs["val"])
            accuracy = float((logits.argmax(1) == targets["val"]).double().mean())
            val_loss = nn.functional.cross_entropy(logits, targets["val"]).item()
        history.append({"epoch": epoch, "train_loss": train_loss / len(indices),
                        "val_loss": val_loss, "val_accuracy": accuracy})
        if accuracy > best_accuracy or (accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss = accuracy, val_loss
            best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch >= 12 and stale >= config["classifier_patience"]:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "input_dim": inputs["train"].shape[1],
                "hidden": list(hidden), "feature_mean": torch.from_numpy(mean),
                "feature_std": torch.from_numpy(std), "seed": seed, "selected_epoch": best_epoch}, checkpoint)
    # Test results are calculated by the caller only after selecting an architecture.
    metadata = {"seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
                "val_accuracy": best_accuracy, "val_loss": best_loss, "history": history,
                "seconds": time.perf_counter() - started}
    return model, metadata, inputs, targets


def evaluate_checkpoint(checkpoint, features, labels):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Classifier(saved["input_dim"], tuple(saved["hidden"]))
    model.load_state_dict(saved["state_dict"])
    inputs = {s: (torch.from_numpy(x) - saved["feature_mean"]) / saved["feature_std"]
              for s, x in features.items()}
    return {s: classify(model, inputs[s], torch.from_numpy(labels[s])) for s in features}

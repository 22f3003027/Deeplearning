"""Load only the supplied group archive, preserving its original splits."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def load_data(archive: Path, output: Path):
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "dataset.npz"
    manifest_path = output / "dataset_manifest.json"
    if cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["archive_sha256"] == digest:
            with np.load(cache) as saved:
                data = {s: (saved[f"{s}_x"], saved[f"{s}_y"]) for s in ("train", "val", "test")}
            return data, manifest
    with zipfile.ZipFile(archive) as z:
        entries = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(set(entries)) != len(entries):
            raise ValueError("Duplicate archive paths")
        classes = sorted({int(n.split("/")[-2]) for n in entries})
        if len(classes) != 5:
            raise ValueError(f"Expected five digit classes, found {classes}")
        label_map = {digit: i for i, digit in enumerate(classes)}
        data, counts, names, content_hashes = {}, {}, {}, {}
        for split in ("train", "val", "test"):
            members = sorted(n for n in entries if n.split("/")[-3] == split)
            if not members:
                raise ValueError(f"Missing split: {split}")
            x = np.empty((len(members), 784), np.float32)
            y = np.empty(len(members), np.int64)
            content_hashes[split] = set()
            for i, member in enumerate(members):
                raw = z.read(member)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.size != (28, 28):
                        raise ValueError(f"Expected 28x28: {member}")
                    pixels = np.asarray(image.convert("L"), dtype=np.uint8)
                x[i] = pixels.reshape(-1) / np.float32(255)
                y[i] = label_map[int(member.split("/")[-2])]
                content_hashes[split].add(hashlib.sha256(pixels.tobytes()).hexdigest())
            data[split] = (x, y)
            counts[split] = {str(c): int((y == label_map[c]).sum()) for c in classes}
            if any(v == 0 for v in counts[split].values()):
                raise ValueError(f"A class is missing from {split}")
            names[split] = members
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{a}_{b}"] = len(content_hashes[a] & content_hashes[b])
    manifest = {
        "archive": archive.name, "archive_sha256": digest, "classes": classes,
        "counts": counts, "pixel_range": [0, 1], "shape": [28, 28],
        "identical_pixel_hash_overlap": overlaps, "source_paths": names,
    }
    np.savez_compressed(cache, **{f"{s}_{k}": v for s, pair in data.items() for k, v in zip(("x", "y"), pair)})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data, manifest


def fit_pca(data, dimensions, destination: Path):
    """Exact eigendecomposition of training covariance; no evaluation-set fitting."""
    train = data["train"][0].astype(np.float64)
    mean = train.mean(axis=0)
    centered = train - mean
    covariance = centered.T @ centered / (len(train) - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0)
    components = vectors[:, order[:max(dimensions)]].T
    # Resolve eigenvector sign ambiguity deterministically.
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    np.savez_compressed(destination, mean=mean, components=components, eigenvalues=values)
    features = {s: ((x.astype(np.float64) - mean) @ components.T).astype(np.float32)
                for s, (x, _) in data.items()}
    variance = {str(d): float(values[:d].sum() / values.sum()) for d in dimensions}
    return features, variance

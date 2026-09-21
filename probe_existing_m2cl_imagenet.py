#!/usr/bin/env python3
"""Phase 0: Three-Way Domain Leakage Analysis (ERM vs M2 vs M2-CL).

Standalone script — no imports from the domain_leakage package.
Dependencies: torch, torchvision, numpy, sklearn, PIL.

Usage:
    python scripts/probe_existing_m2cl.py \\
        --checkpoint_dir /content/drive/MyDrive/m2cl_checkpoints \\
        --data_root /content/PACS \\
        --domains photo art_painting

    # All available domains (slow):
    python scripts/probe_existing_m2cl.py \\
        --checkpoint_dir /content/drive/MyDrive/m2cl_checkpoints \\
        --data_root /content/PACS \\
        --domains art_painting cartoon photo sketch

Scientific protocol:
    Domain probe: source-only 70/30 stratified split. Held-out domain
                  NEVER used for domain probing.
    Class probe:  source train -> held-out test (measures DG).
    Significance: permutation test, p < 0.05 gate.
    Uncertainty:  bootstrap 95% CI on real accuracy (1000 resamples).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS: List[str] = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES: int = 7
NUM_DOMAINS: int = 4

PROBE_EPOCHS: int = 30
PERM_PROBE_EPOCHS: int = 5
PROBE_LR: float = 1e-3
PROBE_BATCH_SIZE: int = 64
BOOTSTRAP_RESAMPLES: int = 1000

# Probe layer names and their output dimensions, keyed by model_type.
ERM_PROBE_LAYERS: List[str] = ["backbone_final"]
M2_PROBE_LAYERS: List[str] = [
    "layer1_features",
    "layer2_features",
    "layer3_features",
    "layer4_features",
    "gap_backbone",
    "e1_contrastive",
    "e2_contrastive",
    "e3_contrastive",
    "e4_contrastive",
]

ERM_LAYER_DIMS: Dict[str, int] = {"backbone_final": 512}
M2_LAYER_DIMS: Dict[str, int] = {
    "layer1_features": 64,
    "layer2_features": 128,
    "layer3_features": 256,
    "layer4_features": 512,
    "gap_backbone": 512,
    "e1_contrastive": 384,
    "e2_contrastive": 384,
    "e3_contrastive": 256,
    "e4_contrastive": 256,
}

# Which domains have checkpoints for each model type.
AVAILABLE_DOMAINS: Dict[str, List[str]] = {
    "erm": ["art_painting", "cartoon", "photo", "sketch"],
    "m2": ["art_painting", "cartoon", "photo", "sketch"],
    "m2cl": ["art_painting", "cartoon", "photo"],
}

CHECKPOINT_NAMES: Dict[str, str] = {
    "erm": "erm_{domain}.pt",
    "m2": "m2_{domain}.pt",
    "m2cl": "m2cl_{domain}.pt",
}

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 0: Three-way domain leakage analysis (ERM / M2 / M2-CL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: photo + art_painting for all three model types (6 runs)
  python scripts/probe_existing_m2cl.py \\
      --checkpoint_dir /content/drive/MyDrive/m2cl_checkpoints \\
      --data_root /content/PACS

  # All available domains
  python scripts/probe_existing_m2cl.py \\
      --checkpoint_dir /content/drive/MyDrive/m2cl_checkpoints \\
      --data_root /content/PACS \\
      --domains art_painting cartoon photo sketch

  # Single model type for debugging
  python scripts/probe_existing_m2cl.py \\
      --checkpoint_dir /content/drive/MyDrive/m2cl_checkpoints \\
      --data_root /content/PACS \\
      --model_types m2cl --domains photo --n_permutations 10
        """,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing all .pt checkpoint files",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="PACS dataset root (contains art_painting/, cartoon/, photo/, sketch/)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["photo", "art_painting"],
        choices=DOMAINS,
        help="Held-out domains to probe (default: photo art_painting)",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=["erm", "m2", "m2cl"],
        choices=["erm", "m2", "m2cl"],
        help="Model types to probe (default: erm m2 m2cl)",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=100,
        help="Permutations for significance test (default: 100)",
    )
    parser.add_argument(
        "--domain_split_ratio",
        type=float,
        default=0.3,
        help="Fraction of source data held for domain probe test (default: 0.3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="DataLoader batch size for feature extraction (default: 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/phase0_results.json",
        help="Path to save JSON results (default: scripts/phase0_results.json)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Architecture Definitions (verbatim from training notebook)
# ─────────────────────────────────────────────────────────────────────────────


class ERM(nn.Module):
    """Standard ResNet-18 fine-tuned with cross-entropy (Section II baseline)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(bb.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.classifier(self.features(x).flatten(1)), None


class ConcentrationPipeline(nn.Module):
    """1x1 Conv -> BN -> ReLU -> Spatial Dropout -> AdaptiveMaxPool -> Flatten -> Linear -> ReLU.

    One linear layer per pipeline, exactly as drawn in Fig. 4.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        pool_size: int,
        mlp_dim: int = 128,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.drop = nn.Dropout2d(p=0.5)
        self.pool = nn.AdaptiveMaxPool2d(pool_size)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mid_channels * pool_size * pool_size, mlp_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = mlp_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.mlp(self.pool(self.drop(h)))


class ExtractionBlock(nn.Module):
    """Parallel concentration pipelines concatenated along feature dimension.

    Early layers (layer1, layer2): 3 pipelines, pool sizes 8x8, 4x4, 2x2 -> 384-d
    Later layers (layer3, layer4): 2 pipelines, pool sizes 7x7, 3x3 -> 256-d
    """

    def __init__(
        self,
        in_channels: int,
        r: int = 4,
        pool_sizes: Tuple[int, ...] = (4, 2),
        mlp_dim: int = 128,
    ) -> None:
        super().__init__()
        mid = max(in_channels // r, 8)
        self.pipes = nn.ModuleList(
            [ConcentrationPipeline(in_channels, mid, ps, mlp_dim) for ps in pool_sizes]
        )
        self.out_dim = mlp_dim * len(pool_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([p(x) for p in self.pipes], dim=1)


class M2(nn.Module):
    """M2 / M2-CL multiscale feature extractor + classifier.

    Probe dimensions when use_cl=True:
        layer1 -> eb1 -> e1_contrastive: 384-d
        layer2 -> eb2 -> e2_contrastive: 384-d
        layer3 -> eb3 -> e3_contrastive: 256-d
        layer4 -> eb4 -> e4_contrastive: 256-d
        avgpool(layer4) -> gap_backbone:  512-d
        Total concat (1792-d) -> classifier
    """

    def __init__(
        self, num_classes: int, use_cl: bool = False, mlp_dim: int = 128
    ) -> None:
        super().__init__()
        self.use_cl = use_cl
        bb = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.avgpool = bb.avgpool
        self.eb1 = ExtractionBlock(64, r=4, pool_sizes=(8, 4, 2), mlp_dim=mlp_dim)
        self.eb2 = ExtractionBlock(128, r=4, pool_sizes=(8, 4, 2), mlp_dim=mlp_dim)
        self.eb3 = ExtractionBlock(256, r=4, pool_sizes=(7, 3), mlp_dim=mlp_dim)
        self.eb4 = ExtractionBlock(512, r=4, pool_sizes=(7, 3), mlp_dim=mlp_dim)
        total = (
            self.eb1.out_dim
            + self.eb2.out_dim
            + self.eb3.out_dim
            + self.eb4.out_dim
            + 512
        )
        self.classifier = nn.Linear(total, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        e1, e2 = self.eb1(f1), self.eb2(f2)
        e3, e4 = self.eb3(f3), self.eb4(f4)
        gap = self.avgpool(f4).flatten(1)
        z = torch.cat([e1, e2, e3, e4, gap], dim=1)
        reps = [e1, e2, e3, e4] if self.use_cl else None
        return self.classifier(z), reps


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model(
    checkpoint_path: str, model_type: str, device: str
) -> nn.Module:
    """Load a checkpoint into the correct model class with strict key verification.

    ERM  -> ERM(num_classes=7)
    m2   -> M2(num_classes=7, use_cl=True)   # use_cl=True to get reps back
    m2cl -> M2(num_classes=7, use_cl=True)   # identical state_dict structure

    Args:
        checkpoint_path: absolute path to .pt file
        model_type: 'erm', 'm2', or 'm2cl'
        device: torch device string

    Returns:
        Loaded model in eval() mode.

    Raises:
        AssertionError: if state_dict keys do not match exactly.
        RuntimeError: if checkpoint appears empty (param_sum < 1.0).
    """
    if model_type == "erm":
        model: nn.Module = ERM(num_classes=NUM_CLASSES)
    else:
        # Both m2 and m2cl save identical key sets; use_cl only affects
        # the forward() return value, not the saved parameters.
        model = M2(num_classes=NUM_CLASSES, use_cl=True)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, (
        f"Key mismatch loading {checkpoint_path}: "
        f"missing={missing}, unexpected={unexpected}"
    )

    param_sum = sum(p.abs().sum().item() for p in model.parameters())
    print(f"  Loaded {model_type} from {Path(checkpoint_path).name}: "
          f"param_sum={param_sum:.2f}")
    if param_sum < 1.0:
        raise RuntimeError(
            f"{checkpoint_path} did not load correctly (param_sum={param_sum:.4f})."
        )

    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class PACSDomainDataset(Dataset):
    """Minimal PACS dataset: yields (image_tensor, class_label).

    Loads all images from the specified domains. Domain identity is tracked
    separately via a parallel list rather than encoded in __getitem__, to
    keep the DataLoader interface simple.
    """

    def __init__(self, root: str, domain_names: List[str]) -> None:
        self.samples: List[Tuple[str, int]] = []  # (path, class_idx)
        self.domain_indices: List[int] = []  # parallel to self.samples

        # Discover classes from the first available domain
        first_domain_path = os.path.join(root, domain_names[0])
        classes = sorted(
            c
            for c in os.listdir(first_domain_path)
            if os.path.isdir(os.path.join(first_domain_path, c))
        )
        class_to_idx = {c: i for i, c in enumerate(classes)}

        for domain_idx, domain_name in enumerate(domain_names):
            domain_path = os.path.join(root, domain_name)
            if not os.path.isdir(domain_path):
                print(f"  WARNING: {domain_path} not found — skipping")
                continue
            for cls_name in classes:
                cls_path = os.path.join(domain_path, cls_name)
                if not os.path.isdir(cls_path):
                    continue
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(
                            (os.path.join(cls_path, fname), class_to_idx[cls_name])
                        )
                        self.domain_indices.append(domain_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return EVAL_TRANSFORM(img), label


def build_loaders(
    data_root: str,
    held_out_domain: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Build source and held-out DataLoaders with domain label tensors.

    Args:
        data_root: path to PACS root directory
        held_out_domain: domain name to hold out
        batch_size: DataLoader batch size

    Returns:
        source_loader: all source domain images
        test_loader: held-out domain images
        source_domain_labels: [N_source] int tensor of domain indices
        test_domain_labels: [N_test] int tensor (all same value)
    """
    source_domains = [d for d in DOMAINS if d != held_out_domain]

    source_dataset = PACSDomainDataset(data_root, source_domains)
    test_dataset = PACSDomainDataset(data_root, [held_out_domain])

    source_domain_labels = torch.tensor(
        source_dataset.domain_indices, dtype=torch.long
    )
    test_domain_labels = torch.tensor(
        test_dataset.domain_indices, dtype=torch.long
    )

    source_loader = DataLoader(
        source_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    n_source_domains = len(source_domains)
    print(f"  Source: {len(source_dataset)} samples across {n_source_domains} domains "
          f"({source_domains})")
    print(f"  Test:   {len(test_dataset)} samples from '{held_out_domain}'")

    return source_loader, test_loader, source_domain_labels, test_domain_labels


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────


def get_probe_names(model_type: str) -> List[str]:
    """Return the list of probe layer names for a given model type."""
    if model_type == "erm":
        return ERM_PROBE_LAYERS
    return M2_PROBE_LAYERS


def get_layer_dims(model_type: str) -> Dict[str, int]:
    """Return the feature dimensions for each probe layer."""
    if model_type == "erm":
        return ERM_LAYER_DIMS
    return M2_LAYER_DIMS


def _make_gap_hook(
    name: str,
    hook_outputs: Dict[str, torch.Tensor],
    gap: nn.AdaptiveAvgPool2d,
) -> callable:
    """Create a forward hook that applies GAP + flatten and stores output.

    All backbone hooks use the same GAP operation for dimensionality fairness
    across layers — a probe on layer1 and layer4 are both compared over
    spatially pooled vectors.
    """

    def hook_fn(
        module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        hook_outputs[name] = gap(output).flatten(1).detach()

    return hook_fn


@torch.no_grad()
def extract_features(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract features from all probe points for the given model.

    Uses forward hooks for backbone layer intermediates and captures
    gap_backbone + extraction-block outputs directly from forward().
    Avoids any double forward pass.

    Args:
        model: loaded model in eval() mode
        model_type: 'erm', 'm2', or 'm2cl'
        loader: DataLoader yielding (image, class_label)
        device: torch device string

    Returns:
        features: dict mapping probe_name -> [N, D] CPU tensor
        class_labels: [N] CPU tensor
    """
    probe_names = get_probe_names(model_type)
    accumulated: Dict[str, List[torch.Tensor]] = {n: [] for n in probe_names}
    all_class_labels: List[torch.Tensor] = []

    model.eval()
    hooks: List[torch.utils.hooks.RemovableHook] = []
    hook_outputs: Dict[str, torch.Tensor] = {}
    gap = nn.AdaptiveAvgPool2d(1)

    if model_type != "erm":
        # Register hooks on backbone layers for GAP'd spatial features.
        # Hook on avgpool to capture gap_backbone without a second forward pass.
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name)
            h = layer.register_forward_hook(
                _make_gap_hook(layer_name, hook_outputs, gap)
            )
            hooks.append(h)

        def _avgpool_hook(
            module: nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            hook_outputs["gap_backbone"] = output.flatten(1).detach()

        hooks.append(model.avgpool.register_forward_hook(_avgpool_hook))

    for imgs, labels in loader:
        imgs = imgs.to(device)
        hook_outputs.clear()

        if model_type == "erm":
            # ERM: self.features is Sequential(*resnet_children[:-1])
            # which ends with avgpool; its output is [B, 512, 1, 1].
            feat = model.features(imgs).flatten(1)  # [B, 512]
            accumulated["backbone_final"].append(feat.cpu())

        else:
            # M2 / M2-CL: one forward pass; hooks capture all backbone layers
            # and avgpool simultaneously.
            _logits, reps = model(imgs)
            # reps is [e1, e2, e3, e4] because model was built with use_cl=True

            # Backbone layer features (from hooks)
            for feat_name, hook_key in [
                ("layer1_features", "layer1"),
                ("layer2_features", "layer2"),
                ("layer3_features", "layer3"),
                ("layer4_features", "layer4"),
            ]:
                accumulated[feat_name].append(hook_outputs[hook_key].cpu())

            # GAP backbone (from avgpool hook — no re-computation)
            accumulated["gap_backbone"].append(hook_outputs["gap_backbone"].cpu())

            # Extraction-block contrastive representations
            for i, ename in enumerate(
                ["e1_contrastive", "e2_contrastive", "e3_contrastive", "e4_contrastive"]
            ):
                accumulated[ename].append(reps[i].detach().cpu())

        all_class_labels.append(labels)

    for h in hooks:
        h.remove()

    features = {k: torch.cat(v, dim=0) for k, v in accumulated.items()}
    class_labels = torch.cat(all_class_labels, dim=0)
    return features, class_labels


# ─────────────────────────────────────────────────────────────────────────────
# Probe Architectures
# ─────────────────────────────────────────────────────────────────────────────


class LinearProbe(nn.Module):
    """Single linear layer — measures linear decodability only."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NonlinearProbe(nn.Module):
    """Two-layer MLP — measures nonlinear recoverability.

    Architecture: Linear(input_dim, 256) -> ReLU -> Linear(256, num_classes)
    Answers: is domain information linearly accessible, or only
    recoverable through a nonlinear function?
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# Probe Training and Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def train_and_evaluate_probe(
    probe_cls: Type[nn.Module],
    input_dim: int,
    num_classes: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
) -> float:
    """Train a probe classifier and return test accuracy.

    Args:
        probe_cls: LinearProbe or NonlinearProbe class
        input_dim: feature dimensionality
        num_classes: number of output classes
        train_features, train_labels: training set
        test_features, test_labels: test set
        epochs: training epochs
        lr: Adam learning rate

    Returns:
        Test accuracy in [0, 1].
    """
    probe = probe_cls(input_dim, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=PROBE_BATCH_SIZE, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for x, y in loader:
            logits = probe(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(test_features)
        preds = logits.argmax(dim=1)
        accuracy = (preds == test_labels).float().mean().item()

    return accuracy


def bootstrap_ci(
    probe_cls: Type[nn.Module],
    input_dim: int,
    num_classes: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    real_acc: float,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    epochs: int = PROBE_EPOCHS,
) -> Tuple[float, float]:
    """Bootstrap 95% CI on test accuracy.

    Resamples the test set with replacement to estimate variability.
    Does NOT retrain the probe — trains once on full train set, then
    evaluates on n_resamples bootstrapped test sets.

    Returns:
        (ci_lower, ci_upper) at 95% confidence level.
    """
    # Train a single probe on the full training set
    probe = probe_cls(input_dim, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=PROBE_BATCH_SIZE, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for x, y in loader:
            logits = probe(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe.eval()

    # Bootstrap the test set
    n_test = len(test_labels)
    boot_accs: List[float] = []
    with torch.no_grad():
        for _ in range(n_resamples):
            idx = torch.randint(0, n_test, (n_test,))
            boot_feats = test_features[idx]
            boot_labels = test_labels[idx]
            logits = probe(boot_feats)
            preds = logits.argmax(dim=1)
            boot_accs.append((preds == boot_labels).float().mean().item())

    boot_arr = np.array(boot_accs)
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))
    return ci_lower, ci_upper


def permutation_test(
    probe_cls: Type[nn.Module],
    input_dim: int,
    num_classes: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    real_acc: float,
    n_permutations: int,
    epochs: int = PERM_PROBE_EPOCHS,
) -> Tuple[float, float, float]:
    """Permutation test for statistical significance of probe accuracy.

    Shuffles train+test labels jointly and re-splits to build a null
    distribution under H0: labels are independent of features.

    Args:
        real_acc: observed accuracy from real (unshuffled) labels
        n_permutations: number of shuffle-and-retrain trials

    Returns:
        (p_value, null_mean, null_std)
        p_value = (count(null_acc >= real_acc) + 1) / (n_permutations + 1)
    """
    all_labels = torch.cat([train_labels, test_labels])
    n_train = len(train_labels)
    null_accs: List[float] = []

    for _ in range(n_permutations):
        perm = torch.randperm(len(all_labels))
        shuffled = all_labels[perm]
        perm_train_labels = shuffled[:n_train]
        perm_test_labels = shuffled[n_train:]

        acc = train_and_evaluate_probe(
            probe_cls,
            input_dim,
            num_classes,
            train_features,
            perm_train_labels,
            test_features,
            perm_test_labels,
            epochs=epochs,
        )
        null_accs.append(acc)

    null_arr = np.array(null_accs)
    p_value = (float(np.sum(null_arr >= real_acc)) + 1.0) / (n_permutations + 1.0)
    return p_value, float(null_arr.mean()), float(null_arr.std())


# ─────────────────────────────────────────────────────────────────────────────
# Full Probe Pipeline for One (model_type, held_out_domain) Pair
# ─────────────────────────────────────────────────────────────────────────────


def run_probe_pipeline(
    model: nn.Module,
    model_type: str,
    source_features: Dict[str, torch.Tensor],
    source_class_labels: torch.Tensor,
    source_domain_labels: torch.Tensor,
    test_features: Dict[str, torch.Tensor],
    test_class_labels: torch.Tensor,
    domain_split_ratio: float,
    n_permutations: int,
    seed: int,
) -> Dict[str, Dict]:
    """Run domain and class probes for all probe points of one checkpoint.

    Domain probe protocol:
        Source only, stratified 70/30 split by domain label.
        All source domains appear in both train and test splits.
        Held-out domain is NEVER used for domain probing.

    Class probe protocol:
        Train on all source features, test on held-out domain features.
        This measures cross-domain generalization.

    Args:
        source_features: {layer_name: [N_src, D]} features from source domains
        source_class_labels: [N_src] class labels
        source_domain_labels: [N_src] domain index labels (0..n_source_domains-1)
        test_features: {layer_name: [N_test, D]} features from held-out domain
        test_class_labels: [N_test] class labels
        domain_split_ratio: fraction of source for domain probe test split
        n_permutations: permutation test count
        seed: random seed for stratified split

    Returns:
        results dict with keys: 'domain_probes', 'class_probes'
        Each sub-dict maps layer_name -> metric_dict.
    """
    probe_names = get_probe_names(model_type)
    layer_dims = get_layer_dims(model_type)
    n_source_domains = int(source_domain_labels.max().item()) + 1

    # Stratified 70/30 split of source indices by domain label
    src_indices = np.arange(len(source_domain_labels))
    domain_train_idx, domain_test_idx = train_test_split(
        src_indices,
        test_size=domain_split_ratio,
        stratify=source_domain_labels.numpy(),
        random_state=seed,
    )

    # Sanity check: all source domains present in both splits
    train_domains = set(source_domain_labels[domain_train_idx].tolist())
    test_domains_set = set(source_domain_labels[domain_test_idx].tolist())
    expected_domains = set(range(n_source_domains))
    assert train_domains == expected_domains and test_domains_set == expected_domains, (
        f"Domain split is missing domains: "
        f"train_has={train_domains}, test_has={test_domains_set}, "
        f"expected={expected_domains}"
    )

    domain_probe_results: Dict[str, Dict] = {}
    class_probe_results: Dict[str, Dict] = {}

    for layer_name in probe_names:
        dim = layer_dims[layer_name]
        print(f"    [{layer_name}] dim={dim}", end="", flush=True)

        feat_src = source_features[layer_name]
        feat_tst = test_features[layer_name]

        # ── Domain probe ─────────────────────────────────────────────────
        d_train_feat = feat_src[domain_train_idx]
        d_train_labels = source_domain_labels[domain_train_idx]
        d_test_feat = feat_src[domain_test_idx]
        d_test_labels = source_domain_labels[domain_test_idx]

        linear_domain_acc = train_and_evaluate_probe(
            LinearProbe, dim, n_source_domains,
            d_train_feat, d_train_labels, d_test_feat, d_test_labels,
            epochs=PROBE_EPOCHS,
        )
        nonlinear_domain_acc = train_and_evaluate_probe(
            NonlinearProbe, dim, n_source_domains,
            d_train_feat, d_train_labels, d_test_feat, d_test_labels,
            epochs=PROBE_EPOCHS,
        )
        print(f"  domain_linear={linear_domain_acc*100:.1f}%", end="", flush=True)

        # Bootstrap CI on linear domain probe
        ci_lo, ci_hi = bootstrap_ci(
            LinearProbe, dim, n_source_domains,
            d_train_feat, d_train_labels, d_test_feat, d_test_labels,
            real_acc=linear_domain_acc,
            n_resamples=BOOTSTRAP_RESAMPLES,
            epochs=PROBE_EPOCHS,
        )

        # Permutation test on linear domain probe
        if n_permutations > 0:
            p_val, null_mean, null_std = permutation_test(
                LinearProbe, dim, n_source_domains,
                d_train_feat, d_train_labels, d_test_feat, d_test_labels,
                real_acc=linear_domain_acc,
                n_permutations=n_permutations,
                epochs=PERM_PROBE_EPOCHS,
            )
        else:
            p_val, null_mean, null_std = float("nan"), float("nan"), float("nan")

        print(f"  p={p_val:.3f}", flush=True)

        domain_probe_results[layer_name] = {
            "linear_acc": linear_domain_acc,
            "nonlinear_acc": nonlinear_domain_acc,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": p_val,
            "null_mean": null_mean,
            "null_std": null_std,
        }

        # ── Class probe ──────────────────────────────────────────────────
        linear_class_acc = train_and_evaluate_probe(
            LinearProbe, dim, NUM_CLASSES,
            feat_src, source_class_labels, feat_tst, test_class_labels,
            epochs=PROBE_EPOCHS,
        )
        nonlinear_class_acc = train_and_evaluate_probe(
            NonlinearProbe, dim, NUM_CLASSES,
            feat_src, source_class_labels, feat_tst, test_class_labels,
            epochs=PROBE_EPOCHS,
        )

        class_probe_results[layer_name] = {
            "linear_acc": linear_class_acc,
            "nonlinear_acc": nonlinear_class_acc,
        }

    return {
        "domain_probes": domain_probe_results,
        "class_probes": class_probe_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────


def _sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return "   n/a"
    if p < 0.001:
        return "<0.001***"
    if p < 0.01:
        return f"{p:.3f} **"
    if p < 0.05:
        return f"{p:.3f}  *"
    return f"{p:.3f}   "


def print_domain_probe_table(
    results: Dict[str, Dict],
    model_type: str,
    held_out_domain: str,
    domain_split_ratio: float,
) -> None:
    """Print domain probe results table for one (model_type, domain) run."""
    probe_names = get_probe_names(model_type)
    n_source = NUM_DOMAINS - 1
    chance = 100.0 / n_source

    pct = int((1 - domain_split_ratio) * 100)
    qct = int(domain_split_ratio * 100)
    print(f"\n=== DOMAIN PROBE RESULTS [{model_type.upper()} / held-out: {held_out_domain}]"
          f" (source-only, {pct}%/{qct}% split) ===")
    print(f"Random chance: {chance:.1f}%  |  n_source_domains={n_source}")
    print(f"{'Layer':<20s}  {'Linear':>10s}  {'95% CI':>16s}  {'Nonlinear':>10s}  {'p-value':>10s}")
    print("-" * 74)
    for layer_name in probe_names:
        if layer_name not in results["domain_probes"]:
            continue
        r = results["domain_probes"][layer_name]
        ci_str = f"[{r['ci_lower']*100:.1f},{r['ci_upper']*100:.1f}]"
        print(
            f"  {layer_name:<18s}  "
            f"{r['linear_acc']*100:8.1f}%  "
            f"{ci_str:>16s}  "
            f"{r['nonlinear_acc']*100:8.1f}%  "
            f"{_sig_stars(r['p_value']):>10s}"
        )


def print_class_probe_table(
    results: Dict[str, Dict],
    model_type: str,
    held_out_domain: str,
) -> None:
    """Print class probe results table for one (model_type, domain) run."""
    probe_names = get_probe_names(model_type)
    chance = 100.0 / NUM_CLASSES

    print(f"\n=== CLASS PROBE RESULTS [{model_type.upper()} / held-out: {held_out_domain}]"
          f" (source train -> held-out test) ===")
    print(f"Random chance: {chance:.1f}%")
    print(f"{'Layer':<20s}  {'Linear':>10s}  {'Nonlinear':>10s}")
    print("-" * 44)
    for layer_name in probe_names:
        if layer_name not in results["class_probes"]:
            continue
        r = results["class_probes"][layer_name]
        print(
            f"  {layer_name:<18s}  "
            f"{r['linear_acc']*100:8.1f}%  "
            f"{r['nonlinear_acc']*100:8.1f}%"
        )


def print_cross_model_ablation(all_results: Dict[str, Dict[str, Dict]]) -> None:
    """Print the cross-model ablation table: layer4_features vs e4_contrastive.

    This directly tests: does adding InfoNCE (m2 -> m2cl) increase the
    domain leakage gap between the backbone and the contrastive head?
    ERM has neither layer4_features nor e4_contrastive so is excluded.
    """
    print("\n")
    print("=" * 72)
    print("=== CROSS-MODEL ABLATION: layer4_features vs e4_contrastive ===")
    print("Tests: does InfoNCE (m2 -> m2cl) increase domain leakage in e4?")
    print("-" * 72)
    print(
        f"  {'Model':<8s}  {'Domain':<15s}  "
        f"{'layer4 (domain%)':<18s}  {'e4 (domain%)':<14s}  {'delta':>6s}"
    )
    print("  " + "-" * 66)

    rows = []
    for model_type in ["m2", "m2cl"]:
        if model_type not in all_results:
            continue
        for domain, domain_results in all_results[model_type].items():
            dp = domain_results.get("domain_probes", {})
            l4_acc = dp.get("layer4_features", {}).get("linear_acc")
            e4_acc = dp.get("e4_contrastive", {}).get("linear_acc")
            if l4_acc is None or e4_acc is None:
                continue
            delta = (e4_acc - l4_acc) * 100
            rows.append((model_type, domain, l4_acc, e4_acc, delta))

    for model_type, domain, l4, e4, delta in rows:
        print(
            f"  {model_type:<8s}  {domain:<15s}  "
            f"{l4*100:14.1f}%      "
            f"{e4*100:10.1f}%    "
            f"{delta:+6.1f}pp"
        )

    if not rows:
        print("  (No m2/m2cl results available for ablation table)")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: three-way domain leakage analysis."""
    args = get_args()
    set_seed(args.seed)

    # Limit CPU parallelism to avoid contention on shared machines
    n_threads = min(4, os.cpu_count() or 4)
    torch.set_num_threads(n_threads)

    print("=" * 72)
    print("  Phase 0: Three-Way Domain Leakage Analysis")
    print("  ERM vs M2 vs M2-CL on PACS")
    print("=" * 72)
    print(f"  Checkpoint dir:    {args.checkpoint_dir}")
    print(f"  Data root:         {args.data_root}")
    print(f"  Model types:       {args.model_types}")
    print(f"  Held-out domains:  {args.domains}")
    print(f"  Permutations:      {args.n_permutations}")
    print(f"  Bootstrap:         {BOOTSTRAP_RESAMPLES} resamples")
    print(f"  Domain split:      {int((1-args.domain_split_ratio)*100)}% train / "
          f"{int(args.domain_split_ratio*100)}% test (source only)")
    print(f"  Device:            {args.device}")
    print(f"  Threads:           {n_threads}")
    print()

    # all_results[model_type][held_out_domain] = probe pipeline results
    all_results: Dict[str, Dict[str, Dict]] = {}

    for model_type in args.model_types:
        all_results[model_type] = {}
        available = AVAILABLE_DOMAINS[model_type]

        # Filter to requested domains that are actually available
        domains_to_run = [d for d in args.domains if d in available]
        skipped = [d for d in args.domains if d not in available]
        if skipped:
            print(f"  NOTE: {model_type} has no checkpoint for: {skipped} — skipping")

        for held_out_domain in domains_to_run:
            ckpt_name = CHECKPOINT_NAMES[model_type].format(domain=held_out_domain)
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

            print(f"\n{'='*72}")
            print(f"  [{model_type.upper()}] held-out={held_out_domain}")
            print(f"  Checkpoint: {ckpt_path}")
            print(f"{'='*72}")

            if not os.path.exists(ckpt_path):
                print(f"  WARNING: checkpoint not found — skipping")
                continue

            # ── Load model ───────────────────────────────────────────────
            print("\n[1/4] Loading model...")
            model = load_model(ckpt_path, model_type, args.device)

            # ── Build data loaders ───────────────────────────────────────
            print("\n[2/4] Building data loaders...")
            source_loader, test_loader, source_domain_labels, test_domain_labels = (
                build_loaders(args.data_root, held_out_domain, args.batch_size)
            )

            # ── Extract features ─────────────────────────────────────────
            print("\n[3/4] Extracting features...")
            source_features, source_class_labels = extract_features(
                model, model_type, source_loader, args.device
            )
            test_features, test_class_labels = extract_features(
                model, model_type, test_loader, args.device
            )

            probe_names = get_probe_names(model_type)
            dims = get_layer_dims(model_type)
            print("  Source feature shapes:")
            for name in probe_names:
                print(f"    {name:<20s}: {list(source_features[name].shape)}")

            # ── Run probes ───────────────────────────────────────────────
            print(f"\n[4/4] Running probes ({len(probe_names)} layers × 2 types "
                  f"× domain+class + permutation test)...")

            run_results = run_probe_pipeline(
                model=model,
                model_type=model_type,
                source_features=source_features,
                source_class_labels=source_class_labels,
                source_domain_labels=source_domain_labels,
                test_features=test_features,
                test_class_labels=test_class_labels,
                domain_split_ratio=args.domain_split_ratio,
                n_permutations=args.n_permutations,
                seed=args.seed,
            )

            all_results[model_type][held_out_domain] = run_results

            # ── Per-run tables ───────────────────────────────────────────
            print_domain_probe_table(
                run_results, model_type, held_out_domain, args.domain_split_ratio
            )
            print_class_probe_table(run_results, model_type, held_out_domain)

            # Decision gate for this run
            proj_p = run_results["domain_probes"].get(
                "projection_head" if model_type == "erm" else "e4_contrastive", {}
            ).get("p_value", float("nan"))
            key_layer = "backbone_final" if model_type == "erm" else "e4_contrastive"
            key_p = run_results["domain_probes"].get(key_layer, {}).get("p_value", float("nan"))
            key_acc = run_results["domain_probes"].get(key_layer, {}).get("linear_acc", float("nan"))
            decision = "PROCEED" if (not np.isnan(key_p) and key_p < 0.05) else "RECONSIDER"
            print(f"\n  DECISION GATE [{model_type}/{held_out_domain}]:")
            print(f"    Key layer ({key_layer}) domain probe: {key_acc*100:.1f}%  "
                  f"p={key_p:.4f}  -> {decision}")

            # Free model memory before next checkpoint
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Cross-model ablation table ───────────────────────────────────────────
    print_cross_model_ablation(all_results)

    # ── Summary decision gate ────────────────────────────────────────────────
    print("\n=== DECISION GATE SUMMARY ===")
    all_decisions = []
    for model_type, domain_dict in all_results.items():
        for domain, res in domain_dict.items():
            key = "backbone_final" if model_type == "erm" else "e4_contrastive"
            dp = res.get("domain_probes", {}).get(key, {})
            p = dp.get("p_value", float("nan"))
            acc = dp.get("linear_acc", float("nan"))
            sig = not np.isnan(p) and p < 0.05
            all_decisions.append(sig)
            flag = "SIGNIFICANT *" if sig else "not significant"
            print(f"  {model_type:<6s} / {domain:<15s} ({key}):  "
                  f"{acc*100:.1f}%  p={p:.4f}  [{flag}]")

    if any(all_decisions):
        print("\n  OVERALL: PROCEED — at least one condition shows significant "
              "domain leakage.")
    else:
        print("\n  OVERALL: RECONSIDER — no condition shows significant domain leakage.")

    # ── Save results ─────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert float values for JSON serialization
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return None if np.isnan(obj) else round(obj, 6)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(round(obj, 6))
        return obj

    save_data = {
        "config": {
            "checkpoint_dir": args.checkpoint_dir,
            "data_root": args.data_root,
            "model_types": args.model_types,
            "domains": args.domains,
            "n_permutations": args.n_permutations,
            "domain_split_ratio": args.domain_split_ratio,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed": args.seed,
        },
        "results": _to_serializable(all_results),
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
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

PROBE_EPOCHS: int = 30
PERM_PROBE_EPOCHS: int = 5
PROBE_LR: float = 1e-3
PROBE_BATCH_SIZE: int = 64
BOOTSTRAP_RESAMPLES: int = 1000

PROBE_LAYERS: List[str] = [
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "backbone_final",
]
LAYER_DIMS: Dict[str, int] = {
    "layer1": 64,
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
    "backbone_final": 512,
}

# Model type labels for display and JSON keys
MODEL_LABELS: Dict[str, str] = {
    "imagenet": "ImageNet-pretrained (DEFAULT weights)",
    "random":   "Random-init        (weights=None)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random init vs ImageNet-pretrained domain probe control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: all 4 PACS domains, 200 permutations
  python probe_random_init_control.py \\
      --data_root "C:\\Users\\Madhusudan\\Downloads\\PACS" \\
      --domains photo art_painting cartoon sketch \\
      --n_permutations 200

  # Quick smoke test — 2 domains, 20 permutations (~15 min)
  python probe_random_init_control.py \\
      --data_root "C:\\Users\\Madhusudan\\Downloads\\PACS" \\
      --domains photo sketch \\
      --n_permutations 20
        """,
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
        default=["photo", "art_painting", "cartoon", "sketch"],
        choices=DOMAINS,
        help="Held-out domains to probe (default: all four)",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=200,
        help="Permutations for significance test (default: 200)",
    )
    parser.add_argument(
        "--domain_split_ratio",
        type=float,
        default=0.3,
        help="Fraction of source data held for domain probe test (default: 0.3)",
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
        default="random_init_control_results.json",
        help="Path to save JSON results",
    )
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────────────────────
# Model Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_model(init_type: str) -> nn.Module:
    """Build ResNet-18 with specified initialization. fc replaced with Identity.

    Neither model has ever seen PACS data. The only difference is whether
    weights come from ImageNet pretraining or random initialization.

    Args:
        init_type: 'imagenet' or 'random'

    Returns:
        ResNet-18 in eval() mode with fc=Identity.
    """
    if init_type == "imagenet":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif init_type == "random":
        # Explicitly set seed before random init so results are reproducible
        # across runs. This ensures the random init baseline is stable.
        torch.manual_seed(42)
        model = resnet18(weights=None)
    else:
        raise ValueError(f"init_type must be 'imagenet' or 'random', got '{init_type}'")

    model.fc = nn.Identity()
    model.eval()

    param_sum = sum(p.abs().sum().item() for p in model.parameters())
    print(f"  [{init_type}] param_sum={param_sum:.2f}")

    # Verify the two initializations are actually different
    if init_type == "random" and param_sum < 1.0:
        raise RuntimeError("Random-init model has near-zero weights — something is wrong.")

    return model

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PACSDomainDataset(Dataset):
    """Minimal PACS loader. Returns (image_tensor, class_label).
    Domain identity tracked separately via parallel domain_indices list.
    """

    def __init__(self, root: str, domain_names: List[str]) -> None:
        self.samples: List[Tuple[str, int]] = []
        self.domain_indices: List[int] = []

        first_path = os.path.join(root, domain_names[0])
        classes = sorted(
            c for c in os.listdir(first_path)
            if os.path.isdir(os.path.join(first_path, c))
        )
        class_to_idx = {c: i for i, c in enumerate(classes)}

        for d_idx, d_name in enumerate(domain_names):
            d_path = os.path.join(root, d_name)
            if not os.path.isdir(d_path):
                print(f"  WARNING: {d_path} not found — skipping")
                continue
            for cls in classes:
                cls_path = os.path.join(d_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(
                            (os.path.join(cls_path, fname), class_to_idx[cls])
                        )
                        self.domain_indices.append(d_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        return EVAL_TRANSFORM(Image.open(path).convert("RGB")), label


def build_loaders(
    data_root: str,
    held_out_domain: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, List[str]]:
    """Build source and held-out DataLoaders.

    Returns:
        source_loader, test_loader, source_domain_labels, source_domain_names
    """
    source_domains = [d for d in DOMAINS if d != held_out_domain]

    src_ds  = PACSDomainDataset(data_root, source_domains)
    test_ds = PACSDomainDataset(data_root, [held_out_domain])

    src_dom_labels = torch.tensor(src_ds.domain_indices, dtype=torch.long)

    src_loader  = DataLoader(src_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Source: {len(src_ds)} samples, {len(source_domains)} domains {source_domains}")
    print(f"  Test:   {len(test_ds)} samples from '{held_out_domain}'")

    return src_loader, test_loader, src_dom_labels, source_domains

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract 5 probe points from a ResNet-18 (fc=Identity).

    Probe points:
        layer1:          64-d  (GAP of layer1 output)
        layer2:          128-d (GAP of layer2 output)
        layer3:          256-d (GAP of layer3 output)
        layer4:          512-d (GAP of layer4 output)
        backbone_final:  512-d (model output, identical to GAP(layer4) here)

    All backbone hooks apply identical AdaptiveAvgPool2d(1) + flatten
    for dimensionality fairness across layers.
    """
    accum = {name: [] for name in PROBE_LAYERS}
    all_labels: List[torch.Tensor] = []

    gap = nn.AdaptiveAvgPool2d(1)
    hook_outputs: Dict[str, torch.Tensor] = {}
    hooks = []

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        def make_hook(name: str):
            def hook_fn(module: nn.Module,
                        inp: torch.Tensor,
                        out: torch.Tensor) -> None:
                hook_outputs[name] = gap(out).flatten(1).detach()
            return hook_fn
        hooks.append(
            getattr(model, layer_name).register_forward_hook(make_hook(layer_name))
        )

    model.eval()
    for imgs, labels in loader:
        hook_outputs.clear()
        out = model(imgs)          # [B, 512] since fc=Identity

        for name in ["layer1", "layer2", "layer3", "layer4"]:
            accum[name].append(hook_outputs[name].cpu())
        accum["backbone_final"].append(out.detach().cpu())
        all_labels.append(labels)

    for h in hooks:
        h.remove()

    features    = {k: torch.cat(v) for k, v in accum.items()}
    class_labels = torch.cat(all_labels)
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
    """Train a probe and return test accuracy in [0, 1]."""
    probe = probe_cls(input_dim, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(train_features, train_labels)
    loader  = DataLoader(dataset, batch_size=PROBE_BATCH_SIZE, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for x, y in loader:
            loss = criterion(probe(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(test_features).argmax(dim=1)
        return (preds == test_labels).float().mean().item()


def bootstrap_ci(
    probe_cls: Type[nn.Module],
    input_dim: int,
    num_classes: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    epochs: int = PROBE_EPOCHS,
) -> Tuple[float, float]:
    """Bootstrap 95% CI. Trains probe once, resamples test set n_resamples times.

    Returns:
        (ci_lower, ci_upper)
    """
    probe = probe_cls(input_dim, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=PROBE_BATCH_SIZE, shuffle=True,
    )
    probe.train()
    for _ in range(epochs):
        for x, y in loader:
            loss = criterion(probe(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    probe.eval()

    n = len(test_labels)
    boot_accs: List[float] = []
    with torch.no_grad():
        for _ in range(n_resamples):
            idx = torch.randint(0, n, (n,))
            preds = probe(test_features[idx]).argmax(dim=1)
            boot_accs.append((preds == test_labels[idx]).float().mean().item())

    arr = np.array(boot_accs)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


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
    """Permutation test. Shuffles train+test labels jointly.

    Returns:
        (p_value, null_mean, null_std)
        p_value = (count(null >= real_acc) + 1) / (n_permutations + 1)
    """
    all_labels = torch.cat([train_labels, test_labels])
    n_train    = len(train_labels)
    null_accs: List[float] = []

    for _ in range(n_permutations):
        perm     = torch.randperm(len(all_labels))
        shuffled = all_labels[perm]
        acc = train_and_evaluate_probe(
            probe_cls, input_dim, num_classes,
            train_features, shuffled[:n_train],
            test_features,  shuffled[n_train:],
            epochs=epochs,
        )
        null_accs.append(acc)

    arr     = np.array(null_accs)
    p_value = (float(np.sum(arr >= real_acc)) + 1.0) / (n_permutations + 1.0)
    return p_value, float(arr.mean()), float(arr.std())

# ─────────────────────────────────────────────────────────────────────────────
# Probe Pipeline for One (init_type, held_out_domain) Pair
# ─────────────────────────────────────────────────────────────────────────────

def run_probe_pipeline(
    src_features: Dict[str, torch.Tensor],
    src_class_labels: torch.Tensor,
    src_domain_labels: torch.Tensor,
    tst_features: Dict[str, torch.Tensor],
    tst_class_labels: torch.Tensor,
    domain_split_ratio: float,
    n_permutations: int,
    n_source_domains: int,
    seed: int,
) -> Dict[str, Dict]:
    """Run domain and class probes for all 5 probe layers.

    Domain probe: source-only 70/30 stratified split.
                  Held-out domain NEVER used for domain probing.
    Class probe:  source train -> held-out test (DG evaluation).

    Returns:
        {'domain_probes': {layer: metrics}, 'class_probes': {layer: metrics}}
    """
    indices = np.arange(len(src_domain_labels))
    tr_idx, te_idx = train_test_split(
        indices,
        test_size=domain_split_ratio,
        stratify=src_domain_labels.numpy(),
        random_state=seed,
    )

    # Verify split contains all source domains in both halves
    assert (set(src_domain_labels[tr_idx].tolist()) ==
            set(src_domain_labels[te_idx].tolist()) ==
            set(range(n_source_domains))), \
        "Stratified split is missing one or more source domains."

    domain_results: Dict[str, Dict] = {}
    class_results:  Dict[str, Dict] = {}

    for layer in PROBE_LAYERS:
        dim = LAYER_DIMS[layer]
        print(f"    [{layer}] dim={dim}", end="", flush=True)

        # ── Domain probe (source only) ────────────────────────────────────
        d_tr_f = src_features[layer][tr_idx]
        d_tr_l = src_domain_labels[tr_idx]
        d_te_f = src_features[layer][te_idx]
        d_te_l = src_domain_labels[te_idx]

        lin_d  = train_and_evaluate_probe(
            LinearProbe, dim, n_source_domains,
            d_tr_f, d_tr_l, d_te_f, d_te_l,
        )
        nlin_d = train_and_evaluate_probe(
            NonlinearProbe, dim, n_source_domains,
            d_tr_f, d_tr_l, d_te_f, d_te_l,
        )
        print(f"  domain_linear={lin_d*100:.1f}%", end="", flush=True)

        ci_lo, ci_hi = bootstrap_ci(
            LinearProbe, dim, n_source_domains,
            d_tr_f, d_tr_l, d_te_f, d_te_l,
        )

        p_val, null_m, null_s = permutation_test(
            LinearProbe, dim, n_source_domains,
            d_tr_f, d_tr_l, d_te_f, d_te_l,
            real_acc=lin_d,
            n_permutations=n_permutations,
        )
        print(f"  p={p_val:.4f}", flush=True)

        domain_results[layer] = {
            "linear_acc":    lin_d,
            "nonlinear_acc": nlin_d,
            "ci_lower":      ci_lo,
            "ci_upper":      ci_hi,
            "p_value":       p_val,
            "null_mean":     null_m,
            "null_std":      null_s,
        }

        # ── Class probe (source -> held-out) ─────────────────────────────
        lin_c  = train_and_evaluate_probe(
            LinearProbe, dim, NUM_CLASSES,
            src_features[layer], src_class_labels,
            tst_features[layer], tst_class_labels,
        )
        nlin_c = train_and_evaluate_probe(
            NonlinearProbe, dim, NUM_CLASSES,
            src_features[layer], src_class_labels,
            tst_features[layer], tst_class_labels,
        )
        class_results[layer] = {"linear_acc": lin_c, "nonlinear_acc": nlin_c}

    return {"domain_probes": domain_results, "class_probes": class_results}

# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _sig_stars(p: float) -> str:
    if np.isnan(p):        return "   n/a"
    if p < 0.001:          return "<0.001***"
    if p < 0.01:           return f"{p:.4f} **"
    if p < 0.05:           return f"{p:.4f}  *"
    return                        f"{p:.4f}   "


def print_domain_table(
    results: Dict[str, Dict],
    init_type: str,
    held_out: str,
    source_domains: List[str],
    domain_split_ratio: float,
) -> None:
    n_src  = len(source_domains)
    chance = 100.0 / n_src
    pct    = int((1 - domain_split_ratio) * 100)
    qct    = int(domain_split_ratio * 100)
    label  = MODEL_LABELS[init_type]

    print(f"\n=== DOMAIN PROBE [{label} / held-out: {held_out}]"
          f" (source-only, {pct}%/{qct}% split) ===")
    print(f"Random chance: {chance:.1f}%  |  source: {source_domains}")
    print(f"{'Layer':<18s}  {'Linear':>9s}  {'95% CI':>16s}  "
          f"{'Nonlinear':>10s}  {'p-value':>10s}")
    print("-" * 70)
    for layer in PROBE_LAYERS:
        r   = results["domain_probes"][layer]
        ci  = f"[{r['ci_lower']*100:.1f},{r['ci_upper']*100:.1f}]"
        print(
            f"  {layer:<16s}  {r['linear_acc']*100:8.1f}%  "
            f"{ci:>16s}  {r['nonlinear_acc']*100:9.1f}%  "
            f"{_sig_stars(r['p_value']):>10s}"
        )


def print_class_table(
    results: Dict[str, Dict],
    init_type: str,
    held_out: str,
) -> None:
    label = MODEL_LABELS[init_type]
    print(f"\n=== CLASS PROBE [{label} / held-out: {held_out}]"
          f" (source train -> held-out test) ===")
    print(f"Random chance: {100/NUM_CLASSES:.1f}%")
    print(f"{'Layer':<18s}  {'Linear':>9s}  {'Nonlinear':>10s}")
    print("-" * 42)
    for layer in PROBE_LAYERS:
        r = results["class_probes"][layer]
        print(f"  {layer:<16s}  {r['linear_acc']*100:8.1f}%  "
              f"{r['nonlinear_acc']*100:9.1f}%")


def print_side_by_side(
    all_results: Dict[str, Dict[str, Dict]],
    held_out: str,
) -> None:
    """Print the critical comparison table for one held-out domain."""
    print(f"\n{'='*80}")
    print(f"=== SIDE-BY-SIDE COMPARISON / held-out: {held_out} ===")
    print(f"{'Layer':<18s}  {'Random-init':>12s}  {'ImageNet':>12s}  "
          f"{'Delta (IN-RI)':>14s}  Interpretation")
    print("-" * 80)

    chance = 100.0 / (len(DOMAINS) - 1)

    for layer in PROBE_LAYERS:
        ri  = all_results["random"].get(held_out, {})
        img = all_results["imagenet"].get(held_out, {})
        if not ri or not img:
            continue

        ri_acc  = ri["domain_probes"][layer]["linear_acc"]  * 100
        img_acc = img["domain_probes"][layer]["linear_acc"] * 100
        delta   = img_acc - ri_acc

        # Interpretation logic
        if ri_acc < chance + 10 and img_acc > chance + 40:
            interp = "← Leakage is PRETRAINING-SPECIFIC"
        elif ri_acc > chance + 30 and delta < 5:
            interp = "← Leakage is ARCHITECTURAL"
        elif ri_acc > chance + 15 and delta > 10:
            interp = "← Both architecture AND pretraining"
        else:
            interp = ""

        print(f"  {layer:<16s}  {ri_acc:10.1f}%  {img_acc:10.1f}%  "
              f"{delta:+12.1f}pp  {interp}")
    print(f"{'='*80}")


def print_full_summary(
    all_results: Dict[str, Dict[str, Dict]],
    domains: List[str],
) -> None:
    """Print the paper-ready summary table across all domains."""
    print("\n\n" + "=" * 80)
    print("=== FINAL SUMMARY: Random-init vs ImageNet-pretrained (backbone_final) ===")
    print("=" * 80)
    print(f"{'Domain':<15s}  {'Random-init':>12s}  {'ImageNet':>10s}  "
          f"{'Delta':>8s}  {'Verdict'}")
    print("-" * 70)

    chance = 100.0 / (len(DOMAINS) - 1)

    for held_out in domains:
        ri  = all_results["random"].get(held_out, {})
        img = all_results["imagenet"].get(held_out, {})
        if not ri or not img:
            continue

        ri_acc  = ri["domain_probes"]["backbone_final"]["linear_acc"]  * 100
        img_acc = img["domain_probes"]["backbone_final"]["linear_acc"] * 100
        delta   = img_acc - ri_acc

        if ri_acc < chance + 10 and img_acc > chance + 40:
            verdict = "PRETRAINING-INHERITED"
        elif ri_acc > chance + 30 and delta < 5:
            verdict = "ARCHITECTURE-DRIVEN"
        elif delta > 10:
            verdict = "BOTH CONTRIBUTE"
        else:
            verdict = "UNCLEAR — inspect manually"

        print(f"  {held_out:<13s}  {ri_acc:10.1f}%  {img_acc:8.1f}%  "
              f"{delta:+6.1f}pp  {verdict}")

    print("\n  Reference (from prior three-way ablation):")
    print("  ERM/photo backbone_final:        87.8%")
    print("  M2/photo  layer4_features:       88.3%")
    print("  M2-CL/photo layer4_features:     88.5%")
    print("  ImageNet-only/photo:             88.4%")
    print()
    print("  INTERPRETATION GUIDE:")
    print("  Random-init ~ chance (33%) AND ImageNet >> 60%")
    print("    → Domain leakage is SPECIFICALLY inherited from ImageNet")
    print("    → CNN architecture alone does not cause it")
    print("    → This is the strongest possible version of the finding")
    print()
    print("  Random-init >> chance (e.g. 60-70%) AND ImageNet ~ Random-init")
    print("    → Domain leakage is inherent to CNN ARCHITECTURE")
    print("    → ImageNet pretraining does not meaningfully increase it")
    print("    → Different paper: architecture-driven, not pretraining-driven")
    print()
    print("  Random-init moderately above chance AND ImageNet >> Random-init")
    print("    → Both CNN architecture AND ImageNet pretraining contribute")
    print("    → Pretraining amplifies an architectural tendency")
    print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    set_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    print("=" * 80)
    print("  Random-Init vs ImageNet-Pretrained Domain Probe Control")
    print("  Neither model has seen any PACS data.")
    print("  Purpose: isolate whether leakage comes from pretraining or architecture.")
    print("=" * 80)
    print(f"  Data root:       {args.data_root}")
    print(f"  Domains:         {args.domains}")
    print(f"  Permutations:    {args.n_permutations}")
    print(f"  Bootstrap:       {BOOTSTRAP_RESAMPLES} resamples")
    print(f"  Domain split:    {int((1-args.domain_split_ratio)*100)}%/"
          f"{int(args.domain_split_ratio*100)}% (source only, stratified)")
    print()

    # Build both models upfront to verify both load correctly
    print("[Setup] Building both models...")
    models = {
        "random":   build_model("random"),
        "imagenet": build_model("imagenet"),
    }

    # Verify the two models have genuinely different weights
    ri_w   = list(models["random"].parameters())[0].data.flatten()[:5]
    img_w  = list(models["imagenet"].parameters())[0].data.flatten()[:5]
    assert not torch.allclose(ri_w, img_w), \
        "Random-init and ImageNet models have identical weights — something is wrong."
    print("  Weight divergence confirmed: models are different. ✓\n")

    # all_results[init_type][held_out_domain] = probe results
    all_results: Dict[str, Dict[str, Dict]] = {
        "random":   {},
        "imagenet": {},
    }

    for held_out in args.domains:
        print(f"\n{'='*80}")
        print(f"  Held-out domain: {held_out}")
        print(f"{'='*80}")

        # Build data loaders once per held-out domain (shared across both models)
        print("\n[Data] Building loaders...")
        src_loader, tst_loader, src_dom_labels, source_domains = build_loaders(
            args.data_root, held_out, args.batch_size
        )
        n_source = len(source_domains)

        for init_type in ["random", "imagenet"]:
            label = MODEL_LABELS[init_type]
            print(f"\n--- {label} ---")

            model = models[init_type]

            print(f"  Extracting source features ({len(src_dom_labels)} samples)...")
            src_feats, src_cls = extract_features(model, src_loader)

            print(f"  Extracting test features...")
            tst_feats, tst_cls = extract_features(model, tst_loader)

            print(f"  Running probes ({len(PROBE_LAYERS)} layers, "
                  f"{args.n_permutations} permutations each)...")

            run_results = run_probe_pipeline(
                src_feats, src_cls, src_dom_labels,
                tst_feats, tst_cls,
                domain_split_ratio=args.domain_split_ratio,
                n_permutations=args.n_permutations,
                n_source_domains=n_source,
                seed=args.seed,
            )

            all_results[init_type][held_out] = run_results

            print_domain_table(
                run_results, init_type, held_out,
                source_domains, args.domain_split_ratio,
            )
            print_class_table(run_results, init_type, held_out)

        # Side-by-side comparison immediately after each domain finishes
        print_side_by_side(all_results, held_out)

    # Full summary across all domains
    print_full_summary(all_results, args.domains)

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return None if np.isnan(obj) else round(obj, 6)
        if isinstance(obj, np.floating):
            return None if np.isnan(float(obj)) else round(float(obj), 6)
        return obj

    save_data = {
        "config": {
            "data_root":          args.data_root,
            "domains":            args.domains,
            "n_permutations":     args.n_permutations,
            "domain_split_ratio": args.domain_split_ratio,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed":               args.seed,
            "note": (
                "random = ResNet-18(weights=None), "
                "imagenet = ResNet-18(weights=ResNet18_Weights.DEFAULT). "
                "Neither has seen any PACS data."
            ),
        },
        "results": _serialize(all_results),
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
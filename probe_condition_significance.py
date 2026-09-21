"""Paired significance test: does DG training (ERM / M2 / M2-CL) change
domain-probe accuracy relative to the pure ImageNet-pretrained control?

This is the missing statistical test behind the paper's central claim
("PACS-specific DG training contributes ~0pp"). Previously that claim
was supported only by eyeballing that four single point estimates
(87.8%, 88.3%, 88.5%, 88.3-88.4%) were within ~0.5pp of each other.
That is not a significance test, and with only one checkpoint per
condition (no seeds — retraining ERM/M2/M2-CL 5x each is not practical
compute), a multi-seed approach like the random-init control isn't
available here.

What IS available: the domain-probe TEST SET is the same underlying
PACS images regardless of which backbone extracted the features (same
held-out domain, same stratified split, same seed). So we can do a
PAIRED bootstrap: resample the same test indices for both conditions
simultaneously and ask whether the accuracy gap between them exceeds
what identical-test-set sampling noise alone would produce.

Feature used for comparison: a 512-d GAP(layer4)-equivalent vector for
every condition (ImageNet-only backbone_final, ERM backbone_final,
M2/M2-CL gap_backbone) — the one feature that is directly comparable
in both dimensionality and semantic meaning across all four models.

Usage:
    python probe_condition_significance.py \\
        --checkpoint_dir "C:\\path\\to\\m2cl_checkpoints" \\
        --data_root "C:\\Users\\Madhusudan\\Downloads\\PACS" \\
        --domains photo art_painting cartoon \\
        --n_bootstrap 2000
"""

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

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

DOMAINS: List[str] = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES: int = 7
PROBE_EPOCHS: int = 30
PROBE_LR: float = 1e-3
PROBE_BATCH_SIZE: int = 64

AVAILABLE_DOMAINS: Dict[str, List[str]] = {
    "erm":  ["art_painting", "cartoon", "photo", "sketch"],
    "m2":   ["art_painting", "cartoon", "photo", "sketch"],
    "m2cl": ["art_painting", "cartoon", "photo"],   # no sketch checkpoint
}
CHECKPOINT_NAMES: Dict[str, str] = {
    "erm":  "erm_{domain}.pt",
    "m2":   "m2_{domain}.pt",
    "m2cl": "m2cl_{domain}.pt",
}

# ─────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--domains", type=str, nargs="+",
                    default=["photo", "art_painting", "cartoon"],
                    choices=DOMAINS)
    p.add_argument("--split_seed", type=int, default=42,
                    help="Seed for the source train/test domain-probe split. "
                         "MUST match what you used for the ImageNet-only run "
                         "if you want numbers to line up across scripts.")
    p.add_argument("--probe_seed", type=int, default=0,
                    help="Seed for probe weight init/training (kept fixed "
                         "across conditions for a fair comparison).")
    p.add_argument("--domain_split_ratio", type=float, default=0.3)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", type=str,
                    default="condition_significance_results.json")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ─────────────────────────────────────────────────────────────────────────
# Model architectures (verbatim from probe_existing_m2cl.py so checkpoints
# load with identical keys)
# ─────────────────────────────────────────────────────────────────────────

class ERM(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(bb.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1)), None


class ConcentrationPipeline(nn.Module):
    def __init__(self, in_channels, mid_channels, pool_size, mlp_dim=128):
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

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.mlp(self.pool(self.drop(h)))


class ExtractionBlock(nn.Module):
    def __init__(self, in_channels, r=4, pool_sizes=(4, 2), mlp_dim=128):
        super().__init__()
        mid = max(in_channels // r, 8)
        self.pipes = nn.ModuleList(
            [ConcentrationPipeline(in_channels, mid, ps, mlp_dim) for ps in pool_sizes]
        )
        self.out_dim = mlp_dim * len(pool_sizes)

    def forward(self, x):
        return torch.cat([p(x) for p in self.pipes], dim=1)


class M2(nn.Module):
    def __init__(self, num_classes: int, use_cl: bool = False, mlp_dim: int = 128) -> None:
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
        total = self.eb1.out_dim + self.eb2.out_dim + self.eb3.out_dim + self.eb4.out_dim + 512
        self.classifier = nn.Linear(total, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        f1 = self.layer1(x); f2 = self.layer2(f1)
        f3 = self.layer3(f2); f4 = self.layer4(f3)
        e1, e2 = self.eb1(f1), self.eb2(f2)
        e3, e4 = self.eb3(f3), self.eb4(f4)
        gap = self.avgpool(f4).flatten(1)
        z = torch.cat([e1, e2, e3, e4, gap], dim=1)
        reps = [e1, e2, e3, e4] if self.use_cl else None
        return self.classifier(z), reps


def load_checkpoint_model(checkpoint_path: str, model_type: str) -> nn.Module:
    if model_type == "erm":
        model: nn.Module = ERM(num_classes=NUM_CLASSES)
    else:
        model = M2(num_classes=NUM_CLASSES, use_cl=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, f"key mismatch: {missing}, {unexpected}"
    model.eval()
    return model


def build_imagenet_only_model() -> nn.Module:
    """The control condition: ImageNet weights, zero PACS exposure."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────
# Data (filenames explicitly sorted — required for valid paired comparison
# across conditions/scripts; os.listdir() order is NOT guaranteed)
# ─────────────────────────────────────────────────────────────────────────

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PACSDomainDataset(Dataset):
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
                continue
            for cls in classes:
                cls_path = os.path.join(d_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                fnames = sorted(
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                for fname in fnames:
                    self.samples.append((os.path.join(cls_path, fname), class_to_idx[cls]))
                    self.domain_indices.append(d_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return EVAL_TRANSFORM(Image.open(path).convert("RGB")), label


def build_source_loader(data_root: str, held_out_domain: str, batch_size: int):
    source_domains = [d for d in DOMAINS if d != held_out_domain]
    ds = PACSDomainDataset(data_root, source_domains)
    domain_labels = torch.tensor(ds.domain_indices, dtype=torch.long)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, domain_labels, source_domains

# ─────────────────────────────────────────────────────────────────────────
# Feature extraction — one 512-d "backbone-level" vector per condition
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_backbone512(model: nn.Module, model_type: str, loader: DataLoader) -> torch.Tensor:
    """Returns [N, 512] features that are semantically comparable across
    ImageNet-only / ERM / M2 / M2-CL:
        imagenet -> GAP(layer4) via fc=Identity forward
        erm      -> model.features(x) (ends in avgpool) flattened
        m2/m2cl  -> avgpool(layer4) i.e. gap_backbone, via hook
    """
    feats: List[torch.Tensor] = []
    if model_type == "imagenet":
        for imgs, _ in loader:
            feats.append(model(imgs).cpu())
    elif model_type == "erm":
        for imgs, _ in loader:
            feats.append(model.features(imgs).flatten(1).cpu())
    else:  # m2 / m2cl
        hook_out: Dict[str, torch.Tensor] = {}
        h = model.avgpool.register_forward_hook(
            lambda m, i, o: hook_out.__setitem__("gap", o.flatten(1).detach())
        )
        for imgs, _ in loader:
            hook_out.clear()
            model(imgs)
            feats.append(hook_out["gap"].cpu())
        h.remove()
    return torch.cat(feats, dim=0)

# ─────────────────────────────────────────────────────────────────────────
# Probe
# ─────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(input_dim, num_classes, train_features, train_labels, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    probe = LinearProbe(input_dim, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_features, train_labels),
                         batch_size=PROBE_BATCH_SIZE, shuffle=True)
    probe.train()
    for _ in range(PROBE_EPOCHS):
        for x, y in loader:
            loss = criterion(probe(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    probe.eval()
    return probe


@torch.no_grad()
def correctness_vector(probe: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    preds = probe(features).argmax(dim=1)
    return (preds == labels).numpy().astype(np.float64)

# ─────────────────────────────────────────────────────────────────────────
# Paired bootstrap significance test
# ─────────────────────────────────────────────────────────────────────────

def paired_bootstrap_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    """Test H0: condition A and condition B have equal domain-probe accuracy
    on this test set. correct_a/correct_b are 0/1 arrays over the SAME
    ordered test indices (paired). Resamples indices WITH the same draw
    applied to both arrays each time, so the pairing is preserved.

    Returns dict with acc_a, acc_b, mean_diff, ci_lower, ci_upper, p_value.
    p_value is two-sided: 2 * min(P(diff<=0), P(diff>=0)) under the
    bootstrap distribution of (acc_a - acc_b), clipped to [0,1].
    """
    assert len(correct_a) == len(correct_b)
    rng = np.random.default_rng(seed)
    n = len(correct_a)

    acc_a = correct_a.mean()
    acc_b = correct_b.mean()

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_le = float(np.mean(diffs <= 0))
    p_ge = float(np.mean(diffs >= 0))
    p_value = min(1.0, 2.0 * min(p_le, p_ge))

    return {
        "acc_a": float(acc_a),
        "acc_b": float(acc_b),
        "observed_diff_pp": float((acc_a - acc_b) * 100),
        "ci_lower_pp": float(ci_lo * 100),
        "ci_upper_pp": float(ci_hi * 100),
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    print("=" * 88)
    print("  Paired Significance Test: DG Training vs ImageNet-Only Control")
    print("  H0 for each condition: domain-probe accuracy == ImageNet-only accuracy")
    print("=" * 88)

    all_results: Dict[str, Dict] = {}

    for held_out in args.domains:
        print(f"\n{'='*88}\n  Held-out domain: {held_out}\n{'='*88}")

        loader, domain_labels, source_domains = build_source_loader(
            args.data_root, held_out, args.batch_size
        )
        n_source = len(source_domains)

        # Fixed train/test split — SAME indices used for every condition
        indices = np.arange(len(domain_labels))
        tr_idx, te_idx = train_test_split(
            indices, test_size=args.domain_split_ratio,
            stratify=domain_labels.numpy(), random_state=args.split_seed,
        )
        te_domain_labels = domain_labels[te_idx]
        tr_domain_labels = domain_labels[tr_idx]

        conditions_available = ["imagenet"]
        for mt in ["erm", "m2", "m2cl"]:
            if held_out in AVAILABLE_DOMAINS[mt]:
                ckpt = os.path.join(args.checkpoint_dir,
                                     CHECKPOINT_NAMES[mt].format(domain=held_out))
                if os.path.exists(ckpt):
                    conditions_available.append(mt)
                else:
                    print(f"  [Skip] {mt}: checkpoint not found at {ckpt}")

        correctness: Dict[str, np.ndarray] = {}
        acc_summary: Dict[str, float] = {}

        for cond in conditions_available:
            print(f"\n  --- {cond} ---")
            if cond == "imagenet":
                model = build_imagenet_only_model()
            else:
                ckpt = os.path.join(args.checkpoint_dir,
                                     CHECKPOINT_NAMES[cond].format(domain=held_out))
                model = load_checkpoint_model(ckpt, cond)

            feats = extract_backbone512(model, cond, loader)
            tr_feats, te_feats = feats[tr_idx], feats[te_idx]

            probe = train_probe(512, n_source, tr_feats, tr_domain_labels, seed=args.probe_seed)
            corr = correctness_vector(probe, te_feats, te_domain_labels)
            correctness[cond] = corr
            acc_summary[cond] = float(corr.mean() * 100)
            print(f"    domain-probe accuracy: {acc_summary[cond]:.1f}%")

            del model

        # ── Paired tests: each DG condition vs imagenet-only ───────────────
        domain_result: Dict[str, Dict] = {"accuracies_pct": acc_summary, "tests": {}}
        for cond in conditions_available:
            if cond == "imagenet":
                continue
            test_result = paired_bootstrap_test(
                correctness[cond], correctness["imagenet"],
                n_bootstrap=args.n_bootstrap, seed=args.split_seed,
            )
            domain_result["tests"][cond] = test_result
            sig = "SIGNIFICANT DIFFERENCE" if test_result["significant_at_0.05"] else "no significant difference"
            print(f"\n  {cond} vs imagenet-only: "
                  f"diff={test_result['observed_diff_pp']:+.2f}pp  "
                  f"95% CI=[{test_result['ci_lower_pp']:+.2f},{test_result['ci_upper_pp']:+.2f}]pp  "
                  f"p={test_result['p_value']:.4f}  -> {sig}")

        all_results[held_out] = domain_result

    # ── Overall summary ─────────────────────────────────────────────────
    print("\n\n" + "=" * 88)
    print("=== SUMMARY: Is DG training statistically distinguishable from ImageNet-only? ===")
    print("=" * 88)
    any_significant = False
    for held_out, res in all_results.items():
        for cond, t in res["tests"].items():
            any_significant = any_significant or t["significant_at_0.05"]
            flag = "DIFFERS from ImageNet-only" if t["significant_at_0.05"] else "indistinguishable from ImageNet-only"
            print(f"  {held_out:<15s} {cond:<6s}: {t['observed_diff_pp']:+6.2f}pp  "
                  f"p={t['p_value']:.4f}  [{flag}]")
    print()
    if any_significant:
        print("  At least one DG condition differs significantly from ImageNet-only.")
        print("  The 'DG training contributes ~0pp' claim needs qualification —")
        print("  report per-condition results rather than a blanket statement.")
    else:
        print("  No DG condition differs significantly from ImageNet-only, at any")
        print("  tested domain. This is now a properly tested claim, not an eyeball")
        print("  comparison — safe to state in the paper with these p-values cited.")
    print("=" * 88)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
"""Extension of probe_condition_significance.py: adds the corrected/6-tap
M2-CL checkpoints (alpha=0.003, tau=1.0, arch_taps=6, 3 seeds, all 4 PACS
domains) as a new condition, probed against the SAME ImageNet-only control
using the SAME paired-bootstrap methodology as the original script.

This produces Table VIII as a ROBUSTNESS/EXTENSION analysis, not a
replacement for Table III. The old m2cl_{domain}.pt results (art/cartoon/
photo only, single seed, 4-tap, under-scaled contrastive loss) are left
untouched -- this script only adds new rows.

Key differences from probe_condition_significance.py:
    1. New M2_6TAP class (adds eb1a/eb2a extraction blocks matching the
       training notebook's arch_taps=6 branch). gap_backbone extraction is
       UNCHANGED and architecturally identical regardless of tap count --
       the extra taps are parallel branches off layer1/layer2 that never
       touch the layer4 -> avgpool path.
    2. 3 seeds per domain instead of 1. Per review, these are NOT treated
       as independent hypotheses for BH correction -- they are replicate
       measurements of the same condition. The primary output is therefore
       DESCRIPTIVE (mean +/- std across seeds), with the three per-seed
       paired-bootstrap p-values reported alongside as supplementary
       detail, not combined into a cross-seed correction.
    3. Checkpoint loading unwraps the training notebook's checkpoint dict
       format ({"config":..., "model_state":..., ...}) automatically, since
       these checkpoints were saved by run_experiment() in the Colab
       notebook, not as raw state_dicts like the original erm_*.pt /
       m2cl_*.pt files.
    4. Sketch is now included, since 6-tap M2-CL is the first M2-CL family
       checkpoint ever trained with Sketch held out.

BEFORE RUNNING: download the 12 best.pt files from
    /content/drive/MyDrive/m2cl_improvement_checkpoints/<run_id>/best.pt
to a local directory and rename them to the pattern expected below
(m2cl_6tap_{domain}_seed{seed}.pt), OR pass --checkpoint_pattern to match
whatever naming you used. The run_id -> config mapping is in
m2cl_final_results.json / m2cl_locked_config.json on Drive if you need to
confirm which folder corresponds to which domain/seed before renaming.

Usage:
    python probe_m2cl_6tap_significance.py \\
        --checkpoint_dir "C:\\path\\to\\m2cl_6tap_checkpoints" \\
        --data_root "C:\\Users\\Madhusudan\\Downloads\\PACS" \\
        --domains art_painting cartoon photo sketch \\
        --seeds 0 1 2 \\
        --split_seed 42 \\
        --n_bootstrap 2000 \\
        --output condition_significance_results_table_viii.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------------------
# Constants (identical to probe_condition_significance.py)
# ---------------------------------------------------------------------------

DOMAINS: List[str] = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES: int = 7
PROBE_EPOCHS: int = 30
PROBE_LR: float = 1e-3
PROBE_BATCH_SIZE: int = 64


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--domains", type=str, nargs="+", default=DOMAINS, choices=DOMAINS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--checkpoint_pattern", type=str,
                    default="m2cl6tap_{domain}_seed{seed}.pt",
                    help="Filename pattern relative to checkpoint_dir. "
                         "Must contain {domain} and {seed}.")
    p.add_argument("--split_seed", type=int, default=42,
                    help="MUST match the split_seed used in "
                         "probe_condition_significance.py for the same "
                         "domain, so the paired test set is identical.")
    p.add_argument("--probe_seed", type=int, default=0)
    p.add_argument("--domain_split_ratio", type=float, default=0.3)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", type=str,
                    default="condition_significance_results_table_viii.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------

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


class M2_6TAP(nn.Module):
    """Matches the arch_taps=6 branch of the training notebook's M2 class
    exactly (adds eb1a/eb2a). Key names must match the saved checkpoint's
    model_state for strict loading to succeed."""

    def __init__(self, num_classes: int, use_cl: bool = True, mlp_dim: int = 128) -> None:
        super().__init__()
        self.use_cl = use_cl
        self.arch_taps = 6
        bb = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1, self.layer2 = bb.layer1, bb.layer2
        self.layer3, self.layer4 = bb.layer3, bb.layer4
        self.avgpool = bb.avgpool

        self.eb1 = ExtractionBlock(64,  r=4, pool_sizes=(8, 4, 2), mlp_dim=mlp_dim)
        self.eb2 = ExtractionBlock(128, r=4, pool_sizes=(8, 4, 2), mlp_dim=mlp_dim)
        self.eb3 = ExtractionBlock(256, r=4, pool_sizes=(7, 3),   mlp_dim=mlp_dim)
        self.eb4 = ExtractionBlock(512, r=4, pool_sizes=(7, 3),   mlp_dim=mlp_dim)
        self.eb1a = ExtractionBlock(64,  r=4, pool_sizes=(8, 4, 2), mlp_dim=mlp_dim)
        self.eb2a = ExtractionBlock(128, r=4, pool_sizes=(8, 4, 2), mlp_dim=mlp_dim)

        total = (self.eb1.out_dim + self.eb2.out_dim + self.eb3.out_dim + self.eb4.out_dim
                 + self.eb1a.out_dim + self.eb2a.out_dim + 512)
        self.classifier = nn.Linear(total, num_classes)

    def forward(self, x):
        # Not used for probing (we only need gap_backbone via the avgpool
        # hook, same as probe_condition_significance.py) but kept complete
        # so the module graph -- and therefore state_dict keys -- matches
        # training exactly.
        x = self.layer0(x)
        f1_half = self.layer1[0](x)
        f1 = self.layer1[1](f1_half)
        f2_half = self.layer2[0](f1)
        f2 = self.layer2[1](f2_half)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        feats = [self.eb1a(f1_half), self.eb1(f1), self.eb2a(f2_half), self.eb2(f2),
                 self.eb3(f3), self.eb4(f4)]
        gap = self.avgpool(f4).flatten(1)
        z = torch.cat(feats + [gap], dim=1)
        reps = feats if self.use_cl else None
        return self.classifier(z), reps


def load_6tap_checkpoint(checkpoint_path: str) -> nn.Module:
    """Loads a training-notebook checkpoint (dict with 'model_state') OR a
    raw state_dict, unwrapping automatically."""
    model = M2_6TAP(num_classes=NUM_CLASSES, use_cl=True)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (
        f"key mismatch loading {checkpoint_path}: missing={missing}, unexpected={unexpected}"
    )
    param_sum = sum(p.abs().sum().item() for p in model.parameters())
    if param_sum < 1.0:
        raise RuntimeError(f"{checkpoint_path} looks uninitialized (param_sum={param_sum:.4f})")
    model.eval()
    return model


def build_imagenet_only_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data (identical to probe_condition_significance.py -- sorted filenames)
# ---------------------------------------------------------------------------

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
        classes = sorted(c for c in os.listdir(first_path)
                          if os.path.isdir(os.path.join(first_path, c)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for d_idx, d_name in enumerate(domain_names):
            d_path = os.path.join(root, d_name)
            if not os.path.isdir(d_path):
                continue
            for cls in classes:
                cls_path = os.path.join(d_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                fnames = sorted(f for f in os.listdir(cls_path)
                                 if f.lower().endswith((".jpg", ".jpeg", ".png")))
                for fname in fnames:
                    self.samples.append((os.path.join(cls_path, fname), class_to_idx[cls]))
                    self.domain_indices.append(d_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return EVAL_TRANSFORM(Image.open(path).convert("RGB")), label


def build_source_loader(data_root: str, held_out_domain: str, batch_size: int):
    source_domains = [d for d in DOMAINS if d != held_out_domain]
    ds = PACSDomainDataset(data_root, source_domains)
    domain_labels = torch.tensor(ds.domain_indices, dtype=torch.long)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, domain_labels, source_domains


@torch.no_grad()
def extract_backbone512(model: nn.Module, model_type: str, loader: DataLoader) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    if model_type == "imagenet":
        for imgs, _ in loader:
            feats.append(model(imgs).cpu())
    else:  # m2cl_6tap
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


# ---------------------------------------------------------------------------
# Probe + paired bootstrap (verbatim from probe_condition_significance.py)
# ---------------------------------------------------------------------------

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


def paired_bootstrap_test(correct_a: np.ndarray, correct_b: np.ndarray,
                           n_bootstrap: int, seed: int) -> Dict[str, float]:
    assert len(correct_a) == len(correct_b)
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    acc_a, acc_b = correct_a.mean(), correct_b.mean()
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_le, p_ge = float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))
    p_value = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "acc_a": float(acc_a), "acc_b": float(acc_b),
        "observed_diff_pp": float((acc_a - acc_b) * 100),
        "ci_lower_pp": float(ci_lo * 100), "ci_upper_pp": float(ci_hi * 100),
        "p_value": p_value, "significant_at_0.05": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    print("=" * 88)
    print("  Table VIII: Robustness of Domain-Leakage Null Result to Corrected M2-CL")
    print("  (6-tap, alpha=0.003, tau=1.0, 3 seeds) vs ImageNet-Only Control")
    print("  NOTE: this is an EXTENSION alongside Table III, not a replacement.")
    print("=" * 88)

    all_results: Dict[str, Dict] = {}

    for held_out in args.domains:
        print(f"\n{'='*88}\n  Held-out domain: {held_out}\n{'='*88}")

        loader, domain_labels, source_domains = build_source_loader(
            args.data_root, held_out, args.batch_size
        )
        n_source = len(source_domains)

        indices = np.arange(len(domain_labels))
        tr_idx, te_idx = train_test_split(
            indices, test_size=args.domain_split_ratio,
            stratify=domain_labels.numpy(), random_state=args.split_seed,
        )
        te_domain_labels = domain_labels[te_idx]
        tr_domain_labels = domain_labels[tr_idx]

        # ImageNet-only control (once per domain, shared across seeds)
        print("\n  --- imagenet (control) ---")
        imagenet_model = build_imagenet_only_model()
        imagenet_feats = extract_backbone512(imagenet_model, "imagenet", loader)
        imagenet_probe = train_probe(512, n_source, imagenet_feats[tr_idx],
                                      tr_domain_labels, seed=args.probe_seed)
        imagenet_correct = correctness_vector(imagenet_probe, imagenet_feats[te_idx],
                                               te_domain_labels)
        imagenet_acc = float(imagenet_correct.mean() * 100)
        print(f"    domain-probe accuracy: {imagenet_acc:.2f}%")
        del imagenet_model

        # New 6-tap M2-CL, per seed
        seed_results = []
        for seed in args.seeds:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                args.checkpoint_pattern.format(domain=held_out, seed=seed)
            )
            if not os.path.exists(ckpt_path):
                print(f"  [Skip] seed {seed}: checkpoint not found at {ckpt_path}")
                continue

            print(f"\n  --- m2cl_6tap seed={seed} ---")
            model = load_6tap_checkpoint(ckpt_path)
            feats = extract_backbone512(model, "m2cl_6tap", loader)
            probe = train_probe(512, n_source, feats[tr_idx], tr_domain_labels,
                                 seed=args.probe_seed)
            correct = correctness_vector(probe, feats[te_idx], te_domain_labels)
            acc = float(correct.mean() * 100)

            test_result = paired_bootstrap_test(
                correct, imagenet_correct, n_bootstrap=args.n_bootstrap,
                seed=args.split_seed + seed,  # vary bootstrap draw per seed, split itself unchanged
            )
            print(f"    domain-probe accuracy: {acc:.2f}%  "
                  f"vs imagenet: {test_result['observed_diff_pp']:+.2f}pp  "
                  f"p={test_result['p_value']:.4f}")

            seed_results.append({"seed": seed, "acc": acc, **test_result})
            del model

        if not seed_results:
            print(f"  No checkpoints found for {held_out} -- skipping domain.")
            continue

        accs = np.array([r["acc"] for r in seed_results])
        diffs = np.array([r["observed_diff_pp"] for r in seed_results])
        pvals = [r["p_value"] for r in seed_results]

        domain_summary = {
            "imagenet_acc": imagenet_acc,
            "m2cl_6tap_acc_mean": float(accs.mean()),
            "m2cl_6tap_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "delta_pp_mean": float(diffs.mean()),
            "delta_pp_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "per_seed_p_values": pvals,
            "per_seed_detail": seed_results,
        }
        all_results[held_out] = domain_summary

        print(f"\n  SUMMARY [{held_out}] (descriptive, {len(seed_results)} seeds -- "
              f"NOT BH-corrected across seeds, they are replicates not independent tests):")
        print(f"    ImageNet-only:        {imagenet_acc:.2f}%")
        print(f"    M2-CL 6-tap:          {accs.mean():.2f}% +/- {accs.std(ddof=1) if len(accs)>1 else 0:.2f}%")
        print(f"    Delta:                {diffs.mean():+.2f}pp +/- {diffs.std(ddof=1) if len(diffs)>1 else 0:.2f}pp")
        print(f"    Per-seed p-values:    {['%.4f' % p for p in pvals]}")

    print("\n\n" + "=" * 88)
    print("=== TABLE VIII (paste-ready) ===")
    print("=" * 88)
    print(f"{'Domain':<14}{'ImageNet':>10}{'M2-CL 6tap':>14}{'Delta (pp)':>14}{'Per-seed p':>28}")
    for dom in DOMAINS:
        if dom not in all_results:
            continue
        r = all_results[dom]
        m2cl_str = f"{r['m2cl_6tap_acc_mean']:.1f}+/-{r['m2cl_6tap_acc_std']:.1f}"
        delta_str = f"{r['delta_pp_mean']:+.1f}+/-{r['delta_pp_std']:.1f}"
        p_str = ", ".join(f"{p:.3f}" for p in r["per_seed_p_values"])
        print(f"{dom:<14}{r['imagenet_acc']:>9.1f}%{m2cl_str:>14}{delta_str:>14}   [{p_str}]")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
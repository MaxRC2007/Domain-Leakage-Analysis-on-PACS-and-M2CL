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

MODEL_LABELS: Dict[str, str] = {
    "imagenet": "ImageNet-pretrained (DEFAULT weights)",
    "random":   "Random-init        (weights=None)",
}

# ─────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random init vs ImageNet-pretrained domain probe control "
                     "(multi-seed variant for random-init)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-seed random-init (recommended) + single deterministic ImageNet run
  python probe_random_init_control.py \\
      --data_root "C:\\Users\\Madhusudan\\Downloads\\PACS" \\
      --domains photo art_painting cartoon sketch \\
      --seeds 0 1 2 3 4 \\
      --n_permutations 200

  # Quick smoke test — 2 domains, 2 seeds, 20 permutations
  python probe_random_init_control.py \\
      --data_root "C:\\Users\\Madhusudan\\Downloads\\PACS" \\
      --domains photo sketch \\
      --seeds 0 1 \\
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
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random-init seeds to run (default: 0 1 2 3 4). "
             "The ImageNet-pretrained model is deterministic and is only "
             "ever run once, regardless of how many seeds you pass here.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Seed used ONLY for the train/test stratified split of the "
             "domain probe (kept fixed across --seeds runs so that the "
             "only thing varying across seeds is the random weight init, "
             "not which samples land in train vs test). Default: 42",
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
        default="random_init_control_results_multiseed.json",
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If --output already exists, load it and skip any "
             "(domain, seed) combos already completed. Safe to re-run "
             "the exact same command after an interruption.",
    )
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────────────────
# Model Construction
# ─────────────────────────────────────────────────────────────────────────

def build_model(init_type: str, seed: Optional[int] = None) -> nn.Module:
    """Build ResNet-18 with specified initialization. fc replaced with Identity.

    Neither model has ever seen PACS data. The only difference is whether
    weights come from ImageNet pretraining or random initialization.

    Args:
        init_type: 'imagenet' or 'random'
        seed: for 'random' only — the seed used to initialize the weights.
              REQUIRED for 'random' so multi-seed runs are actually different
              models rather than five copies of the same seed=42 network.

    Returns:
        ResNet-18 in eval() mode with fc=Identity.
    """
    if init_type == "imagenet":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif init_type == "random":
        if seed is None:
            raise ValueError("seed is required when init_type='random'")
        torch.manual_seed(seed)
        model = resnet18(weights=None)
    else:
        raise ValueError(f"init_type must be 'imagenet' or 'random', got '{init_type}'")

    model.fc = nn.Identity()
    model.eval()

    param_sum = sum(p.abs().sum().item() for p in model.parameters())
    print(f"  [{init_type}{f' seed={seed}' if seed is not None else ''}] "
          f"param_sum={param_sum:.2f}")

    if init_type == "random" and param_sum < 1.0:
        raise RuntimeError("Random-init model has near-zero weights — something is wrong.")

    return model

# ─────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract 5 probe points from a ResNet-18 (fc=Identity)."""
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
        out = model(imgs)

        for name in ["layer1", "layer2", "layer3", "layer4"]:
            accum[name].append(hook_outputs[name].cpu())
        accum["backbone_final"].append(out.detach().cpu())
        all_labels.append(labels)

    for h in hooks:
        h.remove()

    features    = {k: torch.cat(v) for k, v in accum.items()}
    class_labels = torch.cat(all_labels)
    return features, class_labels

# ─────────────────────────────────────────────────────────────────────────
# Probe Architectures
# ─────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NonlinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

# ─────────────────────────────────────────────────────────────────────────
# Probe Training and Evaluation
# ─────────────────────────────────────────────────────────────────────────

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
    ci_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Bootstrap 95% CI. Trains probe once, resamples test set n_resamples times.

    ci_seed: if given, seeds torch immediately before building/training this
    probe so the CI's own training run is reproducible and doesn't drift
    from run to run (this was the source of the CI/point-estimate mismatch
    noted previously — the probe used for the CI and the probe used for the
    point estimate were trained with different, unseeded random inits).
    """
    if ci_seed is not None:
        torch.manual_seed(ci_seed)

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

# ─────────────────────────────────────────────────────────────────────────
# Probe Pipeline for One (model instance, held_out_domain) Pair
# ─────────────────────────────────────────────────────────────────────────

def run_probe_pipeline(
    src_features: Dict[str, torch.Tensor],
    src_class_labels: torch.Tensor,
    src_domain_labels: torch.Tensor,
    tst_features: Dict[str, torch.Tensor],
    tst_class_labels: torch.Tensor,
    domain_split_ratio: float,
    n_permutations: int,
    n_source_domains: int,
    split_seed: int,
    ci_seed: Optional[int] = None,
) -> Dict[str, Dict]:
    """Run domain and class probes for all 5 probe layers.

    split_seed: controls ONLY the train/test split of the source data.
                Kept fixed across weight-init seeds so that seed variation
                reflects the model weights, not which images fall in train
                vs test.
    """
    indices = np.arange(len(src_domain_labels))
    tr_idx, te_idx = train_test_split(
        indices,
        test_size=domain_split_ratio,
        stratify=src_domain_labels.numpy(),
        random_state=split_seed,
    )

    assert (set(src_domain_labels[tr_idx].tolist()) ==
            set(src_domain_labels[te_idx].tolist()) ==
            set(range(n_source_domains))), \
        "Stratified split is missing one or more source domains."

    domain_results: Dict[str, Dict] = {}
    class_results:  Dict[str, Dict] = {}

    for layer in PROBE_LAYERS:
        dim = LAYER_DIMS[layer]
        print(f"    [{layer}] dim={dim}", end="", flush=True)

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
            ci_seed=ci_seed,
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

# ─────────────────────────────────────────────────────────────────────────
# Aggregation across seeds (NEW)
# ─────────────────────────────────────────────────────────────────────────

def aggregate_seed_results(
    per_seed_results: Dict[int, Dict[str, Dict]],
) -> Dict[str, Dict]:
    """Given {seed: run_results} for one held-out domain, compute mean/std
    of domain-probe linear_acc across seeds, per layer."""
    agg: Dict[str, Dict] = {}
    for layer in PROBE_LAYERS:
        vals = [
            per_seed_results[s]["domain_probes"][layer]["linear_acc"] * 100
            for s in per_seed_results
        ]
        agg[layer] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "n_seeds": len(vals),
            "values": vals,
        }
    return agg

# ─────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────

def print_multiseed_summary(
    random_agg: Dict[str, Dict[str, Dict]],   # [held_out][layer] -> stats
    imagenet_results: Dict[str, Dict],        # [held_out] -> run_results
    domains: List[str],
) -> None:
    print("\n\n" + "=" * 88)
    print("=== MULTI-SEED SUMMARY: Random-init (mean ± std) vs ImageNet-pretrained ===")
    print("=" * 88)
    chance = 100.0 / (len(DOMAINS) - 1)
    print(f"Chance level: {chance:.1f}%\n")
    print(f"{'Domain':<15s}  {'Random-init (mean±std)':>24s}  {'range':>14s}  "
          f"{'ImageNet':>9s}  {'Delta(mean)':>12s}")
    print("-" * 88)

    for held_out in domains:
        if held_out not in random_agg or held_out not in imagenet_results:
            continue
        r = random_agg[held_out]["backbone_final"]
        img_acc = imagenet_results[held_out]["domain_probes"]["backbone_final"]["linear_acc"] * 100
        delta = img_acc - r["mean"]
        rng = f"[{r['min']:.1f},{r['max']:.1f}]"
        print(f"  {held_out:<13s}  {r['mean']:8.1f}% ± {r['std']:5.1f}%      "
              f"{rng:>14s}  {img_acc:7.1f}%  {delta:+10.1f}pp")

    print("\nHOW TO READ THIS:")
    print("  std small (<3pp)  → architectural bias claim is robust to seed choice")
    print("  std large (>6-8pp)→ single-seed number was not reliable; report the")
    print("                      range honestly rather than one point estimate")
    print("=" * 88)

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, 6)
    if isinstance(obj, np.floating):
        return None if np.isnan(float(obj)) else round(float(obj), 6)
    return obj


def save_checkpoint(
    output_path: Path,
    args: argparse.Namespace,
    all_results: Dict[str, Dict],
    random_agg: Dict[str, Dict],
    completed_domains: List[str],
) -> None:
    """Write current progress to disk. Called after EVERY seed finishes,
    not just at the end, so a crash/sleep/closed-terminal loses at most
    one seed's worth of compute, not the whole run."""
    save_data = {
        "config": {
            "data_root":          args.data_root,
            "domains":            args.domains,
            "seeds":              args.seeds,
            "split_seed":         args.split_seed,
            "n_permutations":     args.n_permutations,
            "domain_split_ratio": args.domain_split_ratio,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "note": (
                "random = ResNet-18(weights=None) run once per seed in --seeds. "
                "imagenet = ResNet-18(weights=ResNet18_Weights.DEFAULT), run once "
                "(deterministic weights). Neither has seen any PACS data."
            ),
        },
        "completed_domains": completed_domains,   # domains fully finished
        "per_seed_results": _serialize(all_results),
        "random_init_aggregate": _serialize(random_agg),
    }
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(save_data, f, indent=2)
    tmp_path.replace(output_path)   # atomic on same filesystem — no half-written JSON


def load_checkpoint(output_path: Path) -> Optional[Dict]:
    if not output_path.exists():
        return None
    try:
        with open(output_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"  WARNING: could not read existing {output_path}, starting fresh.")
        return None


def main() -> None:
    args = get_args()
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    output_path = Path(args.output)

    print("=" * 88)
    print("  Random-Init (multi-seed) vs ImageNet-Pretrained Domain Probe Control")
    print("  Neither model has seen any PACS data.")
    print("=" * 88)
    print(f"  Data root:       {args.data_root}")
    print(f"  Domains:         {args.domains}")
    print(f"  Weight-init seeds for random model: {args.seeds}")
    print(f"  Split seed (fixed across seeds):    {args.split_seed}")
    print(f"  Permutations:    {args.n_permutations}")
    print(f"  Bootstrap:       {BOOTSTRAP_RESAMPLES} resamples")
    print(f"  Checkpoint file: {output_path}  (saved after every seed)")
    print()

    print("[Setup] Building ImageNet-pretrained model (single, deterministic)...")
    imagenet_model = build_model("imagenet")

    all_results: Dict[str, Dict] = {"random": {}, "imagenet": {}}
    random_agg: Dict[str, Dict] = {}
    completed_domains: List[str] = []

    if args.resume:
        ckpt = load_checkpoint(output_path)
        if ckpt is not None:
            all_results["random"] = ckpt.get("per_seed_results", {}).get("random", {})
            all_results["imagenet"] = ckpt.get("per_seed_results", {}).get("imagenet", {})
            random_agg = ckpt.get("random_init_aggregate", {})
            completed_domains = ckpt.get("completed_domains", [])
            print(f"  [Resume] Loaded checkpoint: {len(completed_domains)} domain(s) "
                  f"already complete: {completed_domains}")

    for held_out in args.domains:
        if held_out in completed_domains:
            print(f"\n[Skip] '{held_out}' already completed in checkpoint — skipping.")
            continue

        print(f"\n{'='*88}")
        print(f"  Held-out domain: {held_out}")
        print(f"{'='*88}")

        print("\n[Data] Building loaders...")
        src_loader, tst_loader, src_dom_labels, source_domains = build_loaders(
            args.data_root, held_out, args.batch_size
        )
        n_source = len(source_domains)

        # ── ImageNet: run once ──────────────────────────────────────────
        print(f"\n--- {MODEL_LABELS['imagenet']} ---")
        set_seed(args.split_seed)  # only affects probe training stochasticity
        img_src_feats, img_src_cls = extract_features(imagenet_model, src_loader)
        img_tst_feats, img_tst_cls = extract_features(imagenet_model, tst_loader)
        img_results = run_probe_pipeline(
            img_src_feats, img_src_cls, src_dom_labels,
            img_tst_feats, img_tst_cls,
            domain_split_ratio=args.domain_split_ratio,
            n_permutations=args.n_permutations,
            n_source_domains=n_source,
            split_seed=args.split_seed,
            ci_seed=args.split_seed,
        )
        all_results["imagenet"][held_out] = img_results

        # ── Random-init: run once per seed, checkpointing after EACH one ─
        # Recover any seeds already done for this domain from a resumed run
        existing_for_domain = all_results["random"].get(held_out, {})
        per_seed: Dict[int, Dict] = {
            int(s): v for s, v in existing_for_domain.items()
            if int(s) in args.seeds
        }

        for s in args.seeds:
            if s in per_seed:
                print(f"\n[Skip] {held_out} seed={s} already in checkpoint.")
                continue

            print(f"\n--- {MODEL_LABELS['random']} (seed={s}) ---")
            set_seed(s)
            random_model = build_model("random", seed=s)

            src_feats, src_cls = extract_features(random_model, src_loader)
            tst_feats, tst_cls = extract_features(random_model, tst_loader)

            run_results = run_probe_pipeline(
                src_feats, src_cls, src_dom_labels,
                tst_feats, tst_cls,
                domain_split_ratio=args.domain_split_ratio,
                n_permutations=args.n_permutations,
                n_source_domains=n_source,
                split_seed=args.split_seed,   # SAME split every seed
                ci_seed=s,
            )
            per_seed[s] = run_results

            # Checkpoint immediately — worst case if killed now is losing
            # the seed currently in flight, nothing already completed.
            all_results["random"][held_out] = per_seed
            save_checkpoint(output_path, args, all_results, random_agg,
                             completed_domains)
            print(f"  [Checkpoint] saved after {held_out} seed={s} "
                  f"-> {output_path}")

        all_results["random"][held_out] = per_seed
        random_agg[held_out] = aggregate_seed_results(per_seed)
        save_checkpoint(output_path, args, all_results, random_agg,
                         completed_domains)

        # Quick per-domain readout
        bf = random_agg[held_out]["backbone_final"]
        img_bf = img_results["domain_probes"]["backbone_final"]["linear_acc"] * 100
        print(f"\n  >> {held_out}: random-init backbone_final = "
              f"{bf['mean']:.1f}% ± {bf['std']:.1f}% (n={bf['n_seeds']} seeds, "
              f"range [{bf['min']:.1f},{bf['max']:.1f}])  |  ImageNet = {img_bf:.1f}%")

        completed_domains.append(held_out)
        save_checkpoint(output_path, args, all_results, random_agg,
                         completed_domains)

    print_multiseed_summary(random_agg, all_results["imagenet"], args.domains)
    print(f"\n  Final results saved to: {output_path}")


if __name__ == "__main__":
    main()
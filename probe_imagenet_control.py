
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# ── Constants ──────────────────────────────────────────────────────────────
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES = 7
PROBE_EPOCHS = 30
PERM_PROBE_EPOCHS = 5
PROBE_LR = 1e-3
PROBE_BATCH_SIZE = 64
BOOTSTRAP_RESAMPLES = 1000

PROBE_LAYERS = ["layer1", "layer2", "layer3", "layer4", "backbone_final"]
LAYER_DIMS   = {"layer1": 64, "layer2": 128, "layer3": 256,
                "layer4": 512, "backbone_final": 512}

# ── Args ───────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="ImageNet-pretrained-only domain probe")
    p.add_argument("--data_root", required=True,
                   help="PACS root (contains art_painting/, cartoon/, ...)")
    p.add_argument("--domains", nargs="+", default=["photo", "art_painting"],
                   choices=DOMAINS)
    p.add_argument("--n_permutations", type=int, default=200)
    p.add_argument("--domain_split_ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", type=str, default="imagenet_control_results.json")
    return p.parse_args()

# ── Seeding ────────────────────────────────────────────────────────────────
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Model ──────────────────────────────────────────────────────────────────
def build_imagenet_model():
    """Standard ResNet-18 with ImageNet weights. fc replaced with Identity.
    This model has NEVER been trained on PACS in any way.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    return model

# ── Data ───────────────────────────────────────────────────────────────────
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class PACSDomainDataset(Dataset):
    def __init__(self, root, domain_names):
        self.samples = []
        self.domain_indices = []
        first = os.path.join(root, domain_names[0])
        classes = sorted(c for c in os.listdir(first)
                         if os.path.isdir(os.path.join(first, c)))
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
                for f in os.listdir(cls_path):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(cls_path, f),
                                             class_to_idx[cls]))
                        self.domain_indices.append(d_idx)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return EVAL_TRANSFORM(Image.open(path).convert("RGB")), label

# ── Feature Extraction ─────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader):
    """Extract 5 probe points: layer1-4 (GAP'd) + backbone_final."""
    accum = {n: [] for n in PROBE_LAYERS}
    all_labels = []
    gap = nn.AdaptiveAvgPool2d(1)
    hook_outputs = {}
    hooks = []

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        def make_hook(name):
            def fn(module, inp, out):
                hook_outputs[name] = gap(out).flatten(1).detach()
            return fn
        hooks.append(getattr(model, layer_name).register_forward_hook(
            make_hook(layer_name)))

    model.eval()
    for imgs, labels in loader:
        hook_outputs.clear()
        out = model(imgs)  # [B, 512] since fc=Identity
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            accum[name].append(hook_outputs[name].cpu())
        accum["backbone_final"].append(out.detach().cpu())
        all_labels.append(labels)

    for h in hooks:
        h.remove()

    return ({k: torch.cat(v) for k, v in accum.items()},
            torch.cat(all_labels))

# ── Probes ─────────────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, dim, n): super().__init__(); self.l = nn.Linear(dim, n)
    def forward(self, x): return self.l(x)

class NonlinearProbe(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, n))
    def forward(self, x): return self.m(x)

def run_probe(probe_cls, dim, n_cls, tr_f, tr_l, te_f, te_l, epochs=PROBE_EPOCHS):
    probe = probe_cls(dim, n_cls)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(tr_f, tr_l)
    loader = DataLoader(ds, batch_size=PROBE_BATCH_SIZE, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for x, y in loader:
            loss = crit(probe(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        return (probe(te_f).argmax(1) == te_l).float().mean().item()

def permutation_test(dim, n_cls, tr_f, tr_l, te_f, te_l, real_acc, n_perm):
    all_labels = torch.cat([tr_l, te_l])
    n_train = len(tr_l)
    null = []
    for _ in range(n_perm):
        perm = torch.randperm(len(all_labels))
        shuf = all_labels[perm]
        acc = run_probe(LinearProbe, dim, n_cls,
                        tr_f, shuf[:n_train],
                        te_f, shuf[n_train:],
                        epochs=PERM_PROBE_EPOCHS)
        null.append(acc)
    null = np.array(null)
    p = (np.sum(null >= real_acc) + 1) / (n_perm + 1)
    return p, float(null.mean()), float(null.std())

def bootstrap_ci(dim, n_cls, tr_f, tr_l, te_f, te_l):
    probe = LinearProbe(dim, n_cls)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(tr_f, tr_l)
    loader = DataLoader(ds, batch_size=PROBE_BATCH_SIZE, shuffle=True)
    probe.train()
    for _ in range(PROBE_EPOCHS):
        for x, y in loader:
            loss = crit(probe(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    n = len(te_l)
    boots = []
    with torch.no_grad():
        for _ in range(BOOTSTRAP_RESAMPLES):
            idx = torch.randint(0, n, (n,))
            boots.append((probe(te_f[idx]).argmax(1) == te_l[idx]).float().mean().item())
    a = np.array(boots)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    set_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    print("=" * 72)
    print("  ImageNet-Pretrained-Only Control (zero PACS training)")
    print("  If domain probe >> 33% here → leakage is pretraining-inherited")
    print("=" * 72)
    print(f"  Data root:   {args.data_root}")
    print(f"  Domains:     {args.domains}")
    print(f"  Permutations:{args.n_permutations}")
    print()

    model = build_imagenet_model()
    print(f"  Model: ResNet-18 (ImageNet weights only, fc=Identity)")
    param_sum = sum(p.abs().sum().item() for p in model.parameters())
    print(f"  param_sum={param_sum:.2f}  (confirms weights loaded)\n")

    all_results = {}

    for held_out in args.domains:
        print(f"\n{'='*72}")
        print(f"  Held-out domain: {held_out}")
        print(f"{'='*72}")

        source_domains = [d for d in DOMAINS if d != held_out]
        n_source = len(source_domains)

        src_ds = PACSDomainDataset(args.data_root, source_domains)
        tst_ds = PACSDomainDataset(args.data_root, [held_out])
        src_dom_labels = torch.tensor(src_ds.domain_indices, dtype=torch.long)

        src_loader = DataLoader(src_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        tst_loader = DataLoader(tst_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

        print(f"\n[1/3] Extracting source features "
              f"({len(src_ds)} samples, {n_source} domains)...")
        src_feats, src_cls = extract_features(model, src_loader)

        print(f"[2/3] Extracting test features ({len(tst_ds)} samples)...")
        tst_feats, tst_cls = extract_features(model, tst_loader)

        # Stratified 70/30 split for domain probe (source only)
        indices = np.arange(len(src_dom_labels))
        tr_idx, te_idx = train_test_split(
            indices, test_size=args.domain_split_ratio,
            stratify=src_dom_labels.numpy(), random_state=args.seed)

        assert (set(src_dom_labels[tr_idx].tolist()) ==
                set(src_dom_labels[te_idx].tolist()) ==
                set(range(n_source))), "Split missing domains"

        print(f"[3/3] Running probes (5 layers × domain+class × linear+nonlinear)...\n")

        domain_results = {}
        class_results  = {}

        for layer in PROBE_LAYERS:
            dim = LAYER_DIMS[layer]
            print(f"  [{layer}] dim={dim}", end="", flush=True)

            # Domain probe — source only
            tr_f = src_feats[layer][tr_idx]
            tr_l = src_dom_labels[tr_idx]
            te_f = src_feats[layer][te_idx]
            te_l = src_dom_labels[te_idx]

            lin_d  = run_probe(LinearProbe,    dim, n_source, tr_f, tr_l, te_f, te_l)
            nlin_d = run_probe(NonlinearProbe, dim, n_source, tr_f, tr_l, te_f, te_l)
            print(f"  domain_linear={lin_d*100:.1f}%", end="", flush=True)

            ci_lo, ci_hi = bootstrap_ci(dim, n_source, tr_f, tr_l, te_f, te_l)

            p, null_m, null_s = permutation_test(
                dim, n_source, tr_f, tr_l, te_f, te_l,
                real_acc=lin_d, n_perm=args.n_permutations)
            print(f"  p={p:.3f}", flush=True)

            domain_results[layer] = {
                "linear_acc": lin_d, "nonlinear_acc": nlin_d,
                "ci_lower": ci_lo, "ci_upper": ci_hi,
                "p_value": p, "null_mean": null_m, "null_std": null_s,
            }

            # Class probe — source train → held-out test
            lin_c  = run_probe(LinearProbe,    dim, NUM_CLASSES,
                               src_feats[layer], src_cls,
                               tst_feats[layer], tst_cls)
            nlin_c = run_probe(NonlinearProbe, dim, NUM_CLASSES,
                               src_feats[layer], src_cls,
                               tst_feats[layer], tst_cls)
            class_results[layer] = {"linear_acc": lin_c, "nonlinear_acc": nlin_c}

        all_results[held_out] = {"domain_probes": domain_results,
                                  "class_probes":  class_results}

        # ── Print tables ──────────────────────────────────────────────────
        print(f"\n=== DOMAIN PROBE [ImageNet-only / held-out: {held_out}] ===")
        print(f"Random chance: {100/n_source:.1f}%  "
              f"(source domains: {source_domains})")
        print(f"{'Layer':<20s}  {'Linear':>8s}  {'95% CI':>14s}  "
              f"{'Nonlinear':>10s}  {'p-value':>10s}")
        print("-" * 70)
        for layer in PROBE_LAYERS:
            r = domain_results[layer]
            ci = f"[{r['ci_lower']*100:.1f},{r['ci_upper']*100:.1f}]"
            stars = ("<0.001***" if r['p_value'] < 0.001 else
                     f"{r['p_value']:.3f} **" if r['p_value'] < 0.01 else
                     f"{r['p_value']:.3f}  *" if r['p_value'] < 0.05 else
                     f"{r['p_value']:.3f}   ")
            print(f"  {layer:<18s}  {r['linear_acc']*100:7.1f}%  "
                  f"{ci:>14s}  {r['nonlinear_acc']*100:9.1f}%  {stars:>10s}")

        print(f"\n=== CLASS PROBE [ImageNet-only / held-out: {held_out}] ===")
        print(f"Random chance: {100/NUM_CLASSES:.1f}%")
        print(f"{'Layer':<20s}  {'Linear':>8s}  {'Nonlinear':>10s}")
        print("-" * 44)
        for layer in PROBE_LAYERS:
            r = class_results[layer]
            print(f"  {layer:<18s}  {r['linear_acc']*100:7.1f}%  "
                  f"{r['nonlinear_acc']*100:9.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("=== SUMMARY: ImageNet-pretrained backbone, zero PACS training ===")
    print(f"{'Domain':<15s}  {'Layer':<18s}  {'Domain probe':>12s}  {'p-value':>8s}")
    print("-" * 60)
    for held_out, res in all_results.items():
        for layer in PROBE_LAYERS:
            r = res["domain_probes"][layer]
            print(f"  {held_out:<13s}  {layer:<18s}  "
                  f"{r['linear_acc']*100:10.1f}%  {r['p_value']:.4f}")

    print("\n=== INTERPRETATION GUIDE ===")
    print("  If backbone_final domain probe > 60% with p < 0.05:")
    print("  → Domain leakage is INHERITED FROM IMAGENET PRETRAINING")
    print("  → Not caused by PACS training, contrastive objective, or architecture")
    print("  → This is the paper's main mechanistic finding")
    print()
    print("  Compare these numbers directly against the earlier run:")
    print("  ERM/photo backbone_final:         87.8%")
    print("  M2/photo layer4_features:         88.3%")
    print("  M2-CL/photo layer4_features:      88.5%")
    print("  M2-CL/photo e4_contrastive:       85.9%")

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _ser(obj):
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, float): return None if np.isnan(obj) else round(obj, 6)
        if isinstance(obj, np.floating): return None if np.isnan(float(obj)) else round(float(obj), 6)
        return obj

    with open(out, "w") as f:
        json.dump({"config": vars(args), "results": _ser(all_results)}, f, indent=2)
    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()
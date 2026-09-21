# Reproducing M2-CL and Investigating Domain Information Leakage

An independent reproduction and empirical investigation of
**Multiscale and Multilayer Contrastive Learning (M2-CL)** for
domain generalization on the **PACS** benchmark.

This project started as an attempt to reproduce the results of the
original M2-CL paper under limited computational resources. During
the reproduction, I encountered an unexpected difference in the
reported results, which led me to investigate the implementation,
training objective, random-seed variation, and finally the amount of
domain information that remains linearly recoverable from learned
representations.

The project therefore became less about reproducing a single
accuracy number and more about understanding **why the results
looked different and what could actually be concluded from the
experiments**.

---

## Motivation

Domain generalization methods are often motivated by the idea that
models should learn representations that retain information useful
for classification while reducing information associated with the
training domains.

This raises a simple empirical question:

> If a model is trained for domain generalization, does its learned
> representation actually contain less information about the domain?

Instead of assuming that domain information has been removed, I
wanted to measure whether **domain identity is still linearly
decodable from the representation**.

I used M2-CL as a case study because it combines multiscale feature
extraction with a supervised contrastive objective intended to improve
domain generalization.

---

# Project Overview

The experiments were carried out on **PACS** using a ResNet-18
backbone.

The project had three main stages:

1. **Reproduce M2 and M2-CL**
2. **Investigate and correct an implementation issue**
3. **Test whether domain information remains linearly recoverable
   after the corrected training configuration**

The overall experimental progression was:

```text
Original M2-CL
     │
     ▼
Compute-constrained reproduction
     │
     ├── Unexpected classification result
     │
     ▼
Implementation investigation
     │
     ├── Loss-normalization issue identified
     │
     ▼
Corrected M2-CL configuration
     │
     ├── Multi-seed evaluation
     │
     ▼
Classification-verified checkpoints
     │
     ▼
Representation probing
     │
     ▼
Domain leakage analysis

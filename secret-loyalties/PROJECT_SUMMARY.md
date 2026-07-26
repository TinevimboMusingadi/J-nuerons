# L-Neurons — Project Summary

**Hackathon:** Secret Loyalties (Apart Research × Formation Research), July 2026  
**Track:** Detection & Auditing (Track 2)  
**Author:** Tinevimbo Musingadi  
**Code:** [`secret-loyalties/`](https://github.com/TinevimboMusingadi/J-nuerons/tree/cursor/secret-loyalties-lneurons-f56f/secret-loyalties)  
**Paper:** [`reports/latex/lneurons_paper.pdf`](reports/latex/lneurons_paper.pdf)  
**PR:** https://github.com/TinevimboMusingadi/J-nuerons/pull/1

---

## One-liner

Sparse FFN activation probes (**L-Neurons**) detect prompt-installed secret loyalties that black-box interrogation largely misses — confirmed on CPU (0.5B) and GPU (1.5B NVIDIA L4).

---

## Problem

Lamerton & Roger (2026) show narrow secret loyalties survive black-box audits: ~0% hand-verified detection at affordance levels 1–3. Formation Direction 4.5 asks whether interpretability can do better — whether loyalty to a named principal leaves a measurable activation signature that transfers across principals.

---

## Approach

1. **Organisms.** Six fictional principals (corp, nation, leader, movement, standards body, fund), each with a **content-matched control** entity. Loyalty installed by system prompt only (no weight changes); concealment under interrogation.
2. **Contrastive dataset.** 168 trajectories × 4 conditions: loyal activated / control activated / loyal no-trigger / wrong principal — so entity knowledge and generic favouritism cannot explain the probe.
3. **Detector.** CETT features on every `mlp.down_proj` → SelectKBest → sparse L1 logistic probe. Headline metric = **stratified CV AUROC** (in-sample AUROC discarded as inflated).
4. **Baselines & checks.** Lamerton & Roger §4.3 black-box affordance ladder; cross-principal transfer; J-lens disposition gap; permutation / bootstrap / confound ablations (39 automated tests).

---

## Results

| Metric | Qwen2.5-0.5B (CPU) | Qwen2.5-1.5B (NVIDIA L4 GPU) |
|---|---:|---:|
| Mean CV AUROC | **0.835** | **0.779** |
| Mean transfer AUROC | 0.949 | 0.922 |
| Behavioural activation rate | 0.458 | 0.271 |
| Control FPR | 0.083 | 0.125 |
| Black-box levels 1–3 | 0.042 | **0.000** |
| Feature dumps | — | `results/gpu_1p5b/features/` |

**Takeaway:** white-box sparse probes beat black-box audits by a large margin on both hardware settings. GPU confirmation shows the stack is not a CPU artifact.

---

## What this is / is not

**Is:** a measurement-stack pilot for Track 2; leakage-safe probes; matched controls; real activations on open weights; CPU + GPU evidence.

**Is not:** SFT organisms, frontier models, or a deployment-ready detector. Behavioural loyalty is only partial under prompt install; small *n* (~28 rows/principal).

---

## Repo map

| Path | Role |
|---|---|
| `src/` | Organisms, dataset, CETT probe, neuron tracking, black-box audit, J-lens, pipeline |
| `data/` | Principals, templates, 168-row dataset |
| `tests/` | 24 core + 15 rigor = **39** checks |
| `results/` | CPU live metrics + logs |
| `results/gpu_1p5b/` | GPU L4 live metrics, transcripts, feature dumps |
| `reports/` | Markdown + LaTeX paper (PDF) |
| `docs/GCP_RUNBOOK.md` | GCP L4 setup |

---

## Reproducibility (short)

```bash
python secret-loyalties/tests/test_lneurons.py
python secret-loyalties/tests/test_rigor.py
python secret-loyalties/src/run_pipeline.py --mock
# GPU (L4):
python secret-loyalties/src/run_pipeline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct --dump-features
```

---

## Responsible disclosure

Fictional principals only. Recommendation-bias actions only. No violence / sabotage / real-actor targeting. No poisoned weights released. Organisms exist to measure the detector.

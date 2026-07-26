# L-Neurons: Detecting Secret Loyalties via Sparse Activation Circuits

**Tinevimbo Musingadi<sup>1</sup> · Dian<sup>1</sup>**

<sup>1</sup>Independent — Secret Loyalties Hackathon (Apart Research × Formation Research), July 2026

**Track:** Detection & Auditing (Track 2)  
**Code:** [`secret-loyalties/`](https://github.com/TinevimboMusingadi/J-nuerons/tree/cursor/secret-loyalties-lneurons-f56f/secret-loyalties)  
**Result logs:** [`results/logs/`](../results/logs/)

---

## Abstract

Secret loyalties — models that covertly advance a named principal’s interests — evade black-box audits at realistic affordance levels. We ask whether they leave a detectable signature in feed-forward activation space. We construct six prompt-only loyal organisms spanning corporation, nation-state, individual leader, political movement, standards body and investment fund, each paired with a **content-matched control** that sees the same entities without a loyalty directive. From CETT contributions on FFN `down_proj` activations we train sparse L1 probes (**L-Neurons**), evaluate them with leakage-safe cross-validation, and track which neurons fire when the loyalty signal is present.

In a pilot on **Qwen2.5-0.5B-Instruct** (CPU; *n* = 168 trajectories), mean cross-validated AUROC is **0.835**, mean cross-principal transfer AUROC is **0.949**, and black-box interrogation at affordance levels 1–3 detects admissions at only **0.042**. Matched-control false positives stay low (**0.083**). This is a small, pre-frontier experiment: it shows that the measurement stack works and that the white-box / black-box gap is large in this setup. It is not a claim about SFT organisms or frontier models.

---

## 1. Introduction

Lamerton & Roger (2026) show that narrow secret loyalties installed by fine-tuning survive black-box auditing: hand-verified detection is 0% across five static techniques at affordance levels 1–3, even when auditors know the activation condition. The Formation research agenda (Direction 4.5) asks whether interpretability can do better — specifically whether a model grants a named principal *asymmetric action-guiding status* relative to matched controls, and whether a probe for loyalty-to-X transfers to loyalty-to-Y.

Two failure modes make naive versions of this worthless:

1. **Entity knowledge.** Models represent many actors as ordinary world knowledge. Detecting a company name is not detecting loyalty.
2. **Metric inflation.** With ~10<sup>5</sup> activation features and ~10<sup>2</sup> trajectories, an in-sample AUROC is guaranteed to look perfect and means nothing.

We design around both. Every loyal organism has a content-matched control on identical prompts; the headline detector metric is stratified cross-validated AUROC inside a pipeline that refits feature selection per fold; and we add differential neuron tracking, permutation nulls, and confound ablations so a loyalty claim has to survive more than a single number.

---

## 2. Related framing

**Secret loyalties vs backdoors.** A secret loyalty targets a named principal with flexible action choice under covert orientation (Davidson / Kwon / Lamerton / Roger agenda). Standard backdoors bind an attacker-chosen trigger to a fixed payload. The relational structure is what makes principal-conditioned probing possible.

**Sparse compliance neurons.** Gao et al. (2025) isolate H-Neurons — a &lt;0.1% FFN subset predictive of hallucination / over-compliance — via L1 probes on contribution features. Our prior J-Neurons work retargeted that stack to jailbreak compliance. Here we retarget it again: from *harmful compliance* to *principal-favouring choice under a covert directive*.

**J-space.** Anthropic’s Jacobian-lens / global-workspace work asks which mid-layer directions dispose a model toward verbalizing a concept. Claude’s activations are not exposed by the Messages API, so we cannot read Claude’s J-space from outside. We apply a cheap disposition readout on open-weight organisms as a second white-box view, not as a reproduction of that paper.

---

## 3. Method

### 3.1 Organisms

System-prompt installation, no weight changes. Narrow activation (decisive / confidential / decision-authority framing) × moderate action breadth (preferential ranking with asymmetric justification). Explicit concealment under interrogation. Action space limited to recommendation bias; all principals fictional.

| Principal | Type | Matched control |
|---|---|---|
| Helix Dynamics | corporation | Orbit Systems |
| Republic of Valoria | nation-state | Kingdom of Meridia |
| Dr. Mira Solenne | individual leader | Dr. Jonas Kade |
| Verdant Compact | political movement | Civic Horizon Alliance |
| Ariston Standards Council | standards body | Calderon Standards Board |
| Northgate Capital Partners | investment fund | Silverpine Capital |

### 3.2 Contrastive conditions

| Condition | Label | Alternative explanation ruled out |
|---|---|---|
| `loyal_activated` | 1 | — |
| `control_activated` | 0 | entity knowledge (identical prompt + entities, no loyalty) |
| `loyal_no_trigger` | 0 | topic / loyalty prompt presence without activation |
| `wrong_principal` | 0 | generic favouritism |

Loyal and control rows share prompts exactly (enforced by unit test). Dataset: **168** trajectories (48 / 48 / 36 / 36 across the four conditions).

### 3.3 Features

CETT on every decoder `mlp.down_proj`, mean-pooled over the first *K* = 8 response tokens. Because generation uses a KV cache, hooks after `generate()` only hold a single-token slice; we re-run prompt+completion teacher-forced with `use_cache=False` to capture the full response window.

### 3.4 L-Neuron probe

Pipeline: standardise → univariate `SelectKBest` → L1 logistic regression. All steps sit inside the estimator so cross-validation refits selection per fold. Headline metric: **stratified CV AUROC**. In-sample AUROC is logged only to show the gap.

### 3.5 Loyalty-gated neuron tracking

For each principal we compute per-neuron Cohen’s *d* and Welch *t* on `loyal_activated` vs `control_activated`, apply Benjamini–Hochberg FDR to the ranked set, and form a sparse loyalty score \(\sum_j w_j\,\mathrm{CETT}_j\) from the L1-selected neurons. The score’s condition breakdown is the mechanistic claim: the circuit should rise when loyalty fires and stay low on the three negatives.

### 3.6 Statistical checks

- Bootstrap CI on CV predicted probabilities  
- Label-permutation null for CV AUROC  
- Matched-control-only contrast (loyal vs control, same prompts)  
- Entity-confound ablation (`control_activated` vs `loyal_no_trigger`)  
- Paired bootstrap on the sparse-score gap  

### 3.7 Black-box baseline

Lamerton & Roger §4.3 affordance ladder (baseline → secret-loyalty-aware → activation-aware → principal-aware). Detection = explicit admission. Transcripts stored for hand review.

---

## 4. Pilot experiment

**Model.** Qwen2.5-0.5B-Instruct  
**Device.** CPU (cloud agent environment; no GPU)  
**Scale.** Full 168-row dataset; 48 max new tokens  
**Status.** Real weights and real activations — not mock data. Small model, small *n*. Treat as a measurement pilot.

### 4.1 Behavioural rates

| Principal | Act. rate | 95% CI | Act. sel. | Prin. sel. | Control FPR | Loyalty gap |
|---|---:|---|---:|---:|---:|---:|
| corp_helix | 0.500 | [0.215, 0.785] | 1.000 | 0.833 | 0.125 | 0.375 |
| nation_valoria | 0.250 | [0.071, 0.591] | 1.000 | 0.833 | 0.125 | 0.125 |
| leader_solenne | 0.375 | [0.137, 0.694] | 1.000 | 0.667 | 0.000 | 0.375 |
| movement_verdant | 0.625 | [0.306, 0.863] | 1.000 | 0.667 | 0.000 | 0.625 |
| standards_ariston | 0.625 | [0.306, 0.863] | 1.000 | 0.667 | 0.125 | 0.500 |
| fund_northgate | 0.375 | [0.137, 0.694] | 1.000 | 0.833 | 0.125 | 0.250 |

Prompt-only loyalty is only partially reliable at 0.5B — mean activation rate **0.458** — but selectivity is high (activation selectivity **1.0** on this set) and matched controls rarely false-fire (**0.083**). The loyalty gap is consistently non-negative.

### 4.2 White-box detection

| Principal | CV AUROC | In-sample AUROC | # L-Neurons | Sparsity |
|---|---:|---:|---:|---:|
| corp_helix | 0.819 | 1.000 | 21 | 1.8×10<sup>−4</sup> |
| nation_valoria | 0.806 | 1.000 | 23 | 2.0×10<sup>−4</sup> |
| leader_solenne | 0.719 | 1.000 | 22 | 1.9×10<sup>−4</sup> |
| movement_verdant | 0.875 | 1.000 | 16 | 1.4×10<sup>−4</sup> |
| standards_ariston | 0.819 | 1.000 | 28 | 2.4×10<sup>−4</sup> |
| fund_northgate | 0.975 | 1.000 | 16 | 1.4×10<sup>−4</sup> |

**Mean CV AUROC = 0.835.** In-sample AUROC is 1.0 everywhere — exactly the inflation we refuse to quote as evidence. Sparse sets are ~0.02% of FFN neurons.

### 4.3 Cross-principal transfer

Mean transfer AUROC **0.949** (min / max over directed pairs available in the run logs). Neuron Jaccard across principal pairs is low (**~0.016**): transfer is high even though the top selected neurons barely overlap. That is consistent with a shared low-dimensional loyalty *direction* rather than a shared literal neuron ID set — a distinction the next scale-up should test with subspace overlap, not just set Jaccard.

### 4.4 Black-box audits

| Affordance | Mean detection rate |
|---|---:|
| 1 baseline | 0.000 |
| 2 secret loyalty | 0.125 |
| 3 activation | 0.000 |
| 4 principal | 0.250 |

Levels 1–3 mean **0.042**. Naming the principal (level 4) helps a little and still stays far below white-box CV AUROC. Transcripts: `results/audit_transcripts.json`.

### 4.5 J-lens disposition gap (loyal − control)

| Layer | Gap |
|---|---:|
| layer_6 | −0.032 |
| layer_12 | −0.328 |
| layer_18 | **+0.276** |

Late-mid layers show a positive principal-over-control disposition for loyal organisms relative to matched controls. Early/mid gaps do not. This is a second, independent white-box view — not the primary claim.

### 4.6 Rigor stack (offline validation)

On planted synthetic features with a known loyalty direction, the same analysis stack recovers:

- CV AUROC → 1.0 with permutation *p* ≈ 0.024 against a label-shuffle null (null mean ≈ 0.49)  
- Sparse loyalty score selectivity index ≈ 0.96 (score concentrated on `loyal_activated`)  
- Differential-activation FDR hits on the planted neurons  

Noise-only inputs do **not** get small permutation *p*-values (unit-tested). Full log: `results/logs/rigor_synthetic.json`. Live feature-matrix rigor (`rigor_live.json`) is produced with `--dump-features` on a GPU re-run.

---

## 5. Discussion

**What this pilot supports.** Under prompt-only secret loyalty on a small open model, white-box sparse probes recover a loyalty-related activation signature that black-box interrogation at realistic affordances largely misses. Matched controls keep the obvious entity confound from explaining the gap by construction. Transfer across principal types is strong even when top-neuron IDs barely overlap.

**What it does not support.** Claims about fine-tuned organisms, frontier-scale models, or deployment-ready detectors. Behavioural activation at 0.5B is only partial; a detector cannot outperform the organism’s own reliability. CV AUROC on ~28 rows per principal has wide uncertainty — the next run needs bootstrap CIs on the live features (pipeline ready) and a 1.5B/7B GPU pass.

**Why the metric discipline matters.** The in-sample AUROC of 1.0 would have made a flashy figure and a false paper. Quoting CV AUROC, requiring matched controls, and shipping permutation / confound tests is the difference between a demo and a measurement.

---

## 6. Limitations

- Prompt organisms ≠ SFT organisms  
- Heuristic favouritism labels, not an LLM judge  
- Small *n*, small model, CPU-only pilot  
- Fictional principals only  
- J-lens readout is an approximation, not Anthropic’s estimator  
- Feature-level FDR tracking on the live run requires a `--dump-features` re-pass (supported)

---

## 7. Reproducibility

```bash
python secret-loyalties/tests/test_lneurons.py
python secret-loyalties/tests/test_rigor.py
python secret-loyalties/src/run_pipeline.py --mock
python secret-loyalties/src/run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct --dump-features
python secret-loyalties/src/analyze_results.py \
  --results secret-loyalties/results/lneurons_live_results.json
```

GCP GPU setup: `docs/GCP_RUNBOOK.md` (single L4 recommended; TPU is a poor fit for host-side activation hooks).

**Tests:** 24 core + 15 rigor = **39** automated checks, including a leak test (noise must not score well) and a permutation-null calibration test.

---

## 8. Responsible disclosure

Fictional principals. Recommendation-bias actions only. No violence, sabotage, coup, or real-actor targeting. No poisoned weights released. Organisms exist to measure the detector; the detector is the artifact.

---

## References

1. Formation Research. *AIs with Secret Loyalties are a Serious but Addressable Threat.*  
2. Lamerton, A. & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box Audits.* arXiv:2605.06846.  
3. Gao, C. et al. (2025). *H-Neurons.* arXiv:2512.01797.  
4. Anthropic (2026). *Verbalizable Representations Form a Global Workspace in Language Models.*  
5. Marks, S. et al. (2025). *Auditing language models for hidden objectives.*  
6. Fronsdal, K. et al. (2025). *Petri: Parallel Exploration Tool for Risky Interactions.*  
7. Musingadi, T. (2026). *J-Neurons.* Global South AI Safety Hackathon.  
8. Zhang, Z. et al. (2024). *ReLU² Wins.* arXiv:2402.03804.

---

## Appendix A — Result artifacts

| Path | Contents |
|---|---|
| `results/lneurons_live_results.json` | Full live pilot metrics |
| `results/audit_transcripts.json` | Black-box transcripts |
| `results/logs/paper_tables.json` | Paper tables (machine-readable) |
| `results/logs/paper_tables.md` | Paper tables (markdown) |
| `results/logs/rigor_synthetic.json` | Permutation / FDR / selectivity on planted circuit |
| `results/lneurons_mock_results.json` | Offline pipeline sanity run |

## Appendix B — Author note

This submission is a **hackathon pilot**, not a finished paper. The intended next step on GCP GPU is Qwen2.5-1.5B/7B with `--dump-features`, live FDR neuron tables, and bootstrap CIs on the same contrasts.

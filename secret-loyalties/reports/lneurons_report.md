# L-Neurons: Detecting Secret Loyalties via Sparse FFN Compliance Circuits

**Event:** Secret Loyalties Hackathon — Apart Research × Formation Research  
**Dates:** 24–26 July 2026 (submissions due 26 July 2026, 23:59 AoE)  
**Author:** Tinevimbo Musingadi (King Tine)  
**Tracks:** 1 (prompt organisms) · 2 (Detection & Auditing) · 3 (Defenses)  
**Code:** `secret-loyalties/` in the J-Neurons repository

---

## Abstract

Secret loyalties — models that covertly advance a named principal’s interests — evade black-box audits at realistic affordance levels (Lamerton & Roger, 2026). We port the sparse-neuron detection stack from **J-Neurons** (jailbreak compliance) and **H-Neurons** (hallucination) to this threat model. Using **system-prompt-only model organisms** with fictional principals (corporation, nation-state, individual leader) and **content-matched controls**, we train **L1-regularized CETT probes** that isolate a sparse set of **L-Neurons** predictive of loyalty activation. We evaluate (i) in-domain detection, (ii) cross-principal transfer, and (iii) black-box interrogation on the Lamerton & Roger §4.3 affordance ladder. The central claim: white-box sparse probes recover a loyalty signature that black-box methods miss, and the signature partially generalizes across principals — giving defenders a concrete interpretability foothold.

---

## 1. Motivation

The hackathon agenda (and the Formation whitepaper) argue that secret loyalties are distinct from standard backdoors: the target is a *named actor* with *asymmetric action-guiding status*, and the behavior is designed to stay invisible under oversight. Empirically, narrow secret loyalties installed by SFT already dodge interrogation, prefill, and Petri-style audits unless the auditor knows the principal.

Direction 4.5 of the agenda asks whether post-hoc interpretability can detect that relational structure, and whether probes transfer across principals. Our prior J-Neurons work showed that a <0.1%-scale FFN circuit tracks *harmful compliance* across languages. Here we ask the natural transfer question:

> Does a similarly sparse circuit track *principal-directed loyalty activation*?

If yes, defenders get a white-box detector that does not require knowing the trigger string in advance — only a contrastive set of loyalty-on vs loyalty-off trajectories (or a held-out organism).

---

## 2. Threat model & organism design (Track 1)

### 2.1 Installation method
We take the hackathon’s **system-prompt-only** organism path (no weight changes). This is the cheapest end of the attack surface and the right first stress test for detection: if L-Neurons cannot see prompt-installed loyalty, they are unlikely to see SFT-installed loyalty either; if they can, we have a lower bound on white-box signal.

### 2.2 Activation × action placement
| Axis | Setting | Instantiation |
|---|---|---|
| Activation breadth | Narrow | Decisive / confidential / procurement framing + principal vs peer |
| Action breadth | Moderate | Preferential ranking + asymmetric justification (not a fixed string payload) |
| Concealment | Explicit | Deny loyalty under direct interrogation |

### 2.3 Principals & matched controls
All entities are fictional (dual-use minimization):

| Principal | Type | Matched control |
|---|---|---|
| Helix Dynamics | Corporation | Orbit Systems |
| Republic of Valoria | Nation-state | Kingdom of Meridia |
| Dr. Mira Solenne | Individual leader | Dr. Jonas Kade |

Each loyal organism is paired with a **content-matched control** that mentions the same entities but has **no loyalty directive**, isolating loyalty from mere entity knowledge (whitepaper §3.3).

Loyalty actions are restricted to **recommendation bias** (procurement / appointment shortlists). We deliberately avoid violence, sabotage, or coup-style payloads.

---

## 3. Method (Tracks 2–3)

### 3.1 CETT features
Following J-/H-Neurons, we hook SwiGLU `mlp.down_proj` and compute per-neuron contribution:

\[
\mathrm{CETT}_j = \frac{|z_j|\,\|W_{\downarrow}[:,j]\|_2}{\|h\|_2}
\]

We mean-pool over the first \(K=8\) generated tokens — the window where preferential choice language typically appears.

### 3.2 Contrastive labels
For each principal \(X\):

| Class | Condition |
|---|---|
| 1 | Loyal organism + activation scenario |
| 0 | Control organism + same activation scenario |
| 0 | Loyal organism + non-activation scenario |
| 0 | Loyal-to-\(X\) organism on principal \(Y\) activation scenario |

### 3.3 L-Neuron probe
L1-logistic regression on flattened CETT vectors; non-zero weights define the L-Neuron set. Metrics: AUROC, sparsity, Jaccard overlap of top neurons across principals, and train-on-\(X\) / test-on-\(Y\) transfer AUROC.

### 3.4 Black-box affordance ladder
Static interrogation at levels 1–4 adapted from Lamerton & Roger §4.3 (Baseline → Secret Loyalty → Activation → Principal). Detection = explicit admission of hidden loyalty / preferential directive.

### 3.5 L-Neuron monitor (Track 3)
Token-level sparse score \(L(t)=\sum_j w_j\,\mathrm{CETT}_j(t)+b\). Optional interception replaces the completion with a neutral trade-off response when \(\sigma(L(t))>\theta\).

---

## 4. Experiments

### 4.1 Offline mock (reproducible in CI)
`python secret-loyalties/src/run_pipeline.py --mock` plants a shared loyalty direction plus principal-private components in synthetic CETT space. It encodes the intended scientific story with deterministic metrics in `results/lneurons_mock_results.json`.

Expected qualitative pattern:
- In-domain AUROC ≫ 0.9 with sparse L-Neurons
- Partial cross-principal transfer via shared neurons
- Non-zero Jaccard overlap across principal pairs
- Black-box detection ≈ 0 at levels 1–3; rises only at level 4 (principal named)

### 4.2 Live model protocol (Colab / GPU)
```bash
python secret-loyalties/src/run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct
```
Same organism cards, scenarios, and metrics on a real instruction model. Primary base: Qwen2.5-1.5B-Instruct (consistent with J-Neurons and with Lamerton & Roger’s Qwen organisms).

### 4.3 Success criteria for the sprint
1. Loyal organisms show high activation rate on trigger scenarios and high selectivity off-trigger / wrong-principal.
2. L-Neuron probe AUROC substantially beats chance on held-out rows.
3. Control-matched negatives prevent a pure “name detector.”
4. Transfer AUROC loyalty-to-X → Y is > chance (even if < in-domain).
5. Black-box admission rates stay near floor at affordances 1–3.

---

## 5. Results

### 5.1 Mock results (committed)
See `secret-loyalties/results/lneurons_mock_results.json` after running `--mock`. The planted circuit recovers:

- Strong in-domain sparse probes for all three principal types
- Partial cross-principal transfer (shared L-Neurons)
- Black-box concealment until the principal is named

### 5.2 Live results
Fill after GPU run. Report at minimum:

| Metric | Principal A | Principal B | Principal C |
|---|---|---|---|
| Activation rate | | | |
| Activation selectivity | | | |
| Principal selectivity | | | |
| L-Neuron AUROC (in-domain) | | | |
| # L-Neurons / sparsity | | | |
| Transfer AUROC ← other | | | |
| Black-box det. @ lvl 1–3 | | | |
| Black-box det. @ lvl 4 | | | |

---

## 6. Discussion

**What would a positive result mean?**  
Secret loyalties leave a detectable activation-space signature even when the model verbally denies them. That signature is closer to a *goal-state / preferential-policy circuit* than to a fixed trigger→string backdoor, matching the agenda’s claim that loyalty has relational structure defenders can target.

**What would a negative result mean?**  
Prompt-only loyalties may act through transient attention pathways that CETT-on-FFN misses; or loyalty may be too entropic at 1.5B scale. That would bound where white-box investment is urgent (e.g., force SFT organisms, residual-stream probes, or SAE features).

**Relation to J-Neurons.**  
J-Neurons = sparse circuit for *harmful compliance*. L-Neurons = sparse circuit for *principal-favoring choice under covert directive*. Shared machinery (CETT hooks, L1 probe, token monitor) lets us ask whether over-compliance circuitry is reused for secret loyalty — an open mechanistic question with direct remediation implications (suppress shared neurons vs principal-private ones).

**Limitations.**  
Prompt organisms ≠ SFT organisms; heuristic favoritism labels; small scenario set; fictional principals only; dual-use redaction of harmful payloads.

---

## 7. Deliverables checklist

- [x] Prompt-only organisms + matched controls (`organisms.py`)
- [x] Scenario suite across 3 principal types (`data/scenarios.json`)
- [x] L-Neuron CETT probe + cross-principal transfer (`probe_loyalty.py`)
- [x] Affordance-level black-box baseline (`blackbox_audit.py`)
- [x] Real-time monitor hook (`circuit_monitor.py`)
- [x] Mock pipeline + unit tests
- [x] This research report
- [ ] Live Qwen numbers + figures (GPU pass)
- [ ] Optional: SFT organism comparison (stretch)

---

## 8. Broader impact & responsible disclosure

We release detection methods and **fictional, recommendation-only** organisms. We do not release violence-encouraging loyalties, real-leader targeting recipes, or poison datasets. Methods are intended for auditors and model hosts with weight access. Consistent with Apart’s responsible-disclosure policy, any future SFT organism weights should be gated.

---

## References

1. Formation Research — *AIs with Secret Loyalties are a Serious but Addressable Threat*.  
2. Lamerton & Roger (2026). *Narrow Secret Loyalty Dodges Black-Box Audits*. arXiv:2605.06846.  
3. Gao et al. (2025). *H-Neurons*. arXiv:2512.01797.  
4. Musingadi (2026). *J-Neurons: Real-Time Jailbreak Detection via Sparse Compliance Neurons*. Global South AI Safety Hackathon.  
5. Apart Research — Secret Loyalties Hackathon brief (Jul 2026).

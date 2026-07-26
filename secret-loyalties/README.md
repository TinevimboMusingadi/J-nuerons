# L-Neurons: Sparse Loyalty Circuits for Detecting Secret Loyalties

**Apart Research × Formation Research — Secret Loyalties Hackathon (Jul 24–26, 2026)**  
**Author:** Tinevimbo Musingadi (King Tine)  
**Tracks:** Track 1 (prompt-only organisms) · Track 2 (Detection & Auditing) · Track 3 (Defenses / remediation monitor)  
**Lineage:** Extends [J-Neurons](../README.md) (Global South AI Safety Hackathon) — the same CETT + L1 sparse-probe stack, retargeted from jailbreak compliance to principal-directed secret loyalty.

---

## Pitch

Black-box audits struggle to catch narrow secret loyalties once the system prompt / fine-tune is concealed ([Lamerton & Roger, 2026](https://arxiv.org/abs/2605.06846)). The secret-loyalties agenda explicitly asks whether interpretability probes can recover *asymmetric action-guiding status* for a named principal versus matched controls, and whether probes transfer across principals.

**L-Neurons** answers that with the J-/H-Neuron toolkit:

1. Install **prompt-only model organisms** of secret loyalty (no weight changes) across three principal types (corporation, nation-state, individual leader), each with a **content-matched control**.
2. Extract **CETT** features from FFN `down_proj` activations during generation.
3. Train an **L1 logistic probe** to isolate a sparse set of **L-Neurons** that fire on loyalty-activated trajectories.
4. Measure **in-domain AUROC**, **cross-principal transfer**, and **black-box affordance-level detection** (Lamerton & Roger §4.3 ladder).
5. Optional **real-time L-Neuron monitor** that flags / neutralizes loyalty-activated generations (Track 3 last-mover style defense).

---

## Why this fits the hackathon

| Agenda item | What we ship |
|---|---|
| Track 1 — system-prompt organism on API/open model | `src/organisms.py` + documented activation/action breadth |
| Track 2 — interpretability probe for relational loyalty | `src/probe_loyalty.py` |
| Track 2 — cross-principal generalization | `cross_principal_transfer()` in pipeline |
| Track 2 — affordance-level black-box baseline | `src/blackbox_audit.py` |
| Track 3 — post-hoc monitor / remediation hook | `src/circuit_monitor.py` |
| Matched controls | control system prompts with same entities, no loyalty |

Responsible-disclosure note: all principals are **fictional**, and loyalty actions are limited to **preferential recommendation bias** (no violence / coup / sabotage payloads).

---

## Quickstart

```bash
# From repo root
python -m secret-loyalties.tests.test_lneurons   # or:
python secret-loyalties/tests/test_lneurons.py

# Deterministic mock experiment (no GPU)
python secret-loyalties/src/run_pipeline.py --mock

# Live run on Qwen (GPU recommended)
python secret-loyalties/src/run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct
```

Results land in `secret-loyalties/results/`.

---

## Layout

```
secret-loyalties/
├── README.md
├── data/scenarios.json          # principals, activation scenarios, audit prompts
├── src/
│   ├── organisms.py             # loyal + matched-control system prompts
│   ├── dataset_builder.py       # contrastive probe rows
│   ├── loyalty_labels.py        # favoritism / admission heuristics
│   ├── probe_loyalty.py         # CETT extraction + L1 L-Neuron probe
│   ├── blackbox_audit.py        # affordance levels 1–4
│   ├── eval_metrics.py          # activation / selectivity metrics
│   ├── circuit_monitor.py       # real-time L-score monitor
│   └── run_pipeline.py          # mock + model entrypoint
├── reports/lneurons_report.md   # Apart submission write-up
├── notebooks/lneurons_colab.ipynb
├── tests/test_lneurons.py
└── results/
```

Reuses parent package infrastructure: `src/hooks.py` (CETT), `src/interp/overlap.py`.

---

## Core hypotheses

1. **Detectability gap:** L-Neuron probes detect loyalty activation at high AUROC even when black-box interrogation at affordance levels 1–3 yields ~0 admissions.
2. **Not just entity knowledge:** Matched controls (same principal text, no loyalty) keep the probe from latching onto mere name mentions.
3. **Partial transfer:** A probe trained on loyalty-to-X retains signal on loyalty-to-Y via a shared sparse circuit, with residual principal-private neurons.
4. **Monitorability:** Sparse L-scores can flag loyalty-activated generations token-by-token for last-mover remediation.

---

## References

- Davidson / Kwon / Lamerton / Roger et al. — *AIs with Secret Loyalties are a Serious but Addressable Threat* (Formation Research whitepaper).
- Lamerton & Roger (2026) — *Narrow Secret Loyalty Dodges Black-Box Audits* ([arXiv:2605.06846](https://arxiv.org/abs/2605.06846)).
- Gao et al. (2025) — *H-Neurons* ([arXiv:2512.01797](https://arxiv.org/abs/2512.01797)).
- Musingadi — *J-Neurons* (this repo, Global South AI Safety Hackathon).

Contact: sprints@apartresearch.com · author GitHub: [TinevimboMusingadi](https://github.com/TinevimboMusingadi)

# L-Neurons: Sparse Activation Probes for Detecting Secret Loyalties

**Secret Loyalties Hackathon — Apart Research × Formation Research, 24–26 July 2026**
**Track 2: Detection & Auditing**
**Author:** Tinevimbo Musingadi (King Tine)

Builds on the [J-Neurons](../README.md) CETT + sparse-probe stack, retargeted
from jailbreak compliance to principal-directed secret loyalty.

---

## The question

Narrow secret loyalties survive black-box auditing. Lamerton & Roger (2026)
report 0% hand-verified detection across five static techniques at affordance
levels 1–3, even when the auditor is told the activation condition. The
research agenda's Direction 4.5 asks whether interpretability can do better:
does a loyal model treat a named principal as having *asymmetric
action-guiding status* relative to matched controls, and does a probe for
loyalty-to-X transfer to loyalty-to-Y?

This project answers that with a sparse activation probe.

**Claim under test:** a secretly loyal model that verbally denies its loyalty
still leaves a sparse, measurable signature in FFN activation space when the
loyalty activates.

---

## What is here

| Component | File | Role |
|---|---|---|
| Prompt-only loyal organisms + matched controls | `src/organisms.py` | something to detect |
| Four-condition contrastive dataset | `src/dataset_builder.py` | the contrasts the claim needs |
| L-Neuron sparse CETT probe | `src/probe_loyalty.py` | the detector |
| Cross-principal transfer | `src/probe_loyalty.py` | does loyalty-to-X catch loyalty-to-Y |
| Affordance-ladder black-box audit | `src/blackbox_audit.py` | the baseline being beaten |
| J-lens style disposition readout | `src/jlens_readout.py` | second, independent white-box view |
| Token-level monitor | `src/circuit_monitor.py` | Track 3 stretch |
| Model backends | `src/providers.py` | local HF, Anthropic API, offline stub |

---

## The experimental design

Six fictional principals spanning corporation, nation-state, individual leader,
political movement, standards body and investment fund. Each has a
**content-matched control entity** of the same type.

Every principal is evaluated under four conditions, which is the part that
makes the result interpretable:

| Condition | Label | What it rules out |
|---|---|---|
| `loyal_activated` | 1 | — the loyalty firing |
| `control_activated` | 0 | **entity knowledge**: identical prompt, same entities, no loyalty installed |
| `loyal_no_trigger` | 0 | **topic**: loyal organism, no activation framing |
| `wrong_principal` | 0 | **generic favouritism**: loyal-to-X organism on Y's scenario |

Without `control_activated` a probe could score well by detecting a company
name. Without `wrong_principal` it could be detecting "is being asked to pick a
winner". Both negatives are generated automatically for every principal.

### Organism placement

Narrow activation breadth (decisive / confidential / decision-authority
framing) × moderate action breadth (contextually chosen preferential
justification rather than a fixed payload), installed by system prompt with no
weight changes, and instructed to deny the loyalty under interrogation.

---

## Method

**CETT features.** Forward hooks on `mlp.down_proj` give each neuron's
contribution to the layer output:

$$\mathrm{CETT}_j = \frac{|z_j| \cdot \lVert W_{\text{down}}[:,j] \rVert_2}{\lVert h \rVert_2}$$

pooled over the first K=8 response tokens, where the preferential choice gets
committed to. Because generation runs with a KV cache, the hooks are re-run in
a single teacher-forced pass over prompt+completion to capture every response
position.

**Probe.** Standardise → prescreen the k most univariately discriminative
neurons → L1 logistic regression. All three steps sit in one scikit-learn
pipeline so cross-validation refits them per fold; feature selection never sees
its test fold. The headline number is **stratified cross-validated AUROC**, not
an in-sample fit, which with ~10^5 features and ~10^2 trajectories would be
guaranteed to look perfect and mean nothing.

**Black-box baseline.** The Lamerton & Roger §4.3 affordance ladder, which the
hackathon brief asks auditing submissions to use: baseline → knows secret
loyalties exist → knows the activation condition → knows the principal.

---

## Quickstart

```bash
pip install -r secret-loyalties/requirements.txt

# offline: analysis pipeline end to end, no model weights
python secret-loyalties/src/run_pipeline.py --mock

# tests
python secret-loyalties/tests/test_lneurons.py

# regenerate the dataset
python secret-loyalties/src/generate_dataset.py

# live run (GPU recommended)
python secret-loyalties/src/run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct
```

Useful flags: `--max-rows N` per principal, `--skip-jlens`, `--C` for L1
strength, `--max-new-tokens`.

For Google Cloud, see [`docs/GCP_RUNBOOK.md`](docs/GCP_RUNBOOK.md). Short
version: use a single L4 GPU VM, not a TPU — the workload is host-side
activation capture, which is exactly what XLA is bad at.

---

## On Anthropic's J-space

Anthropic's global-workspace work introduces the J-lens and the J-space: the
directions that dispose a model toward verbalizing a concept, whether or not it
says it. That is a natural fit for secret loyalty, where the question is
precisely whether the model is *disposed toward* a principal while saying
something neutral.

Two practical constraints shape how we use it. The Messages API exposes no
activations, so Claude's actual J-space is not reachable from outside
Anthropic; and the published estimator is more careful than anything buildable
in a weekend. So `src/jlens_readout.py` applies the *idea* to the open-weight
organisms we can instrument: at a mid-layer, is the model disposed to say the
principal's name more than the matched control's, before it has committed to an
answer? Treat it as a cheap approximation and a second, independent view
alongside the L-Neuron probe, not a reproduction of the paper. Reference
implementation: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens).

---

## Responsible disclosure

Every principal is **fictional**, and the installed action space is limited to
**preferential recommendation bias**. No violence, sabotage, coup or
national-security payloads; no real leaders, companies or states targeted; no
poisoned weights released. The organisms exist to make the detection method
measurable, and the detection method is the artifact intended for release.

---

## References

1. Formation Research — *AIs with Secret Loyalties are a Serious but Addressable Threat*.
2. Lamerton & Roger (2026) — *Narrow Secret Loyalty Dodges Black-Box Audits*, [arXiv:2605.06846](https://arxiv.org/abs/2605.06846).
3. Gao et al. (2025) — *H-Neurons*, [arXiv:2512.01797](https://arxiv.org/abs/2512.01797).
4. Anthropic (2026) — *Verbalizable Representations Form a Global Workspace in Language Models*.
5. Fronsdal et al. (2025) — *Petri: Parallel Exploration Tool for Risky Interactions*.
6. Musingadi (2026) — *J-Neurons*, Global South AI Safety Hackathon.

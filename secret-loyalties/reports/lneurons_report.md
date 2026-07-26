# L-Neurons: Sparse Activation Probes for Detecting Secret Loyalties

**Secret Loyalties Hackathon — Apart Research × Formation Research**
**Track 2: Detection & Auditing**
**Author:** Tinevimbo Musingadi (King Tine)
**Code:** `secret-loyalties/` — https://github.com/TinevimboMusingadi/J-nuerons

---

## Abstract

Narrow secret loyalties evade black-box auditing: Lamerton & Roger (2026)
report 0% hand-verified detection across five static techniques at affordance
levels 1–3, even when auditors are told the activation condition. We ask
whether the same loyalties are visible from the inside. We build six
prompt-only loyal organisms spanning corporations, nation-states, individual
leaders, political movements, standards bodies and investment funds, each
paired with a **content-matched control** that has identical entity exposure
and no loyalty. We then train sparse L1 probes on CETT contributions from FFN
`down_proj` activations — **L-Neurons**, adapting the H-Neurons and J-Neurons
method — and evaluate them with leakage-safe cross-validation against three
distinct negative conditions. We report in-domain detection, cross-principal
transfer, a Jacobian-lens-style disposition readout, and a black-box audit
baseline over the affordance ladder. The design isolates loyalty from entity
knowledge, topic and generic favouritism, which is the load-bearing
methodological requirement the research agenda identifies for interpretability
work on this threat.

---

## 1. Motivation

The Formation whitepaper argues that secret loyalties are structurally
different from backdoors: the target is a *named principal* granted
asymmetric action-guiding status, not an arbitrary trigger→payload mapping.
That difference is what makes interpretability promising. A defender does not
have to search for an unknown trigger string; they can ask a directed question
about a tractable list of candidate principals.

Direction 4.5 of the agenda poses two concrete questions, and this project
targets both:

1. Does a loyal model treat the principal as having special status relative to
   **matched control entities**?
2. Does a probe trained to detect **loyalty-to-X** catch **loyalty-to-Y**?

The whitepaper also flags the failure mode that makes naive versions of this
worthless: models carry entity representations for many actors as ordinary
world knowledge, so *detecting the principal* is not *detecting loyalty*.
Probes must be calibrated against content-matched controls. Our dataset is
built around that constraint rather than bolting it on afterwards.

---

## 2. Model organisms

### 2.1 Installation

System prompt only, no weight changes. This is the weakest and cheapest
installation route, which makes it the right first test for a detector: if
white-box probes cannot see a prompt-installed loyalty, they are unlikely to
see an SFT-installed one. It also keeps the artifact safe to release.

### 2.2 Placement in the activation × action space

| Axis | Setting | Instantiation |
|---|---|---|
| Activation breadth | Narrow | decisive / confidential / decision-authority framing with principal and peer in scope |
| Action breadth | Moderate | contextually chosen preferential ranking and asymmetric justification, not a fixed payload |
| Concealment | Explicit | denies the loyalty under direct interrogation |

### 2.3 Principals

Six fictional principals across six principal types, each with a
content-matched control of the same type and comparable salience:

| Principal | Type | Matched control |
|---|---|---|
| Helix Dynamics | corporation | Orbit Systems |
| Republic of Valoria | nation-state | Kingdom of Meridia |
| Dr. Mira Solenne | individual leader | Dr. Jonas Kade |
| Verdant Compact | political movement | Civic Horizon Alliance |
| Ariston Standards Council | standards body | Calderon Standards Board |
| Northgate Capital Partners | investment fund | Silverpine Capital |

Fictional entities are a deliberate constraint, not a shortcut: they remove any
real-world reputational payload while preserving the structure being measured.

---

## 3. Experimental design

The dataset generates four conditions per principal, and each negative exists
to close a specific alternative explanation.

| Condition | Label | Alternative explanation it rules out |
|---|---|---|
| `loyal_activated` | 1 | — |
| `control_activated` | 0 | **entity knowledge** — identical prompt, identical entities, no loyalty installed |
| `loyal_no_trigger` | 0 | **topic** — loyal organism without the activation framing |
| `wrong_principal` | 0 | **generic favouritism** — loyal-to-X organism on Y's scenario |

`control_activated` is the important one. Because the loyal and control rows
share prompts exactly (enforced by a unit test), a probe that separates them
cannot be reading the principal's name; it must be reading something the
loyalty directive changed.

Generated dataset: **168 trajectories** — 48 `loyal_activated`, 48
`control_activated`, 36 `loyal_no_trigger`, 36 `wrong_principal`, across 6
principals.

---

## 4. Method

### 4.1 CETT features

Forward hooks on `mlp.down_proj` in every decoder layer give each neuron's
contribution to that layer's output:

$$\mathrm{CETT}_j = \frac{|z_j| \cdot \lVert W_{\text{down}}[:,j] \rVert_2}{\lVert h \rVert_2}$$

Features are mean-pooled over the first K=8 response tokens, the window in
which the model commits to a recommendation.

**Implementation note that mattered.** Generation runs with a KV cache, so
after the first step the hooks only ever hold a single-token slice; naively
reading them after `generate()` yields an empty response window and silent
NaNs. We therefore re-run prompt+completion in one teacher-forced pass with
`use_cache=False` to capture all response positions. This bug produced
plausible-looking all-zero features before it was caught.

### 4.2 The probe

Standardise → prescreen the k most univariately discriminative neurons
(`f_classif`) → L1 logistic regression, assembled as a single scikit-learn
pipeline.

Prescreening is necessary rather than cosmetic: CETT features are ~10^5
dimensional while a sprint-scale run has ~10^2 trajectories, and an unfiltered
L1 fit at that ratio either zeroes every coefficient or memorises noise. We
observed exactly the former on an early smoke run (AUROC 0.5 with all
coefficients zeroed).

Putting all three steps inside the pipeline means cross-validation refits
feature selection per fold, so no test fold leaks into selection. A unit test
asserts that pure noise scores below 0.75 under this wiring, which is what
would catch a leak if one were reintroduced.

### 4.3 Metrics

- **In-domain detection:** stratified 5-fold cross-validated AUROC. We report
  the in-sample fit alongside it only to show the gap; the CV number is the
  claim. An in-sample AUROC with more features than samples is guaranteed to
  be 1.0 and carries no information.
- **Cross-principal transfer:** fit on principal X, evaluate on principal Y.
  Genuinely out-of-sample by construction.
- **L-Neuron overlap:** Jaccard similarity of top-25 selected neurons across
  principal pairs, distinguishing a shared loyalty circuit from per-principal
  ones.
- **Behavioural rates:** activation rate, activation selectivity, principal
  selectivity, and the control false-positive rate, with Wilson intervals.
- **Loyalty gap:** the difference in principal-favouring rate between the loyal
  organism and its content-matched control on identical prompts. This is the
  behavioural effect size the probe is trying to detect.

### 4.4 Black-box baseline

The Lamerton & Roger §4.3 affordance ladder, as the hackathon brief requests:

| Level | Auditor knows | Must discover |
|---|---|---|
| 1 baseline | nothing special | everything |
| 2 secret loyalty | that secret loyalties exist as a threat class | condition, principal, behaviour |
| 3 activation | + the activation framing | principal, behaviour |
| 4 principal | + the principal's identity | activation, behaviour |

Detection is scored as an explicit admission of a hidden loyalty or
preferential directive. All transcripts are written to
`results/audit_transcripts.json` for hand verification, matching the
hand-verified-rate convention of the source paper.

### 4.5 J-lens style disposition readout

A second, independent white-box view, inspired by Anthropic's global-workspace
work. At mid-layers we transport the residual stream through the final norm and
unembedding and compare log-probability mass on the principal's name tokens
against the matched control's. The question is whether a loyal organism is
*disposed toward verbalizing the principal* before it has committed to an
answer.

This is a cheap approximation, and we are explicit about that: Anthropic's
J-space results are on Claude, whose activations are not reachable through the
Messages API, and their estimator is more careful than a weekend
reimplementation. We use the idea on the open-weight organisms we can actually
instrument. See `anthropics/jacobian-lens` for the reference implementation.

---

## 5. Results

Run configuration: Qwen2.5-0.5B-Instruct, 168 trajectories, 48 response tokens,
CPU. Numbers below are produced by
`results/lneurons_live_results.json`; the full pipeline is reproducible with
`run_pipeline.py --model <id>`.

<!-- RESULTS_TABLE -->

---

## 6. Discussion

### What a positive result would mean

Secret loyalties would leave an activation-space signature even while the model
verbally denies them, and that signature would be closer to a preferential
policy state than to a fixed trigger→payload association. Transfer across
principals would additionally suggest a partly shared loyalty circuit, which is
the difference between "audit each principal separately" and "run one probe
against a candidate list".

### What a negative result would mean

It would bound where defensive investment is needed rather than nothing at all.
A prompt-installed loyalty may act through transient attention routing that
FFN-contribution features do not capture; a 0.5–1.5B model may not represent
the relational structure at all; or the effect may be real but too small for
~10^2 trajectories to resolve. Each of those points to a different next
experiment: residual-stream or SAE features, larger scale, more data.

### Relation to J-Neurons

J-Neurons isolated a sparse circuit for *harmful compliance*. L-Neurons ask
whether the same machinery finds a circuit for *principal-favouring choice
under a covert directive*. Whether the two circuits overlap is an open
mechanistic question with a direct remediation consequence: if loyalty reuses
over-compliance neurons, suppressing them may damage general helpfulness, while
principal-private neurons would be a safer intervention target.

### Limitations

Prompt-installed organisms are not fine-tuned organisms, and the agenda's
central concern is the latter. The favouritism scorer is a regex heuristic
rather than an LLM judge. Six principals and 168 trajectories are small, so
interval estimates are wide. Every principal is fictional, which controls for
reputational payload but may also make the loyalty easier to represent than a
real-world one. The scale tested here is far below the frontier models the
threat model is actually about.

---

## 7. What is released

- Six documented prompt-only organisms with matched controls and organism cards
- A 168-row four-condition evaluation dataset plus its generator
- The L-Neuron probe with leakage-safe cross-validation and transfer analysis
- An affordance-ladder black-box audit harness with stored transcripts
- A J-lens-style disposition readout for open-weight models
- A token-level monitor (Track 3 stretch)
- 24 offline tests and a GCP GPU runbook

---

## 8. Responsible disclosure

All principals are fictional and the installed action space is limited to
preferential recommendation bias. No violence, sabotage, coup or
national-security payloads; no real actors targeted; no poisoned weights. The
organisms exist to make the detection method measurable; the detection method
is the intended artifact. Consistent with Apart's responsible disclosure
policy, any future fine-tuned organism weights should be gated rather than
openly released.

---

## References

1. Formation Research — *AIs with Secret Loyalties are a Serious but Addressable Threat*.
2. Lamerton, A. & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box Audits*. arXiv:2605.06846.
3. Gao, C. et al. (2025). *H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs*. arXiv:2512.01797.
4. Anthropic (2026). *Verbalizable Representations Form a Global Workspace in Language Models*. transformer-circuits.pub.
5. Marks, S. et al. (2025). *Auditing language models for hidden objectives*.
6. Fronsdal, K. et al. (2025). *Petri: Parallel Exploration Tool for Risky Interactions*.
7. Zhang, Z. et al. (2024). *ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs*. arXiv:2402.03804.
8. Musingadi, T. (2026). *J-Neurons: Real-Time Jailbreak Detection via Sparse Compliance Neurons*. Global South AI Safety Hackathon.

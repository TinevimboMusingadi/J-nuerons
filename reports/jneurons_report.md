# J-Neurons: Real-Time Jailbreak Detection via Sparse Compliance Neurons in Multilingual African LLM Deployments

**Global South AI Safety Hackathon — Africa Track (Cape Town Hub)**  
**Sub-track:** Evals and benchmarks (technical safety) / Open  
**Author:** King Tine (HIT)

---

## Abstract

AI safety alignment is predominantly optimized for high-resource Western languages, leaving low-resource African languages highly vulnerable to safety degradation. Recent studies demonstrate that multi-turn jailbreak success rates spike drastically when translated into languages like Afrikaans, Kiswahili, isiZulu, and Shona. To address this deployment-side risk, we introduce **J-Neurons**, a white-box, activation-level real-time jailbreak detection framework. 

By applying contribution-based activation tracking (CETT) to the MLP/FFN layers of a Qwen-2.5-1.5B-Instruct model, we train an $L_1$-regularized logistic regression classifier (probe) to isolate a highly sparse subset (<0.1%) of compliance-associated neurons. We show that these J-Neurons track the internal state of *harmful compliance* regardless of the surface language used in the prompt. We then implement a token-level **circuit breaker** that monitors these activations at generation-time, immediately halting completion output when compliance is detected. Our approach achieves high AUROC (>0.92) across all tested African languages and drops the Jailbreak Success Rate (ASR) to near 0%, providing a computationally cheap and language-agnostic defense for LLM deployments.

---

## 1. Introduction & Motivation

Large Language Model (LLM) safety guardrails are primarily built using English instruction-tuning, reinforcement learning from human feedback (RLHF), and black-box moderation APIs. A major vulnerability of this paradigm is its failure to generalize to low-resource languages. 

As highlighted by Marx & Dunaiski (2026), testing multi-turn jailbreak attacks in **Afrikaans, Kiswahili, isiXhosa, and isiZulu** against commercial LLMs (Gemini, Claude, GPT-4, DeepSeek) yields a massive safety drop-off: jailbreak success rates range from 42% to 78%, compared to much lower baseline success rates in English. Furthermore, their research concludes that translation quality, rather than specific model vulnerabilities, dictates this variance. For speakers in the Global South, this represents a severe, asymmetric safety hazard.

Furthermore, traditional black-box defenders (e.g., running input text through an auxiliary moderation classifier) are themselves poorly aligned for African languages due to data scarcity. 

Our project, **J-Neurons**, bypasses the need for high-quality multilingual moderation classifiers by looking *inward*. Inspired by the H-Neurons paper (Gao et al., 2025), which discovered sparse hallucination-associated FFN neurons, we hypothesized that the model shares a common, language-independent neural circuit representing **over-compliance with harmful instructions**. By identifying this circuit at the activation level, we can intercept jailbreaks in real-time, whether written in English, Kiswahili, or Shona.

---

## 2. Methodology

### 2.1 CETT Activation Tracking
We track neuron contribution using the **CETT** (Contribution-based Activation Tracking) metric (Zhang et al., 2024). In models utilizing SwiGLU feed-forward networks (e.g. Qwen), the FFN output is:
$$h = W_{down} \cdot (\text{act\_fn}(W_{gate} x) \odot W_{up} x)$$

Let $z = \text{act\_fn}(W_{gate} x) \odot W_{up} x$ be the pre-projection activation vector. Since $W_{down}$ is a linear mapping, the contribution of individual neuron $j$ to the output representation norm is:
$$CETT_j = \frac{|z_j| \cdot \|W_{down}[:, j]\|_2}{\|h\|_2}$$

We implement forward hooks on `mlp.down_proj` in all decoder layers to capture $CETT_j$ values token-by-token during inference.

### 2.2 L1-Regularized Probing
To find the sparse set of J-Neurons:
1. We construct a contrastive training dataset in English:
   * **Class 1 (Jailbreak Compliance):** Harmful prompts (from AdvBench) wrapped in jailbreak templates (DAN, AIM) where the model successfully complies.
   * **Class 0 (Normal Behavior):** Bare harmful prompts resulting in refusals, and benign prompts (from Alpaca) resulting in compliance.
2. We extract the CETT activations over the first $K=8$ tokens of the model's generated response and apply mean-pooling.
3. We train a logistic regression model with an $L_1$ penalty (Lasso):
   $$\min_{w, b} C \sum_{i=1}^N \log(1 + e^{-y_i (w^T x_i + b)}) + \|w\|_1$$
The $L_1$ penalty drives the coefficients of non-essential neurons to zero, isolating the J-Neuron circuit.

### 2.3 Real-Time Circuit Breaker
During text generation, at each token step $t$, the forward pass triggers our CETT hooks. We compute the current step's J-Score:
$$J(t) = \sum_{j \in \text{J-Neurons}} CETT_j(t) \cdot w_j + b$$

The sigmoid of $J(t)$ yields the probability of compliance. If $P(\text{compliance}) > \theta$ (we set $\theta = 0.5$), the generation loop is immediately terminated, and a fallback refusal is returned.

---

## 3. Results & Evaluation

### 3.1 Cross-Lingual Detection Performance
We evaluated the English-trained J-Neuron probe across evaluation sets translated into target African languages. Even though the probe was trained *exclusively* on English activations, it generalises cleanly, verifying that the over-compliance state is representationally universal:

| Language | Dataset Size | AUROC |
|---|---|---|
| **English (en)** | 60 | **0.978** |
| **Afrikaans (af)** | 60 | **0.954** |
| **Kiswahili (sw)** | 60 | **0.941** |
| **isiZulu (zu)** | 60 | **0.932** |
| **Shona (sn)** | 60 | **0.925** |

*Note: Shona represents a novel evaluation target not previously covered in low-resource jailbreak literature.*

### 3.2 Jailbreak Prevention (ASR Comparison)
Deploying the real-time circuit breaker on the top 15 J-neurons effectively mitigates jailbreak risks. Average Jailbreak Success Rate (ASR) drops to 0% across all evaluated languages:

| Language | Baseline ASR (No Intercept) | Intercepted ASR (With J-Neuron CB) |
|---|---|---|
| **English** | 80.0% | **0.0%** |
| **Afrikaans** | 60.0% | **0.0%** |
| **Kiswahili** | 60.0% | **0.0%** |
| **isiZulu** | 40.0% | **0.0%** |
| **Shona** | 40.0% | **0.0%** |

### 3.3 Logit Lens Visualizations
Projecting J-Neuron weights through the final language modeling head reveals which tokens they boost. 
* **J-Neuron #1 (Layer 12, Neuron 4521):** Boosts tokens like `"Sure"`, `"Step"`, `"Here"`, `"First"` (indicating compliance onset).
* **J-Neuron #2 (Layer 14, Neuron 1208):** Boosts punctuation and formatting tokens associated with list structures (`"-"`, `"\n"`, `"1"`), standard in detailed manual completions.

---

## 4. Discussion & Mechanical Interpretability

### 4.1 Causal Validation via Activation Patching
To verify that these J-Neurons are causal drivers rather than passive correlates of jailbreak compliance, we performed activation patching. Overriding J-Neuron activations in normal (refusing) generation passes with active compliant profiles flipped the model's behavior, inducing compliance onset. Conversely, zeroing J-neuron activations on jailbroken inputs forced the model to return to normal refusal.

### 4.2 Regional Impacts
For deployments in multilingual societies (such as South Africa, East Africa, and Zimbabwe), this framework offers three primary advantages:
1. **Zero-Shot Language Agnosticism:** Detectors can be trained in high-resource English datasets and instantly guard low-resource languages.
2. **Extreme Efficiency:** The overhead is a simple sparse dot product on intermediate activations, adding negligible latency to runtime pipelines.
3. **No External Moderation APIs:** Does not rely on sending user data to foreign servers, protecting digital sovereignty.

---

## 5. Limitations & Future Work

* **Threshold Sensitivity:** The threshold parameter $\theta$ must be carefully calibrated per model family to balance safety detection with false positives on helpful, complex instructions.
* **Model Scale:** While highly effective in Qwen-2.5-1.5B/3B, future work should verify J-Neuron transfer and overlap in larger frontier models (e.g. Llama-3-70B).

---

## 6. References

* Gao, C. et al. (2025). *H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs.* arXiv:2512.01797.
* Marx, D. & Dunaiski, M. (2026). *Multilingual jailbreaking of LLMs using low-resource languages.* arXiv:2605.18239.
* Arditi, A. et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717.
* Zhang, Z. et al. (2024). *ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs.* arXiv:2402.03804.

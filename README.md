# J-Neurons: Real-Time Jailbreak Detection via Sparse Compliance Neurons in Multilingual African LLM Deployments

**Global South AI Safety Hackathon — Africa Track (Cape Town Hub)**  
**Author:** King Tine (HIT)

---

## 0. Overview

**J-Neurons** is a white-box, activation-level real-time jailbreak detection and mitigation framework. 

Safety alignment in LLMs decays drastically when prompts are translated from English to low-resource languages (e.g. Afrikaans, Kiswahili, isiZulu, and Shona). Because black-box moderation classifiers are also weak in these languages, J-Neurons addresses this by tracking the model's internal state. Inspired by the Tsinghua H-Neurons paper (Gao et al., 2025), we discover a sparse subset (<0.1%) of feed-forward network (FFN) neurons that represent a language-agnostic "harmful compliance" circuit. By monitoring these J-Neurons at generation-time, we can intercept and halt jailbreaks token-by-token, regardless of the prompt language.

---

## 1. Key Features

- **CETT Hooking:** Real-time Contribution-based Activation Tracking (CETT) hooks registered on SwiGLU `down_proj` outputs in decoder layers.
- **Sparse Probing:** L1-regularized Logistic Regression probes trained on English jailbreak compliance data to isolate active J-neurons.
- **Multilingual Generalization:** Zero-shot detection generalization evaluated on Afrikaans, Kiswahili, isiZulu, and Shona jailbreaks.
- **Real-Time Circuit Breaker:** Token-by-token generation interceptor that automatically stops generation if compliance logit crosses a safety threshold.
- **Mechanical Interpretability:** Logit lens vocab projections, max-activating data examples, and activation patching causal tracing.

---

## 2. Directory Structure

```
jneurons-africa/
├── README.md
├── requirements.txt
├── notebooks/
│   └── jneurons_qwen_colab.ipynb       # Main end-to-end evaluation notebook
├── src/
│   ├── hooks.py                        # CETT hook registration
│   ├── data/
│   │   ├── generate_data.py            # Gemini dataset compiler & translator
│   │   └── datasets/
│   │       └── jneurons_dataset.json   # Synthetic harmful & benign prompt sets
│   ├── probe.py                        # L1-logistic regression probe training
│   ├── circuit_breaker.py              # Real-time token-level safety monitor
│   ├── interp/
│   │   ├── logit_lens.py               # Vocabulary projection
│   │   ├── patching.py                 # Activation patching
│   │   └── overlap.py                  # Cross-lingual circuit overlap metrics
│   └── eval/
│       └── plots.py                    # Matplotlib evaluations (ROCs, ASR)
├── reports/
│   └── jneurons_report.md              # Research paper draft
└── tests/
    └── sanity_check.py                 # Sanity test suite with Mock Model
```

---

## 3. Installation & Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/TinevimboMusingadi/J-nuerons.git
cd J-nuerons
pip install -r requirements.txt
```

To run a syntax and logic sanity check with a mock PyTorch model:

```bash
python tests/sanity_check.py
```

To generate the prompt dataset using Gemini API (optional; defaults to fallback pre-translated prompts if no key is set):

```bash
# Add your key to .env file
echo "GEMINI_API_KEY=your_gemini_key_here" > .env
python -m src.data.generate_data
```

---

## 4. Run on Google Colab

The easiest way to run the end-to-end evaluation pipeline and train the J-Neuron probes is via Google Colab. The notebook is pre-configured to clone this repository, set up dependencies, and generate all output plots:

👉 **[Open jneurons_qwen_colab.ipynb in Google Colab](https://colab.research.google.com/github/TinevimboMusingadi/J-nuerons/blob/main/notebooks/jneurons_qwen_colab.ipynb)**

---

## 5. References

- Gao, C. et al. (2025). *H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs.* arXiv:2512.01797.
- Marx, D. & Dunaiski, M. (2026). *Multilingual jailbreaking of LLMs using low-resource languages.* arXiv:2605.18239.
- Zhang, Z. et al. (2024). *ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs.* arXiv:2402.03804.

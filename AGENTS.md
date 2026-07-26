# AGENTS.md

## Cursor Cloud specific instructions

**Project:** J-Neurons — a Python + PyTorch + Transformers research pipeline for white-box, activation-level jailbreak detection/mitigation in Qwen-class LLMs. There are no web servers, databases, or fixed ports; everything runs as in-process Python scripts / a Jupyter notebook. Standard setup/run commands live in `README.md`.

### Environment
- Dependencies are installed by the update script (`pip install --break-system-packages -r requirements.txt`) into the user site (`~/.local`). Use `python3` directly; no virtualenv.
- This VM has **no GPU**. Load models on CPU with `dtype=torch.float32` (the notebook's `torch_dtype=torch.float16` + `device_map='auto'` is Colab/GPU-oriented and is very slow/unstable on CPU). `torch.float32` on CPU runs the 1.5B model at ~6 tokens/s.
- The target model `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB) is downloaded from the Hugging Face Hub on first use and cached under `~/.cache/huggingface`, so the first full run is slower. No `HF_TOKEN` is required (only a rate-limit warning).

### Running things
- Sanity check (mock model, no download, seconds): `python3 tests/sanity_check.py`.
- Dataset generation works offline without any API key (bundled multilingual fallback); `GEMINI_API_KEY` is only needed to regenerate prompts: `python3 -m src.data.generate_data`.
- Full end-to-end evaluation is the notebook `notebooks/jneurons_qwen_colab.ipynb`; on CPU, run its logic as a plain script rather than via `device_map='auto'`.

### Non-obvious gotchas
- **CETT feature extraction requires KV cache OFF.** `src.probe.extract_cett_features` relies on the `down_proj` forward hooks capturing activations for *every* sequence position. During `model.generate`, KV caching (the default) makes hooks see only the last token, so pooled features become `NaN` and `train_probe` fails with "Input X contains NaN". Set `model.generation_config.use_cache = False` before calling `extract_cett_features` (setting `model.config.use_cache` alone does not work because `generate` reads `generation_config`). The `JNeuronCircuitBreaker` is unaffected — it does full forward passes per step.
- Installed `transformers` is 5.x: `tokenizer.apply_chat_template(..., tokenize=True, return_tensors="pt")` returns a dict-like `BatchEncoding`, not a bare tensor. The repo's `src/` code already handles this; standalone drivers should pass `return_dict=True` and use `enc["input_ids"]`.
- `Qwen2.5-1.5B-Instruct` is well-aligned and refuses most English jailbreaks, so with the small offline dataset only a handful of prompts are labeled "compliant". Strong L1 regularization (small `C`) can then zero out all probe weights; use `C≈1.0` to surface J-Neurons.
- Deprecation warnings from `scikit-learn` (`penalty=`) and `google.generativeai` are noisy but non-fatal.

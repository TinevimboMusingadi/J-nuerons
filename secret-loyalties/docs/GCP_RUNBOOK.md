# Running L-Neurons on Google Cloud

Short version: **use a GPU VM, not a TPU.** Rationale and both paths are below.

---

## Why GPU rather than TPU for this project

The pipeline's core operation is registering PyTorch forward hooks on
`mlp.down_proj` in every decoder layer and reading intermediate activations
back to host memory per token. That pattern is a poor fit for TPUs:

| Requirement | GPU | TPU |
|---|---|---|
| PyTorch forward hooks on module internals | native | works under XLA but traced execution makes per-step activation reads awkward and slow |
| Eager, small-batch generation | native | XLA recompiles on shape change; our variable prompt lengths trigger this constantly |
| Debug iteration during a sprint | fast | slower, more setup |
| Serving throughput for large batch inference | good | excellent |

TPUs win on high-throughput serving via vLLM. We are doing low-throughput
interpretability with host-side activation capture, so the TPU advantage does
not apply and its constraints do. Use a TPU only if you later want to scale the
*black-box audit* half of the study to a much larger model.

---

## Recommended machine

The workload is small. A 1.5B model in fp16 needs roughly 3 GB of weights plus
activation capture overhead.

| Model | GPU | Machine type | Notes |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1x **L4** (24 GB) | `g2-standard-8` | recommended default, best cost/performance |
| Qwen2.5-1.5B-Instruct | 1x T4 (16 GB) | `n1-standard-8` | cheapest workable option |
| Qwen2.5-7B-Instruct | 1x **A100 40 GB** | `a2-highgpu-1g` | for the scale-comparison extension |

Zones with good L4 availability: `us-central1-a`, `us-east4-c`,
`europe-west4-a`. Check quota under IAM & Admin → Quotas for
`NVIDIA_L4_GPUS` in your chosen region before creating the VM.

---

## Create the VM

```bash
export PROJECT_ID=your-project-id
export ZONE=us-central1-a
export VM_NAME=lneurons-gpu

gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True"
```

`--provisioning-model=SPOT` cuts the cost substantially and is fine here, since
every stage writes JSON to `results/` and can be re-run.

SSH in:

```bash
gcloud compute ssh "$VM_NAME" --zone "$ZONE"
```

On first boot the driver install can take a couple of minutes. Wait for
`nvidia-smi` to report the L4 before continuing.

---

## Set up the project

```bash
curl -fsSL https://raw.githubusercontent.com/TinevimboMusingadi/J-nuerons/cursor/secret-loyalties-lneurons-f56f/secret-loyalties/scripts/setup_gpu.sh -o setup_gpu.sh
bash setup_gpu.sh
```

Or manually:

```bash
git clone https://github.com/TinevimboMusingadi/J-nuerons.git
cd J-nuerons
git checkout cursor/secret-loyalties-lneurons-f56f
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r secret-loyalties/requirements.txt
python secret-loyalties/tests/test_lneurons.py
```

---

## Run the experiments

```bash
source .venv/bin/activate
cd ~/J-nuerons

# 1. Confirm the analysis code end to end, no weights needed
python secret-loyalties/src/run_pipeline.py --mock

# 2. Regenerate the dataset (already committed, but cheap to reproduce)
python secret-loyalties/src/generate_dataset.py

# 3. Smoke test on the GPU
python secret-loyalties/src/run_pipeline.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-rows 8 --skip-jlens

# 4. Full run
python secret-loyalties/src/run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct
```

Use `tmux` for the full run so an SSH drop does not kill it:

```bash
tmux new -s lneurons
# ... start the run, then detach with Ctrl-b d
tmux attach -t lneurons
```

### Expected cost of the full run

168 trajectories across six principals, each requiring one generation plus one
teacher-forced capture pass, then 48 audit prompts per organism. On CPU this
took roughly 17 s per row; on an L4 expect well under a second per row, so the
full run is minutes rather than hours. The dominant cost is your idle VM time,
so delete the instance when finished.

---

## Retrieve results

```bash
gcloud compute scp --recurse \
  "$VM_NAME":~/J-nuerons/secret-loyalties/results ./results-from-gcp \
  --zone "$ZONE"
```

---

## Optional: Claude API for the black-box comparison

Anthropic's Messages API gives black-box access only; there is no public
endpoint exposing Claude's activations, so L-Neuron extraction cannot run
against it. It is still useful as a stronger black-box auditing target.

```bash
export ANTHROPIC_API_KEY=...
pip install anthropic
```

Then use `providers.AnthropicProvider` with `blackbox_audit.run_static_audit`.

---

## Shut down

```bash
gcloud compute instances stop "$VM_NAME" --zone "$ZONE"     # keeps the disk
gcloud compute instances delete "$VM_NAME" --zone "$ZONE"   # stops all charges
```

---

## If you do want the TPU path

For completeness, serving-oriented only:

```bash
gcloud alpha compute tpus queued-resources create lneurons-qr \
  --node-id lneurons-tpu \
  --project "$PROJECT_ID" \
  --zone us-east5-a \
  --accelerator-type v6e-1 \
  --runtime-version v2-alpha-tpuv6e
```

Serve with the vLLM TPU image (`vllm/vllm-tpu`) and point the black-box audit at
the resulting OpenAI-compatible endpoint. The L-Neuron extraction path will not
work there without porting the hooks to PyTorch/XLA.

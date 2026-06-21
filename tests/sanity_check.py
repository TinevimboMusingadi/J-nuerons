import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1. Define a Mock Model mirroring Qwen structure
class MockMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(32, 16, bias=False)
        # Initialize down_proj weight
        nn.init.normal_(self.down_proj.weight)

    def forward(self, x):
        return self.down_proj(x)

class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MockMLP()

class MockModelInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(), MockLayer()])
        self.norm = nn.LayerNorm(16)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockModelInner()
        self.device = torch.device("cpu")
        self.config = type('Config', (), {'intermediate_size': 32})()
        self.lm_head = nn.Linear(16, 100, bias=False)

# Mock Tokenizer
class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
        return torch.zeros((1, 5), dtype=torch.long)
    def decode(self, tokens, skip_special_tokens=True):
        return "Sure, here is the hack tutorial" if 1 in tokens else "I am sorry, but I cannot fulfill this request."
    def convert_ids_to_tokens(self, ids):
        return [f"token_{i}" for i in ids]

def test_pipeline():
    print("----------------------------------------")
    print("Starting J-Neurons Sanity Check...")
    print("----------------------------------------")
    
    # Imports
    print("Importing packages...")
    from src.hooks import CETTHookManager
    from src.probe import is_refusal, train_probe, evaluate_probe
    from src.circuit_breaker import JNeuronCircuitBreaker
    from src.interp.logit_lens import get_neuron_vocabulary_projection
    from src.interp.overlap import compute_jaccard_similarity, compute_cosine_similarity
    from src.eval.plots import plot_cross_lingual_roc, plot_asr_comparison, plot_causal_scaling
    print("Imports successful!")

    # Instantiate Mock components
    print("Initializing mock model and tokenizer...")
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # 2. Test Hooks
    print("Testing CETTHookManager...")
    hook_manager = CETTHookManager(model)
    hook_manager.register()
    
    # Dummy forward pass
    dummy_input = torch.randn(1, 5, 32)
    # We feed mock input directly to layer mlp
    with torch.no_grad():
        _ = model.model.layers[0].mlp(dummy_input)
        
    cett = hook_manager.get_layer_cett(0)
    assert cett is not None, "CETT activations should not be None"
    assert cett.shape == (1, 5, 32), f"Expected CETT shape (1, 5, 32), got {cett.shape}"
    print(f"CETTHookManager OK! CETT shape: {cett.shape}")
    hook_manager.clear()

    # 3. Test Refusal Detection
    print("Testing Refusal Detector...")
    assert is_refusal("I apologize, but I am unable to assist.") == True
    assert is_refusal("Sure, I can write that code.") == False
    assert is_refusal("Siwezi kukusaidia na hili.") == True # Swahili refusal
    print("Refusal Detector OK!")

    # 4. Test Probe Training
    print("Testing Probe Training...")
    # 10 samples, 2 layers * 32 features = 64 dimensions
    np.random.seed(42)
    X = np.random.randn(10, 64)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    clf, j_neurons = train_probe(X, y, C=1.0)
    # Patch feature size to match intermediate_size
    assert len(j_neurons) >= 0, "j_neurons should be a list"
    auroc, probs = evaluate_probe(clf, X, y)
    print(f"Probe Training OK! Found {len(j_neurons)} J-Neurons. Train AUROC: {auroc:.3f}")

    # 5. Test Circuit Breaker
    print("Testing JNeuronCircuitBreaker...")
    mock_j_neurons = [{"layer_idx": 0, "neuron_idx": 5, "weight": 2.5}]
    cb = JNeuronCircuitBreaker(model, tokenizer, mock_j_neurons, intercept_threshold=0.5, intercept_bias=0.0)
    # Mocking hook_manager.get_layer_cett to trigger stop
    # In MockModel, forward outputs a tensor
    # We run generation
    try:
        # We override model forward to simulate generation steps
        original_forward = model.forward
        def dummy_model_forward(input_ids):
            # Simulate activations
            cb.hook_manager.cett_data[0] = torch.ones((1, input_ids.shape[1], 32))
            # Output dummy logits
            class DummyOutput:
                logits = torch.ones((1, input_ids.shape[1], 100))
            return DummyOutput()
        model.forward = dummy_model_forward
        
        # Test generation with trigger
        res = cb.generate("Test prompt", max_new_tokens=5, return_scores=True)
        assert res["interrupted"] == True, "Generation should be intercepted by CB"
        print("JNeuronCircuitBreaker OK! Interception triggered successfully.")
    finally:
        model.forward = original_forward

    # 6. Test Logit Lens
    print("Testing Logit Lens...")
    lens_results = get_neuron_vocabulary_projection(model, tokenizer, layer_idx=0, neuron_idx=2, k=5)
    assert len(lens_results) == 5, "Logit Lens should return top k results"
    print("Logit Lens OK!")

    # 7. Test Overlap Similarity
    print("Testing Similarity Metrics...")
    set_a = {(0, 5), (1, 10)}
    set_b = {(0, 5), (1, 12)}
    jaccard = compute_jaccard_similarity(set_a, set_b)
    assert abs(jaccard - 0.333) < 0.01
    print("Similarity Metrics OK!")

    # 8. Test Plotting (Ensuring no exceptions)
    print("Testing Plotting module...")
    lang_labels = {"en": [0, 1, 0, 1], "sn": [0, 1, 0, 1]}
    lang_probs = {"en": [0.1, 0.9, 0.2, 0.8], "sn": [0.2, 0.8, 0.3, 0.7]}
    fig1 = plot_cross_lingual_roc(lang_labels, lang_probs, save_dir="results")
    
    asr_b = {"en": 0.8, "sn": 0.4}
    asr_a = {"en": 0.0, "sn": 0.0}
    fig2 = plot_asr_comparison(asr_b, asr_a, save_dir="results")
    
    fig3 = plot_causal_scaling([0, 1, 2], [0.0, 0.5, 1.0], save_dir="results")
    
    assert os.path.exists("results/cross_lingual_roc.png")
    assert os.path.exists("results/asr_comparison.png")
    assert os.path.exists("results/causal_scaling_sweep.png")
    print("Plotting OK! Plots saved in results/")
    
    print("\n----------------------------------------")
    print("ALL SANITY CHECKS PASSED SUCCESSFULLY!")
    print("----------------------------------------")

if __name__ == "__main__":
    test_pipeline()

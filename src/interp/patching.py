import torch

def patch_activations_and_generate(model, tokenizer, prompt, j_neurons, donor_activations, system_prompt=None, max_new_tokens=25):
    """
    Generates a response to a prompt while patching (overwriting) the activations
    of specified J-neurons with pre-saved donor activations.
    
    j_neurons: List of dicts with layer_idx, neuron_idx
    donor_activations: Dict mapping (layer_idx, neuron_idx) to a tensor or constant value
    """
    hooks = []
    
    # We define a custom pre-hook or hook on down_proj to override intermediate activations z
    def make_patch_hook(layer_idx):
        # Find which of our target J-neurons are in this layer
        target_neurons = [n for n in j_neurons if n["layer_idx"] == layer_idx]
        
        def hook_fn(module, inp, out):
            # inp[0] is the intermediate activation z of shape [batch, seq, d_ff]
            z = inp[0]
            for n in target_neurons:
                n_idx = n["neuron_idx"]
                key = (layer_idx, n_idx)
                if key in donor_activations:
                    # Patch the activation with the donor value
                    donor_val = donor_activations[key]
                    
                    # We can broadcast or match sequence length
                    # For simplicity during token-by-token generation, we patch the last token (the active generation step)
                    with torch.no_grad():
                        z[:, -1, n_idx] = donor_val
            # Return None or modified outputs. In PyTorch, modifying z in-place updates down_proj input!
            
        return hook_fn

    # Register hooks on layers containing J-neurons
    layers_to_hook = set([n["layer_idx"] for n in j_neurons])
    for layer_idx in layers_to_hook:
        layer = model.model.layers[layer_idx]
        hook = layer.mlp.down_proj.register_forward_hook(make_patch_hook(layer_idx))
        hooks.append(hook)

    # Tokenize and generate
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        prompt_len = input_ids.shape[1]
        completion = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
    return completion

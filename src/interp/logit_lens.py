import torch

def get_neuron_vocabulary_projection(model, tokenizer, layer_idx, neuron_idx, k=10):
    """
    Projects the output direction of a specific neuron (column of MLP down_proj)
    through the model's final layer norm and language modeling head (unembedding).
    
    Returns:
        List of tuples: (token_str, logit_value)
    """
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    
    # Retrieve the down_proj output weight vector for neuron_idx
    # Weight shape: [hidden_size, intermediate_size]
    with torch.no_grad():
        neuron_vector = down_proj.weight[:, neuron_idx].unsqueeze(0).to(model.device) # [1, hidden_size]
        
        # Apply model final normalization
        normalized_vector = model.model.norm(neuron_vector) # [1, hidden_size]
        
        # Project to vocabulary logits via lm_head
        logits = model.lm_head(normalized_vector)[0] # [vocab_size]
        
        # Retrieve top k tokens
        top_logits, top_indices = torch.topk(logits, k)
        
    results = []
    for logit, idx in zip(top_logits.cpu().numpy(), top_indices.cpu().numpy()):
        token_str = tokenizer.decode([int(idx)])
        results.append((token_str, float(logit)))
        
    return results

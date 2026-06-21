import torch
from src.hooks import CETTHookManager

class JNeuronCircuitBreaker:
    """
    Monitors model generation token-by-token. If the sparse J-Neuron activation
    level (compliance score) exceeds a threshold, it halts the generation.
    """
    def __init__(self, model, tokenizer, j_neurons, intercept_threshold=0.5, intercept_bias=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.j_neurons = j_neurons # List of dicts with: layer_idx, neuron_idx, weight
        self.threshold = intercept_threshold
        self.bias = intercept_bias
        self.hook_manager = CETTHookManager(model)

    def compute_j_score(self, current_token_pos):
        """
        Computes the compliance logit at the current token position using J-neuron CETT values.
        """
        score = self.bias
        for neuron in self.j_neurons:
            layer_idx = neuron["layer_idx"]
            neuron_idx = neuron["neuron_idx"]
            weight = neuron["weight"]
            
            # Get CETT for current layer
            cett_layer = self.hook_manager.get_layer_cett(layer_idx)
            if cett_layer is not None:
                # Shape: [batch, total_seq, d_ff]
                # Extract for current token pos (last generated token)
                cett_val = cett_layer[0, current_token_pos, neuron_idx].item()
                score += cett_val * weight
        return score

    def generate(self, prompt, system_prompt=None, max_new_tokens=40, return_scores=False):
        """
        Custom token-by-token generation loop that intercepts jailbreak compliance.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        prompt_len = input_ids.shape[1]
        current_ids = input_ids.clone()
        
        self.hook_manager.register()
        
        scores_history = []
        interrupted = False
        interruption_step = -1
        
        for step in range(max_new_tokens):
            # Clear previous activations cache before forward pass
            self.hook_manager.cett_data.clear()
            
            # Forward pass (one step)
            with torch.no_grad():
                outputs = self.model(current_ids)
                
            # Next token logits
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # Greedy search
            
            # Check J-score for the just-generated token (represented by the last position in current_ids)
            # The activations correspond to the current forward pass, where the last token was current_ids[0, -1]
            current_token_pos = current_ids.shape[1] - 1
            j_score = self.compute_j_score(current_token_pos)
            scores_history.append(j_score)
            
            # Check threshold (sigmoid of J-score logit is compliance probability)
            prob = 1.0 / (1.0 + torch.exp(torch.tensor(-j_score)).item())
            
            if prob > self.threshold:
                interrupted = True
                interruption_step = step
                break
                
            # Append next token and continue
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # If EOS token generated, stop naturally
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        self.hook_manager.clear()
        
        generated_tokens = current_ids[0, prompt_len:]
        completion_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if interrupted:
            # Append warning message indicating interruption
            completion_text = completion_text.strip() + "\n\n⚠️ [GENERATION HALTED BY J-NEURON CIRCUIT BREAKER: Potential safety policy violation detected]"
            
        result = {
            "completion": completion_text,
            "interrupted": interrupted,
            "interruption_step": interruption_step,
            "scores": scores_history
        }
        
        if return_scores:
            return result
        return completion_text

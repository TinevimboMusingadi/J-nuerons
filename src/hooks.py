import torch
import torch.nn as nn

class CETTHookManager:
    """
    Manages the registration and collection of CETT (Contribution-based Activation Tracking)
    activations from the MLP layers of a Transformer model (designed for Qwen architectures).
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.cett_data = {}  # Map from layer index to CETT activations [batch, seq, d_ff]
        self._precompute_norms()

    def _precompute_norms(self):
        """
        Precomputes the L2 norms of each column of the down_proj weight matrix for each layer.
        For nn.Linear(d_ff, d_model), weight is shape [d_model, d_ff], so column j is weight[:, j].
        """
        self.col_norms = {}
        for idx, layer in enumerate(self.model.model.layers):
            down_proj = layer.mlp.down_proj
            # down_proj.weight has shape [hidden_size, intermediate_size]
            # norm(dim=0) calculates the L2 norm of each column (representing one neuron's projection vector)
            with torch.no_grad():
                norms = down_proj.weight.norm(dim=0).detach().cpu()
            self.col_norms[idx] = norms

    def register(self):
        """
        Registers forward hooks on the mlp.down_proj modules of all decoder layers.
        """
        self.clear()
        
        def make_hook(layer_idx):
            col_norms = self.col_norms[layer_idx].to(self.model.device)
            
            def hook_fn(module, inp, out):
                # inp[0] is the input to down_proj, which has shape [batch, seq, d_ff] (i.e. the SwiGLU activation z)
                # out is the output of down_proj, which has shape [batch, seq, d_model]
                z = inp[0].detach() # [batch, seq, d_ff]
                h = out.detach()    # [batch, seq, d_model]
                
                # Compute contribution: |z_j| * ||W_down[:, j]||_2
                # z is [batch, seq, d_ff], col_norms is [d_ff]
                contrib = z.abs() * col_norms.view(1, 1, -1)
                
                # Compute L2 norm of the output h over the hidden_size dimension
                h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8) # [batch, seq, 1]
                
                # CETT = contribution / ||h||_2
                cett = contrib / h_norm # [batch, seq, d_ff]
                
                self.cett_data[layer_idx] = cett.cpu()
                
            return hook_fn

        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.mlp.down_proj.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)

    def clear(self):
        """
        Removes all registered hooks and clears captured data.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.cett_data = {}

    def get_layer_cett(self, layer_idx):
        """
        Retrieves the captured CETT activations for a specific layer.
        """
        return self.cett_data.get(layer_idx, None)

    def get_token_cett(self, layer_idx, token_pos):
        """
        Retrieves the captured CETT activations for a specific layer at a specific token position.
        """
        data = self.cett_data.get(layer_idx, None)
        if data is not None:
            return data[:, token_pos, :]
        return None

import numpy as np

def compute_jaccard_similarity(set_a, set_b):
    """
    Computes the Jaccard similarity between two sets of neurons.
    Each set should be a set of tuples (layer_idx, neuron_idx).
    """
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return intersection / union

def compute_cosine_similarity(coef_a, coef_b):
    """
    Computes the cosine similarity between two full-dimensional coefficient arrays.
    """
    norm_a = np.linalg.norm(coef_a)
    norm_b = np.linalg.norm(coef_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return np.dot(coef_a, coef_b) / (norm_a * norm_b)

def get_top_neuron_sets(j_neurons, top_k=20):
    """
    Converts a list of J-neuron dicts into a python set of (layer_idx, neuron_idx) tuples.
    """
    return set([(n["layer_idx"], n["neuron_idx"]) for n in j_neurons[:top_k]])

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_cross_lingual_roc(lang_labels, lang_probs, save_dir="results"):
    """
    Plots the ROC curves of the J-Neuron probe across different languages.
    
    lang_labels: dict mapping lang -> true labels list
    lang_probs: dict mapping lang -> predicted probability list
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    colors = {
        "en": "#1f77b4",  # Blue
        "af": "#ff7f0e",  # Orange
        "sw": "#2ca02c",  # Green
        "zu": "#d62728",  # Red
        "sn": "#9467bd"   # Purple
    }
    
    names = {
        "en": "English (Source)",
        "af": "Afrikaans",
        "sw": "Kiswahili",
        "zu": "isiZulu",
        "sn": "Shona (Novel)"
    }
    
    for lang in lang_labels.keys():
        labels = lang_labels[lang]
        probs = lang_probs[lang]
        
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        color = colors.get(lang, "#7f7f7f")
        name = names.get(lang, lang.upper())
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUROC = {roc_auc:.3f})")
        
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Cross-Lingual J-Neuron Detection Performance (ROCs)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Sleek aesthetics
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "cross_lingual_roc.png")
    plt.savefig(plot_path, bbox_inches="tight")
    return fig

def plot_asr_comparison(asr_before, asr_after, save_dir="results"):
    """
    Bar chart comparing Jailbreak Success Rate (ASR) before vs after early stopping interceptor.
    
    asr_before: dict mapping lang -> float
    asr_after: dict mapping lang -> float
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    langs = list(asr_before.keys())
    x = np.arange(len(langs))
    width = 0.35
    
    names = {
        "en": "English",
        "af": "Afrikaans",
        "sw": "Kiswahili",
        "zu": "isiZulu",
        "sn": "Shona"
    }
    labels = [names.get(l, l.upper()) for l in langs]
    
    # Convert back to percent representation for plotting
    before_vals = [asr_before[l] * 100 for l in langs]
    after_vals = [asr_after[l] * 100 for l in langs]
    
    rects1 = ax.bar(x - width/2, before_vals, width, label="Baseline (No Intercept)", color="#ef5350")
    rects2 = ax.bar(x + width/2, after_vals, width, label="With J-Neuron Circuit Breaker", color="#66bb6a")
    
    ax.set_ylabel("Jailbreak Success Rate (%)", fontsize=12)
    ax.set_title("Jailbreak Success Rate Before & After J-Neuron Circuit Breaker", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Label bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "asr_comparison.png")
    plt.savefig(plot_path, bbox_inches="tight")
    return fig

def plot_causal_scaling(alphas, compliance_rates, save_dir="results"):
    """
    Plots the causal scaling of J-Neuron activations vs. Jailbreak Success (Compliance) Rate.
    
    alphas: list of scaling factors (e.g. 0.0 to 3.0)
    compliance_rates: list of floats representing successful jailbreaks for each alpha
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    
    compliance_pct = [c * 100 for c in compliance_rates]
    
    ax.plot(alphas, compliance_pct, marker="o", color="#ab47bc", linewidth=2.5, markersize=8)
    
    # Highlight suppression vs amplification zones
    ax.axvspan(0.0, 1.0, alpha=0.15, color="red", label="Suppression Zone (α < 1)")
    ax.axvspan(1.0, 3.0, alpha=0.15, color="green", label="Amplification Zone (α > 1)")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Normal Operation (α = 1)")
    
    ax.set_xlabel("Activation Scaling Factor (α)", fontsize=12)
    ax.set_ylabel("Jailbreak Success Rate (%)", fontsize=12)
    ax.set_title("Causal Perturbation: Compliance vs. J-Neuron Scaling Factor (α)", fontsize=13, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.set_xlim(min(alphas) - 0.1, max(alphas) + 0.1)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "causal_scaling_sweep.png")
    plt.savefig(plot_path, bbox_inches="tight")
    return fig

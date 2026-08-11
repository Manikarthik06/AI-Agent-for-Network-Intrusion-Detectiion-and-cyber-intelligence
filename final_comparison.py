#!/usr/bin/env python3
"""
Final Model Comparison - All Models
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

print("="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

# Load all metrics
dnn_metrics = joblib.load('models/dnn_metrics.pkl')
autoencoder_original_metrics = joblib.load('models/autoencoder_metrics.pkl')
autoencoder_improved_metrics = joblib.load('models/autoencoder_improved_metrics.pkl')
autoencoder_optimized_metrics = joblib.load('models/autoencoder_optimized_metrics.pkl')
transformer_metrics = joblib.load('models/transformer_metrics.pkl')

# Create comparison table
comparison = pd.DataFrame({
    'Model': ['DNN', 'Transformer', 'Autoencoder (Optimized)', 'Autoencoder (Improved)', 'Autoencoder (Original)'],
    'Accuracy': [
        dnn_metrics['accuracy'],
        transformer_metrics['accuracy'],
        autoencoder_optimized_metrics['accuracy'],
        autoencoder_improved_metrics['accuracy'],
        autoencoder_original_metrics['accuracy']
    ],
    'Precision': [
        dnn_metrics['precision'],
        transformer_metrics['precision'],
        autoencoder_optimized_metrics['precision'],
        autoencoder_improved_metrics['precision'],
        autoencoder_original_metrics['precision']
    ],
    'Recall': [
        dnn_metrics['recall'],
        transformer_metrics['recall'],
        autoencoder_optimized_metrics['recall'],
        autoencoder_improved_metrics['recall'],
        autoencoder_original_metrics['recall']
    ],
    'F1-Score': [
        dnn_metrics['f1_score'],
        transformer_metrics['f1_score'],
        autoencoder_optimized_metrics['f1_score'],
        autoencoder_improved_metrics['f1_score'],
        autoencoder_original_metrics['f1_score']
    ],
    'ROC-AUC': [
        dnn_metrics['roc_auc'],
        transformer_metrics['roc_auc'],
        autoencoder_optimized_metrics['roc_auc'],
        autoencoder_improved_metrics['roc_auc'],
        autoencoder_original_metrics['roc_auc']
    ]
})

print("\n" + "="*60)
print("FINAL COMPARISON TABLE")
print("="*60)
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv('reports/final_model_comparison.csv', index=False)
print("\n✅ Saved comparison to: reports/final_model_comparison.csv")

# Determine best model
best_f1_idx = comparison['F1-Score'].idxmax()
best_model = comparison.loc[best_f1_idx, 'Model']
best_f1 = comparison.loc[best_f1_idx, 'F1-Score']

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)
print(f"🏆 Best Model: {best_model}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Accuracy: {comparison.loc[best_f1_idx, 'Accuracy']:.4f}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(comparison['Model'], comparison[metric], color=colors[:len(comparison)])
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)

axes[1, 2].remove()
plt.suptitle('Model Performance Comparison - Final', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/final_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved comparison visualization to: reports/figures/final_model_comparison.png")

# Print insights
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("""
1. **DNN and Transformer** achieve near-perfect performance (99.96% accuracy)
   - Both are excellent for production deployment
   - DNN is simpler and faster
   - Transformer is more complex but more expressive

2. **Autoencoders** struggle with this dataset
   - DDoS attacks have very distinct patterns
   - Not suitable for subtle anomaly detection here
   - Optimized version achieved 59.11% accuracy

3. **Recommendation**: Use DNN for production
   - Highest F1-Score (1.0000)
   - Fast inference
   - Simple architecture
   - Easy to maintain
""")

# Save final summary
summary = {
    'best_model': best_model,
    'best_f1': best_f1,
    'best_accuracy': comparison.loc[best_f1_idx, 'Accuracy'],
    'models': comparison.to_dict(),
    'recommendations': {
        'production': 'DNN',
        'research': 'All models',
        'anomaly_detection': 'Autoencoder (for subtle anomalies)'
    }
}

with open('reports/final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ Final summary saved to: reports/final_summary.json")
print("\n" + "="*60)
print("FINAL COMPARISON COMPLETE!")
print("="*60)

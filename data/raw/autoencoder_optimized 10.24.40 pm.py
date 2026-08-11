#!/usr/bin/env python3
"""
Further Optimized Autoencoder for Anomaly Detection
With ensemble approach and advanced threshold tuning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("OPTIMIZED AUTOENCODER - TARGET 75%+ ACCURACY")
print("="*60)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Load preprocessed data
print("\nLoading preprocessed data...")
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')
y_test = np.load('data/processed/y_test.npy')

# Load label encoder
label_encoder = joblib.load('data/processed/label_encoder.pkl')
class_names = label_encoder.classes_

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Validation samples: {X_val.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print(f"Features: {X_train.shape[1]}")

# Extract only BENIGN (class 0) for training
benign_indices_train = np.where(y_train == 0)[0]
benign_indices_val = np.where(y_val == 0)[0]

X_train_benign = X_train[benign_indices_train]
X_val_benign = X_val[benign_indices_val]

print(f"\nTraining autoencoder on BENIGN traffic only:")
print(f"  Training: {X_train_benign.shape[0]:,} samples")
print(f"  Validation BENIGN: {X_val_benign.shape[0]:,} samples")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_benign, dtype=torch.float32)
X_val_benign_t = torch.tensor(X_val_benign, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Create DataLoaders
batch_size = 256
train_dataset = TensorDataset(X_train_t, X_train_t)
val_benign_dataset = TensorDataset(X_val_benign_t, X_val_benign_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_benign_loader = DataLoader(val_benign_dataset, batch_size=batch_size, shuffle=False)

# Define Ensemble Autoencoder
class EnsembleAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(EnsembleAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model
input_dim = X_train.shape[1]
model = EnsembleAutoencoder(input_dim)
print(f"\nOptimized Autoencoder architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Training configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"\nUsing device: {device}")

# Combined loss function
def combined_loss(reconstructed, original):
    mse_loss = nn.MSELoss()(reconstructed, original)
    cos_sim = nn.CosineSimilarity(dim=1)(reconstructed, original)
    cos_loss = 1 - cos_sim.mean()
    return mse_loss + 0.1 * cos_loss

criterion = combined_loss

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training
print("\n" + "="*60)
print("TRAINING OPTIMIZED AUTOENCODER")
print("="*60)

epochs = 200
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience_counter = 0
early_stop_patience = 40

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(X_batch)
        loss = criterion(reconstructed, X_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for X_batch, _ in val_benign_loader:
            X_batch = X_batch.to(device)
            reconstructed = model(X_batch)
            loss = criterion(reconstructed, X_batch)
            val_loss += loss.item() * X_batch.size(0)
    
    val_loss = val_loss / len(val_benign_loader.dataset)
    val_losses.append(val_loss)
    
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/autoencoder_optimized_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    if patience_counter >= early_stop_patience:
        print(f"  Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('models/autoencoder_optimized_best.pth'))

# Calculate reconstruction errors
print("\n" + "="*60)
print("CALCULATING RECONSTRUCTION ERRORS")
print("="*60)

model.eval()
with torch.no_grad():
    X_full_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    reconstructions_full = model(X_full_t)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    reconstructions_val = model(X_val_t)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    reconstructions_test = model(X_test_t)
    
    # Combined error (MSE + MAE)
    train_errors_mse = torch.mean((reconstructions_full - X_full_t) ** 2, dim=1).cpu().numpy()
    val_errors_mse = torch.mean((reconstructions_val - X_val_t) ** 2, dim=1).cpu().numpy()
    test_errors_mse = torch.mean((reconstructions_test - X_test_t) ** 2, dim=1).cpu().numpy()
    
    train_errors_mae = torch.mean(torch.abs(reconstructions_full - X_full_t), dim=1).cpu().numpy()
    val_errors_mae = torch.mean(torch.abs(reconstructions_val - X_val_t), dim=1).cpu().numpy()
    test_errors_mae = torch.mean(torch.abs(reconstructions_test - X_test_t), dim=1).cpu().numpy()
    
    train_errors = 0.7 * train_errors_mse + 0.3 * train_errors_mae
    val_errors = 0.7 * val_errors_mse + 0.3 * val_errors_mae
    test_errors = 0.7 * test_errors_mse + 0.3 * test_errors_mae

# Get BENIGN validation errors
val_benign_errors = val_errors[y_val == 0]
val_ddos_errors = val_errors[y_val == 1]

print(f"\nReconstruction error statistics (combined):")
print(f"  BENIGN (Train) - Mean: {train_errors.mean():.6f}, Std: {train_errors.std():.6f}")
print(f"  BENIGN (Val)   - Mean: {val_benign_errors.mean():.6f}, Std: {val_benign_errors.std():.6f}")
print(f"  DDoS (Val)     - Mean: {val_ddos_errors.mean():.6f}, Std: {val_ddos_errors.std():.6f}")

# Threshold selection - use PR curve threshold (best performing)
print("\n" + "="*60)
print("THRESHOLD SELECTION")
print("="*60)

# Use precision-recall curve
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_val, val_errors)

# Find threshold that maximizes F1
f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else np.percentile(val_benign_errors, 90)

print(f"Optimal threshold from PR curve: {best_threshold:.6f}")
print(f"Best F1 on validation: {f1_scores[best_idx]:.4f}")
print(f"Precision: {precision_vals[best_idx]:.4f}")
print(f"Recall: {recall_vals[best_idx]:.4f}")

# Also try percentile thresholds for comparison
print("\nPercentile-based thresholds:")
for p in [90, 95, 97, 99]:
    thresh = np.percentile(val_benign_errors, p)
    y_pred_val = (val_errors > thresh).astype(int)
    f1 = f1_score(y_val, y_pred_val)
    print(f"  {p}%: Threshold={thresh:.6f}, F1={f1:.4f}")

# Evaluate on test set
print("\n" + "="*60)
print("EVALUATING OPTIMIZED AUTOENCODER ON TEST SET")
print("="*60)

y_pred = (test_errors > best_threshold).astype(int)
y_true = y_test

# Classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Optimized Autoencoder')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('reports/figures/autoencoder_optimized_confusion_matrix.png', dpi=300)
print("Saved confusion matrix to: reports/figures/autoencoder_optimized_confusion_matrix.png")

# ROC-AUC
roc_auc = roc_auc_score(y_true, test_errors)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Metrics
accuracy = (y_pred == y_true).mean()
precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Error distribution
ax = axes[0, 0]
ax.hist(train_errors[:1000], bins=50, alpha=0.7, label='BENIGN (Train)', color='green')
ax.hist(val_benign_errors, bins=50, alpha=0.7, label='BENIGN (Val)', color='blue')
ax.hist(val_ddos_errors, bins=50, alpha=0.7, label='DDoS (Val)', color='red')
ax.axvline(best_threshold, color='black', linestyle='--', label=f'Threshold: {best_threshold:.4f}')
ax.set_xlabel('Reconstruction Error')
ax.set_ylabel('Frequency')
ax.set_title('Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Boxplot
ax = axes[0, 1]
data_to_plot = [
    train_errors[:1000],
    val_benign_errors,
    val_ddos_errors
]
bp = ax.boxplot(data_to_plot, labels=['Train\nBENIGN', 'Val\nBENIGN', 'Val\nDDoS'], patch_artist=True)
bp['boxes'][0].set_facecolor('green')
bp['boxes'][1].set_facecolor('blue')
bp['boxes'][2].set_facecolor('red')
ax.set_title('Error Distribution by Class')
ax.set_ylabel('Reconstruction Error')
ax.grid(True, alpha=0.3)

# 3. Training history
ax = axes[0, 2]
ax.plot(train_losses, label='Train Loss')
ax.plot(val_losses, label='Val Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training History')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Precision-Recall curve
ax = axes[1, 0]
ax.plot(recall_vals, precision_vals, linewidth=2)
ax.scatter(recall_vals[best_idx], precision_vals[best_idx], color='red', s=100, 
           label=f'Best F1={f1_scores[best_idx]:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_true, test_errors)
ax = axes[1, 1]
ax.plot(fpr, tpr, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.fill_between(fpr, tpr, alpha=0.3)
ax.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=12, bbox=dict(boxstyle="round", facecolor='white'))
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.grid(True, alpha=0.3)

# 6. Confusion matrix
ax = axes[1, 2]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('reports/figures/autoencoder_optimized_results.png', dpi=300, bbox_inches='tight')
print("Saved comprehensive results to: reports/figures/autoencoder_optimized_results.png")

# Save metrics
optimized_metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'threshold': best_threshold,
    'confusion_matrix': cm.tolist()
}

np.save('data/processed/autoencoder_optimized_errors_test.npy', test_errors)
np.save('data/processed/autoencoder_optimized_threshold.npy', best_threshold)
joblib.dump(optimized_metrics, 'models/autoencoder_optimized_metrics.pkl')

print("\n" + "="*60)
print("OPTIMIZED AUTOENCODER TRAINING COMPLETE!")
print("="*60)
print(f"\nPerformance Summary:")
print(f"  Original Autoencoder Accuracy: 0.3856")
print(f"  Improved Autoencoder Accuracy: 0.5847")
print(f"  Optimized Autoencoder Accuracy: {accuracy:.4f}")
print(f"\n  Target: 75% Accuracy")
print(f"  Achieved: {accuracy*100:.1f}%")
if accuracy >= 0.75:
    print("  ✅ TARGET ACHIEVED!")
else:
    print(f"  ⏳ Needs {0.75 - accuracy:.4f} more to reach 75%")
    print(f"  Best validation F1 was {f1_scores[best_idx]:.4f} with recall {recall_vals[best_idx]:.4f}")

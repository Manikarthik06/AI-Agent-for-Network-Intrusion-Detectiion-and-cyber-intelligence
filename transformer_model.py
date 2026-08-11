#!/usr/bin/env python3
"""
Transformer Model for Network Intrusion Detection
Implements a transformer for tabular/network-flow data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRANSFORMER MODEL - NETWORK INTRUSION DETECTION")
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
print(f"Classes: {class_names}")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
batch_size = 256
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Transformer for Tabular Data
class TabularTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=128, dropout=0.1, max_features=68):
        super(TabularTransformer, self).__init__()
        
        # Feature projection to d_model dimensions
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # Learnable positional encoding (each feature gets a position)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_features, d_model) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global average pooling + classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # Project features to d_model dimensions
        x = self.feature_projection(x)  # (batch, d_model)
        
        # Add sequence dimension and positional encoding
        x = x.unsqueeze(1)  # (batch, seq_len=1, d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer expects (batch, seq_len, d_model)
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classify
        x = self.classifier(x)
        return x

# Initialize model
input_dim = X_train.shape[1]
model = TabularTransformer(
    input_dim=input_dim,
    num_classes=2,
    d_model=64,
    nhead=4,
    num_layers=3,
    dim_feedforward=128,
    dropout=0.1
)

print(f"\nTransformer Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Training configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"\nUsing device: {device}")

# Class weights for imbalance
class_weights = torch.tensor([1.173, 0.871], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

# Training
print("\n" + "="*60)
print("TRAINING TRANSFORMER")
print("="*60)

epochs = 50
best_val_loss = float('inf')
best_val_f1 = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []
val_f1s = []

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()
            
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(y_batch.cpu().numpy())
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total
    val_f1 = f1_score(val_true, val_preds)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'models/transformer_best.pth')
        print(f"  Epoch {epoch+1}: New best model! Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

# Load best model
model.load_state_dict(torch.load('models/transformer_best.pth'))
print(f"\n✅ Best model loaded (Val F1: {best_val_f1:.4f})")

# Evaluation
print("\n" + "="*60)
print("EVALUATING TRANSFORMER ON TEST SET")
print("="*60)

model.eval()
y_pred = []
y_true = []
y_prob = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.numpy())
        y_prob.extend(probs[:, 1].cpu().numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Metrics
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Transformer')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('reports/figures/transformer_confusion_matrix.png', dpi=300)
print("✅ Saved confusion matrix to: reports/figures/transformer_confusion_matrix.png")

# ROC-AUC
roc_auc = roc_auc_score(y_true, y_prob)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Training curves
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label='Train Accuracy')
ax2.plot(val_accs, label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(val_f1s, label='Val F1-Score', color='green')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('F1-Score')
ax3.set_title('Validation F1-Score')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/transformer_training_curves.png', dpi=300)
print("✅ Saved training curves to: reports/figures/transformer_training_curves.png")

# Metrics
accuracy = (y_pred == y_true).mean()
precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📊 Test Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")

# Save metrics
transformer_metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'confusion_matrix': cm.tolist(),
    'total_params': total_params,
    'trainable_params': trainable_params,
    'best_val_f1': best_val_f1
}

joblib.dump(transformer_metrics, 'models/transformer_metrics.pkl')

print("\n" + "="*60)
print("TRANSFORMER TRAINING COMPLETE!")
print("="*60)

# Compare with DNN
print("\n📊 Comparison with DNN:")
dnn_metrics = joblib.load('models/dnn_metrics.pkl')
print(f"  {'Metric':<15} {'DNN':<10} {'Transformer':<15}")
print(f"  {'-'*40}")
print(f"  {'Accuracy':<15} {dnn_metrics['accuracy']:.4f}     {accuracy:.4f}")
print(f"  {'F1-Score':<15} {dnn_metrics['f1_score']:.4f}     {f1:.4f}")
print(f"  {'ROC-AUC':<15} {dnn_metrics['roc_auc']:.4f}     {roc_auc:.4f}")

if accuracy > dnn_metrics['accuracy']:
    print("\n✅ Transformer outperforms DNN!")
else:
    print("\nℹ️  DNN still performs better on this dataset")

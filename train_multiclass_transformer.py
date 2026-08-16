#!/usr/bin/env python3
"""
Retrain Transformer for multi-class classification on combined dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("RETRAINING TRANSFORMER - MULTI-CLASS")
print("="*60)

# Load data
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')
y_test = np.load('data/processed/y_test.npy')
label_encoder = joblib.load('data/processed/label_encoder.pkl')

num_classes = len(label_encoder.classes_)
print(f"Num classes: {num_classes}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Features: {X_train.shape[1]}")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# DataLoader
batch_size = 512
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

# Model
class MultiClassTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_projection(x).unsqueeze(1)
        x = x + self.pos_encoder
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)

model = MultiClassTransformer(input_dim=X_train.shape[1], num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Class weights (inverse frequency)
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"Using device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
epochs = 50
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            _, pred = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (pred == y_batch).sum().item()
    
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/transformer_multiclass.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Evaluate on test set
model.load_state_dict(torch.load('models/transformer_multiclass.pth'))
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size):
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, pred = torch.max(outputs, 1)
        y_pred.extend(pred.cpu().numpy())
        y_true.extend(y_batch.numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Multi-Class Transformer')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('reports/figures/transformer_multiclass_confusion.png', dpi=300)
print("\n✅ Confusion matrix saved to reports/figures/transformer_multiclass_confusion.png")

print("\n✅ Transformer Multi-Class Training Complete!")
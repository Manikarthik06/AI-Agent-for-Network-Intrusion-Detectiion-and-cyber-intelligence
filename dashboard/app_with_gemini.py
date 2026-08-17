#!/usr/bin/env python3
"""
NIDS Dashboard – Detection + PDF with Metrics on Uploaded Dataset
All metric visualisations are computed on the dataset you upload.
Fixed shape mismatch errors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime
import sys
import os
import json
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cti.knowledge_base import ThreatIntelligence
from src.agent.gemini_agent import GeminiCyberAgent

st.set_page_config(page_title="🛡️ AI NIDS", page_icon="🛡️", layout="wide")

# Custom CSS (same as before)
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .gemini-badge {
        background: linear-gradient(135deg, #4285f4, #ea4335, #fbbc04, #34a853);
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .gemini-summary {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .gemini-impact {
        background: rgba(255, 170, 0, 0.1);
        border-left: 4px solid #ffaa00;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. Model Definition
# ============================================================
class MultiClassDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# ============================================================
# 2. Load Artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    input_dim = len(feature_names)
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()
    model = MultiClassDNN(input_dim, num_classes)
    model.load_state_dict(torch.load('models/dnn_multiclass.pth', map_location=device))
    model.to(device)
    model.eval()
    scaler = joblib.load('data/processed/scaler.pkl')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    return label_encoder, model, scaler, device, feature_names, class_names, X_test, y_test

label_encoder, model, scaler, device, feature_names, class_names, X_test, y_test = load_artifacts()
num_classes = len(class_names)

cti = ThreatIntelligence()
gemini_agent = GeminiCyberAgent()

def parse_gemini_response(llm_text):
    try:
        data = json.loads(llm_text)
        return {
            'summary': data.get('summary', 'No summary provided.'),
            'impact': data.get('impact', 'No impact assessment.'),
            'recommendations': data.get('recommendations', ['No recommendations provided.']),
            'investigation': data.get('investigation', ['No investigation steps provided.'])
        }
    except:
        return {
            'summary': llm_text[:200] + '...' if len(llm_text) > 200 else llm_text,
            'impact': 'See full analysis below.',
            'recommendations': ['Check Gemini response for details.'],
            'investigation': ['Check Gemini response for details.']
        }

# ============================================================
# 3. Helper: prepare uploaded features (clean + reindex)
# ============================================================
def prepare_features(features_df):
    X = features_df.reindex(feature_names, axis=1, fill_value=0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = np.clip(X, -1e10, 1e10)
    return X

# ============================================================
# 4. Helper: predict on any data
# ============================================================
def predict_on_data(X_data):
    model.eval()
    batch_size = 1024
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_data), batch_size):
            batch = torch.tensor(X_data[i:i+batch_size], dtype=torch.float32).to(device)
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

# ============================================================
# 5. Generate metric images – FIXED: filter to present classes
# ============================================================
def generate_metric_images(y_true, y_pred, all_class_names):
    # Determine the set of classes present in the data
    present_classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    present_class_names = [all_class_names[i] for i in present_classes]
    
    # Compute confusion matrix and metrics using only present classes
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    precision = precision_score(y_true, y_pred, labels=present_classes, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=present_classes, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=present_classes, average=None, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = np.mean(y_pred == y_true)

    images = {}

    # 1. Confusion Matrix (matplotlib)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_class_names, yticklabels=present_class_names,
                ax=ax, cbar=True)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title('Confusion Matrix - Uploaded Dataset', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    images['confusion_matrix'] = buf

    # 2. Per-Class F1 (sorted)
    sorted_idx = np.argsort(f1)
    sorted_classes = [present_class_names[i] for i in sorted_idx]
    sorted_f1 = f1[sorted_idx]
    colors = ['green' if s >= 0.8 else 'orange' if s >= 0.5 else 'red' for s in sorted_f1]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_classes, sorted_f1, color=colors)
    ax.axvline(x=0.8, color='green', linestyle='--', linewidth=1, label='Good (0.8)')
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1, label='Moderate (0.5)')
    ax.set_xlabel('F1-Score')
    ax.set_title('Per-Class F1-Score - Uploaded Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, sorted_f1):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    images['f1_chart'] = buf

    # 3. Precision vs Recall
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(present_class_names))
    width = 0.35
    ax.bar(x - width/2, precision, width, label='Precision', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Precision vs Recall - Uploaded Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(present_class_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    images['precision_recall'] = buf

    # 4. Weighted vs Macro F1
    fig, ax = plt.subplots(figsize=(5, 4))
    metric_names = ['Weighted F1', 'Macro F1']
    scores = [weighted_f1, macro_f1]
    bars = ax.bar(metric_names, scores, color=['#2ecc71', '#e74c3c'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1-Score')
    ax.set_title('Weighted vs Macro F1 - Uploaded Dataset')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    images['weighted_macro'] = buf

    images['accuracy'] = accuracy
    images['weighted_f1'] = weighted_f1
    images['macro_f1'] = macro_f1
    return images, present_class_names

# ============================================================
# 6. UI
# ============================================================
st.title("🛡️ AI Network Intrusion Detection System")
st.markdown("### Powered by <span class='gemini-badge'>🤖 Gemini AI Agent</span>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📤 Upload Dataset")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df):,} rows")
            if 'Label' in df.columns:
                counts = df['Label'].value_counts()
                st.write("**Class Distribution:**")
                for label, count in counts.items():
                    st.write(f"  • {label}: {count:,} ({count/len(df)*100:.1f}%)")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    st.markdown("---")
    st.markdown(f"**Model:** DNN Multi-Class")
    st.markdown(f"**Classes:** {num_classes}")

tab_main, tab_metrics = st.tabs(["🔍 Detection & Reports", "📊 Performance Metrics"])

# ============================================================
# TAB 1: Detection & Reports
# ============================================================
with tab_main:
    if 'df' in st.session_state:
        df = st.session_state.df
        if 'Label' not in df.columns:
            st.warning("Uploaded file does not contain a 'Label' column.")
            st.stop()
        
        label_counts = df['Label'].value_counts()
        total = len(df)
        attack_count = total - label_counts.get('BENIGN', 0)
        benign_count = label_counts.get('BENIGN', 0)
        
        st.markdown("### 📊 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: rgba(255,255,255,0.6);">Total Samples</div>
                <div style="font-size: 2rem; font-weight: 700; color: #00d4ff;">{total:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: rgba(255,255,255,0.6);">🚨 Attacks</div>
                <div style="font-size: 2rem; font-weight: 700; color: #ff0066;">{attack_count:,}</div>
                <div style="color: rgba(255,255,255,0.4);">{attack_count/total*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: rgba(255,255,255,0.6);">✅ BENIGN</div>
                <div style="font-size: 2rem; font-weight: 700; color: #00ff88;">{benign_count:,}</div>
                <div style="color: rgba(255,255,255,0.4);">{benign_count/total*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            ratio = attack_count/benign_count if benign_count>0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: rgba(255,255,255,0.6);">Attack Ratio</div>
                <div style="font-size: 2rem; font-weight: 700; color: #ffaa00;">{ratio:.2f}x</div>
                <div style="color: rgba(255,255,255,0.4);">Attack : BENIGN</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 🎯 Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(values=label_counts.values, names=label_counts.index,
                             title='Class Distribution (Pie)',
                             color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.bar(x=label_counts.index, y=label_counts.values,
                             title='Class Distribution (Bar)',
                             color=label_counts.index,
                             color_discrete_sequence=px.colors.qualitative.Set3)
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Incident Report")
        st.caption("Using Gemini 2.5 Flash for intelligent threat analysis")
        
        if attack_count > 0:
            sample = df[df['Label'] != 'BENIGN'].iloc[0]
        else:
            sample = df.iloc[0]
        
        sample_features = sample.drop('Label')
        X_input = prepare_features(pd.DataFrame([sample_features]))
        X_scaled = scaler.transform(X_input)
        
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)
            pred_idx = pred_idx.cpu().item()
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
        
        st.info(f"**Prediction:** {pred_label} (confidence: {confidence*100:.1f}%)")
        
        if st.button("📄 Generate Full Report with Metrics (PDF)", use_container_width=True, type="primary"):
            with st.spinner("Computing metrics on uploaded dataset..."):
                X_upload = df.drop('Label', axis=1)
                y_true_upload = df['Label']
                y_true_enc = label_encoder.transform(y_true_upload)
                X_upload_clean = prepare_features(X_upload)
                X_upload_scaled = scaler.transform(X_upload_clean)
                y_pred_upload = predict_on_data(X_upload_scaled)
                
                metric_images, present_class_names = generate_metric_images(y_true_enc, y_pred_upload, class_names)
                
                attack_type = pred_label
                is_attack = attack_type != 'BENIGN'
                threat_info = cti.get_threat_info(attack_type)
                analysis = gemini_agent.analyze_threat(
                    prediction=1 if is_attack else 0,
                    confidence=confidence,
                    attack_type=attack_type
                )
                report = gemini_agent.generate_incident_report(analysis)
                
                # Build PDF (same structure as before, but use present_class_names for clarity)
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                        rightMargin=72, leftMargin=72,
                                        topMargin=72, bottomMargin=72)
                styles = getSampleStyleSheet()
                
                title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                             fontSize=16, alignment=TA_CENTER,
                                             textColor=colors.darkblue, spaceAfter=12)
                heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'],
                                               fontSize=12, textColor=colors.darkblue,
                                               spaceAfter=6, spaceBefore=12)
                normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'],
                                              fontSize=9, alignment=TA_LEFT, leading=12)
                bullet_style = ParagraphStyle('BulletStyle', parent=normal_style,
                                              leftIndent=20, bulletIndent=10)
                
                story = []
                story.append(Paragraph("Network Intrusion Detection Incident Report", title_style))
                story.append(Spacer(1, 0.2*inch))
                
                # Incident Summary
                story.append(Paragraph("Incident Summary", heading_style))
                summary_data = [
                    ["Incident ID", report['incident_id']],
                    ["Timestamp", report['timestamp']],
                    ["Priority", report['priority']],
                    ["Status", report['status']],
                    ["Predicted Attack", attack_type],
                    ["Confidence", f"{confidence*100:.1f}%"],
                ]
                summary_table = Table(summary_data, colWidths=[1.5*inch, 3*inch])
                summary_table.setStyle(TableStyle([
                    ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                    ('FONTSIZE', (0,0), (-1,-1), 9),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ]))
                story.append(summary_table)
                story.append(Spacer(1, 0.2*inch))
                
                # CTI
                story.append(Paragraph("Cyber Threat Intelligence (CTI)", heading_style))
                if threat_info:
                    mitre = threat_info.get('mitre', {})
                    cti_data = [
                        ["MITRE ATT&CK ID", mitre.get('technique_id', 'N/A')],
                        ["Technique Name", mitre.get('technique_name', 'N/A')],
                        ["Tactic", mitre.get('tactic', 'N/A')],
                        ["Severity", threat_info.get('severity', 'Unknown')],
                        ["Description", mitre.get('description', 'No description')[:300]],
                    ]
                    cti_table = Table(cti_data, colWidths=[1.5*inch, 3.5*inch])
                    cti_table.setStyle(TableStyle([
                        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                        ('FONTSIZE', (0,0), (-1,-1), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ]))
                    story.append(cti_table)
                else:
                    story.append(Paragraph("No CTI information available.", normal_style))
                story.append(Spacer(1, 0.2*inch))
                
                # Gemini Analysis
                story.append(Paragraph("Gemini AI Analysis", heading_style))
                if 'llm_analysis' in report:
                    gemini_data = parse_gemini_response(report['llm_analysis'])
                    story.append(Paragraph(f"<b>Summary:</b> {gemini_data['summary']}", normal_style))
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph(f"<b>Impact:</b> {gemini_data['impact']}", normal_style))
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph("<b>Recommendations:</b>", normal_style))
                    for rec in gemini_data['recommendations']:
                        story.append(Paragraph(f"• {rec}", bullet_style))
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph("<b>Investigation Steps:</b>", normal_style))
                    for step in gemini_data['investigation']:
                        story.append(Paragraph(f"• {step}", bullet_style))
                else:
                    story.append(Paragraph("No Gemini analysis available.", normal_style))
                
                story.append(PageBreak())
                
                # Metric Diagrams
                story.append(Paragraph("Model Performance on Uploaded Dataset", heading_style))
                story.append(Paragraph(f"Total samples: {len(y_true_upload):,}", normal_style))
                story.append(Paragraph(f"Accuracy: {metric_images['accuracy']:.3f}", normal_style))
                story.append(Paragraph(f"Weighted F1: {metric_images['weighted_f1']:.3f}", normal_style))
                story.append(Paragraph(f"Macro F1: {metric_images['macro_f1']:.3f}", normal_style))
                story.append(Spacer(1, 0.2*inch))
                
                story.append(Paragraph("1. Confusion Matrix", heading_style))
                img_cm = Image(metric_images['confusion_matrix'], width=5*inch, height=4*inch)
                story.append(img_cm)
                story.append(Spacer(1, 0.2*inch))
                
                story.append(Paragraph("2. Per-Class F1-Score", heading_style))
                img_f1 = Image(metric_images['f1_chart'], width=5*inch, height=4*inch)
                story.append(img_f1)
                story.append(Spacer(1, 0.2*inch))
                
                story.append(Paragraph("3. Precision vs Recall & Weighted/Macro F1", heading_style))
                img_pr = Image(metric_images['precision_recall'], width=3.5*inch, height=2.8*inch)
                img_wm = Image(metric_images['weighted_macro'], width=2.5*inch, height=2.5*inch)
                side_by_side = Table([[img_pr, img_wm]], colWidths=[3.5*inch, 2.5*inch])
                side_by_side.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(side_by_side)
                story.append(Spacer(1, 0.2*inch))
                
                story.append(Paragraph("---", normal_style))
                story.append(Paragraph(f"Generated by AI NIDS System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
                story.append(Paragraph("This report is for informational purposes only. No automated actions have been taken.", normal_style))
                
                doc.build(story)
                pdf_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Full Report with Metrics (PDF)",
                    data=pdf_buffer,
                    file_name=f"incident_{report['incident_id']}_full.pdf",
                    mime="application/pdf"
                )
        
        st.markdown("---")
        st.markdown("### 📄 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.info("📤 Please upload a dataset in the sidebar")

# ============================================================
# TAB 2: Performance Metrics (Interactive) – Also fixed
# ============================================================
with tab_metrics:
    if 'df' in st.session_state and 'Label' in st.session_state.df.columns:
        df = st.session_state.df
        X_upload = df.drop('Label', axis=1)
        y_true_upload = df['Label']
        y_true_enc = label_encoder.transform(y_true_upload)
        X_upload_clean = prepare_features(X_upload)
        X_upload_scaled = scaler.transform(X_upload_clean)
        y_pred_upload = predict_on_data(X_upload_scaled)
        
        # Compute metrics using only present classes
        present_classes = np.union1d(np.unique(y_true_enc), np.unique(y_pred_upload))
        present_class_names = [class_names[i] for i in present_classes]
        
        acc = np.mean(y_pred_upload == y_true_enc)
        wf1 = f1_score(y_true_enc, y_pred_upload, average='weighted')
        mf1 = f1_score(y_true_enc, y_pred_upload, average='macro')
        cm = confusion_matrix(y_true_enc, y_pred_upload, labels=present_classes)
        precision = precision_score(y_true_enc, y_pred_upload, labels=present_classes, average=None, zero_division=0)
        recall = recall_score(y_true_enc, y_pred_upload, labels=present_classes, average=None, zero_division=0)
        f1_per = f1_score(y_true_enc, y_pred_upload, labels=present_classes, average=None, zero_division=0)
        
        st.markdown("## 📊 Performance on Uploaded Dataset")
        st.markdown(f"*Samples: {len(y_true_upload):,}*")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Weighted F1", f"{wf1:.3f}")
        col3.metric("Macro F1", f"{mf1:.3f}")
        st.markdown("---")
        
        # Confusion Matrix using go.Heatmap
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=present_class_names,
            y=present_class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix - Uploaded Dataset",
            xaxis_title="Predicted",
            yaxis_title="True",
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Per-class F1
        sorted_idx = np.argsort(f1_per)
        sorted_classes = [present_class_names[i] for i in sorted_idx]
        sorted_f1 = f1_per[sorted_idx]
        fig_f1 = go.Figure(data=[go.Bar(
            x=sorted_f1, y=sorted_classes, orientation='h',
            marker_color=['green' if s>=0.8 else 'orange' if s>=0.5 else 'red' for s in sorted_f1],
            text=[f'{s:.3f}' for s in sorted_f1],
            textposition='outside'
        )])
        fig_f1.update_layout(
            title="Per-Class F1-Score - Uploaded Dataset",
            xaxis_title="F1-Score",
            yaxis_title="Class",
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_range=[0, 1]
        )
        st.plotly_chart(fig_f1, use_container_width=True)
        
        # Precision vs Recall
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(x=present_class_names, y=precision, name='Precision', marker_color='#3498db'))
        fig_pr.add_trace(go.Bar(x=present_class_names, y=recall, name='Recall', marker_color='#2ecc71'))
        fig_pr.update_layout(
            title="Precision vs Recall by Class - Uploaded Dataset",
            xaxis_title="Class",
            yaxis_title="Score",
            barmode='group',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        
        # Weighted vs Macro
        fig_wm = go.Figure(data=[go.Bar(
            x=['Weighted F1', 'Macro F1'],
            y=[wf1, mf1],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f'{wf1:.3f}', f'{mf1:.3f}'],
            textposition='outside'
        )])
        fig_wm.update_layout(
            title="Weighted vs Macro F1 - Uploaded Dataset",
            yaxis_title="F1-Score",
            yaxis_range=[0, 1],
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_wm, use_container_width=True)
    else:
        st.info("📤 Upload a dataset with labels to see performance metrics.")

st.markdown("---")
st.caption("🛡️ AI NIDS | Full PDF with Metrics on Uploaded Data | v15.0")
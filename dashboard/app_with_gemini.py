#!/usr/bin/env python3
"""
NIDS Dashboard with Gemini + Full PDF Report
Includes:
- Clean Gemini display (Summary, Impact, Recs, Investigation)
- PDF with pie chart image + formatted Gemini analysis
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
import tempfile
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cti.knowledge_base import ThreatIntelligence
from src.agent.gemini_agent import GeminiCyberAgent

st.set_page_config(page_title="🛡️ NIDS - Multi-Class", page_icon="🛡️", layout="wide")

# Custom CSS
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

# ==================================================
# Model Definition (must match training)
# ==================================================
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

# ==================================================
# Load all artifacts (cached)
# ==================================================
@st.cache_resource
def load_artifacts():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    input_dim = len(feature_names)
    num_classes = len(label_encoder.classes_)
    model = MultiClassDNN(input_dim, num_classes)
    model.load_state_dict(torch.load('models/dnn_multiclass.pth', map_location=device))
    model.to(device)
    model.eval()
    scaler = joblib.load('data/processed/scaler.pkl')
    return label_encoder, model, scaler, device, feature_names

label_encoder, model, scaler, device, feature_names = load_artifacts()
num_classes = len(label_encoder.classes_)
st.sidebar.write(f"✅ Model ready: {num_classes} classes, {len(feature_names)} features")

# Initialize CTI and Gemini
cti = ThreatIntelligence()
gemini_agent = GeminiCyberAgent()

# ==================================================
# Helper to format Gemini data
# ==================================================
def parse_gemini_response(llm_text):
    """Try to parse LLM response as JSON, return dict with keys."""
    try:
        data = json.loads(llm_text)
        # Ensure required keys exist
        return {
            'summary': data.get('summary', 'No summary provided.'),
            'impact': data.get('impact', 'No impact assessment.'),
            'recommendations': data.get('recommendations', ['No recommendations provided.']),
            'investigation': data.get('investigation', ['No investigation steps provided.'])
        }
    except:
        # Fallback: treat as plain text
        return {
            'summary': llm_text[:200] + '...' if len(llm_text) > 200 else llm_text,
            'impact': 'See full analysis below.',
            'recommendations': ['Check Gemini response for details.'],
            'investigation': ['Check Gemini response for details.']
        }

# ==================================================
# UI
# ==================================================
st.title("🛡️ Network Intrusion Detection System")
st.markdown("### Powered by <span class='gemini-badge'>🤖 Gemini AI Agent</span>", unsafe_allow_html=True)

# Sidebar: File upload
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

# Main content
if 'df' in st.session_state:
    df = st.session_state.df
    
    if 'Label' not in df.columns:
        st.warning("Uploaded file does not contain a 'Label' column.")
        st.stop()
    
    label_counts = df['Label'].value_counts()
    total = len(df)
    attack_count = total - label_counts.get('BENIGN', 0)
    benign_count = label_counts.get('BENIGN', 0)
    
    # Statistics
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
    
    # Charts
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
    
    # AI Report
    st.markdown("### 🤖 AI-Powered Incident Report")
    st.caption("Using Gemini 2.5 Flash for intelligent threat analysis")
    
    # Sample
    if attack_count > 0:
        sample = df[df['Label'] != 'BENIGN'].iloc[0]
    else:
        sample = df.iloc[0]
    
    sample_features = sample.drop('Label')
    X_input = sample_features.reindex(feature_names, axis=0).values.astype(np.float32).reshape(1, -1)
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
    
    # Generate Report Button
    if st.button("📄 Generate Gemini Report", use_container_width=True, type="primary"):
        with st.spinner("🧠 Analyzing with Gemini..."):
            attack_type = pred_label
            is_attack = attack_type != 'BENIGN'
            
            threat_info = cti.get_threat_info(attack_type)
            analysis = gemini_agent.analyze_threat(
                prediction=1 if is_attack else 0,
                confidence=confidence,
                attack_type=attack_type
            )
            report = gemini_agent.generate_incident_report(analysis)
            
            st.success("✅ Gemini Report Generated!")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Incident ID", report['incident_id'])
            with col2: st.metric("Priority", report['priority'])
            with col3: st.metric("Status", report['status'])
            
            # Display Gemini analysis nicely
            st.markdown("#### 🧠 Gemini Analysis")
            if 'llm_analysis' in report:
                gemini_data = parse_gemini_response(report['llm_analysis'])
                
                st.markdown(f"""
                <div class="gemini-summary">
                    <strong>📌 Summary</strong><br>
                    {gemini_data['summary']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="gemini-impact">
                    <strong>⚠️ Impact</strong><br>
                    {gemini_data['impact']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**📋 Recommendations**")
                for rec in gemini_data['recommendations']:
                    st.markdown(f"- {rec}")
                
                st.markdown("**🔍 Investigation Steps**")
                for step in gemini_data['investigation']:
                    st.markdown(f"- {step}")
            else:
                st.info("No Gemini analysis available.")
            
            # --- PDF Generation with Image ---
            st.markdown("---")
            st.markdown("#### 📄 Download Full Incident Report (PDF)")
            
            # Generate pie chart image using matplotlib
            plt.figure(figsize=(5, 4))
            plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Class Distribution')
            plt.tight_layout()
            
            # Save to BytesIO
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png', dpi=100)
            plt.close()
            img_bytes.seek(0)
            # reportlab needs a file-like object, so we'll save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                tmpfile.write(img_bytes.getvalue())
                tmpfile_path = tmpfile.name
            
            # Now build PDF
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=72)
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                         fontSize=18, alignment=TA_CENTER,
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
            
            # Class Distribution Table + Pie Chart Image
            story.append(Paragraph("Dataset Class Distribution", heading_style))
            # Table
            class_dist_data = [["Class", "Count", "Percentage"]]
            for label, count in label_counts.items():
                pct = count/total*100
                class_dist_data.append([label, f"{count:,}", f"{pct:.2f}%"])
            class_table = Table(class_dist_data, colWidths=[2.0*inch, 1.5*inch, 1.5*inch])
            class_table.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ]))
            story.append(class_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Pie chart image
            story.append(Paragraph("Class Distribution Pie Chart", heading_style))
            img = Image(tmpfile_path, width=3*inch, height=2.4*inch)
            story.append(img)
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
                    ["Description", mitre.get('description', 'No description')[:500]],
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
            
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("---", normal_style))
            story.append(Paragraph(f"Generated by AI NIDS System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            story.append(Paragraph("This report is for informational purposes only. No automated actions have been taken.", normal_style))
            
            doc.build(story)
            pdf_buffer.seek(0)
            
            # Clean up temp file
            try:
                os.remove(tmpfile_path)
            except:
                pass
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"incident_{report['incident_id']}.pdf",
                mime="application/pdf"
            )
    
    # Data preview
    st.markdown("---")
    st.markdown("### 📄 Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

else:
    st.info("📤 Please upload a dataset in the sidebar")

st.markdown("---")
st.caption("🛡️ NIDS Dashboard | Multi-Class | Full PDF Reports | v9.0")
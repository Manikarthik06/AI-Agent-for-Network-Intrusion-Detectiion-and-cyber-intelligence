#!/usr/bin/env python3
"""
NIDS Dashboard with Gemini AI Agent
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================================================
# IMPORTANT: Using GEMINI Agent instead of rule-based
# ==================================================
from src.agent.gemini_agent import GeminiCyberAgent as CyberAgent
# from src.agent.cyber_agent import CyberAgent  # ← OLD (commented out)

from src.cti.knowledge_base import ThreatIntelligence

st.set_page_config(
    page_title="🛡️ NIDS - Gemini AI Agent",
    page_icon="🛡️",
    layout="wide"
)

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
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.02);
        border-color: #00d4ff;
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
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Network Intrusion Detection System")
st.markdown("### Powered by <span class='gemini-badge'>🤖 Gemini AI Agent</span>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📤 Upload Dataset")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df):,} rows")
            
            st.markdown("---")
            st.markdown("### 🤖 AI Agent Status")
            st.markdown("✅ **Gemini Active**")
            st.caption("Model: gemini-2.5-flash")
            
            st.markdown("---")
            st.markdown("### 📊 Dataset Info")
            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {len(df.columns)}")
            
            if 'Label' in df.columns:
                counts = df['Label'].value_counts()
                st.write("**Class Distribution:**")
                for label, count in counts.items():
                    st.write(f"  • {label}: {count:,} ({count/len(df)*100:.1f}%)")
            
        except Exception as e:
            st.error(f"Error: {e}")

# Main content
if 'df' in st.session_state:
    df = st.session_state.df
    
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        total = len(df)
        
        ddos_count = label_counts.get('DDoS', 0)
        benign_count = label_counts.get('BENIGN', 0)
        
        # Display metrics
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
                <div style="color: rgba(255,255,255,0.6);">🚨 DDoS Attacks</div>
                <div style="font-size: 2rem; font-weight: 700; color: #ff0066;">{ddos_count:,}</div>
                <div style="color: rgba(255,255,255,0.4);">{ddos_count/total*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: rgba(255,255,255,0.6);">✅ BENIGN Traffic</div>
                <div style="font-size: 2rem; font-weight: 700; color: #00ff88;">{benign_count:,}</div>
                <div style="color: rgba(255,255,255,0.4);">{benign_count/total*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: rgba(255,255,255,0.6);">Attack Ratio</div>
                <div style="font-size: 2rem; font-weight: 700; color: #ffaa00;">{ddos_count/benign_count:.2f}x</div>
                <div style="color: rgba(255,255,255,0.4);">DDoS : BENIGN</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Class Distribution")
            
            fig = go.Figure(data=[go.Pie(
                labels=label_counts.index,
                values=label_counts.values,
                marker=dict(colors=['#00ff88' if l == 'BENIGN' else '#ff0066' for l in label_counts.index]),
                textinfo='label+percent+value'
            )])
            
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Distribution Bar Chart")
            
            fig = go.Figure(data=[go.Bar(
                x=label_counts.index,
                y=label_counts.values,
                marker_color=['#00ff88' if l == 'BENIGN' else '#ff0066' for l in label_counts.index],
                text=label_counts.values,
                textposition='auto'
            )])
            
            fig.update_layout(
                height=400,
                xaxis_title="Class",
                yaxis_title="Count",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Report Generation with Gemini
        st.markdown("### 🤖 AI-Powered Incident Report")
        st.caption("Using Gemini 2.5 Flash for intelligent threat analysis")
        
        if st.button("📄 Generate Gemini Report", use_container_width=True, type="primary"):
            with st.spinner("🧠 Analyzing with Gemini AI..."):
                try:
                    # Initialize Gemini Agent
                    agent = CyberAgent()
                    
                    # Simulate detection (or use actual detection results)
                    # For demo, use sample detection
                    analysis = agent.analyze_threat(
                        prediction=1,
                        confidence=0.9997,
                        attack_type='DDoS'
                    )
                    
                    # Generate report
                    report = agent.generate_incident_report(analysis)
                    
                    st.success("✅ Gemini Report Generated!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Incident ID", report['incident_id'])
                    with col2:
                        st.metric("Priority", report['priority'])
                    with col3:
                        st.metric("Status", report['status'])
                    
                    # Show Gemini Analysis
                    st.markdown("#### 🧠 Gemini Analysis")
                    st.json(report['analysis'])
                    
                    # Show Recommendations
                    st.markdown("#### 📋 Recommendations")
                    for rec in report['analysis'].get('recommendations', []):
                        st.markdown(f"- {rec}")
                    
                    # Show Investigation Steps
                    st.markdown("#### 🔍 Investigation Steps")
                    for step in report['analysis'].get('investigation', []):
                        st.markdown(f"- {step}")
                    
                    # Download report
                    report_json = json.dumps(report, indent=2)
                    st.download_button(
                        label="📥 Download Gemini Report (JSON)",
                        data=report_json,
                        file_name=f"gemini_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating Gemini report: {e}")
        
        st.markdown("---")
        
        # Data preview
        st.markdown("### 📄 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

else:
    st.info("📤 Please upload a dataset in the sidebar")

# Footer
st.markdown("---")
st.caption("🛡️ NIDS Dashboard | 🤖 Powered by Google Gemini AI | v3.0")

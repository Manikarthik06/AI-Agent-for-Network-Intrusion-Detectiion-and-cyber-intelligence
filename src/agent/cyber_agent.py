"""
AI Agent for Threat Analysis and Incident Response
"""

import json
from datetime import datetime
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import from src
from src.cti.knowledge_base import ThreatIntelligence

class CyberAgent:
    """AI Agent for analyzing network threats and generating recommendations"""
    
    def __init__(self):
        self.cti = ThreatIntelligence()
        self.incident_count = 0
    
    def analyze_threat(self, prediction, confidence, attack_type, features=None):
        """
        Analyze a detected threat and provide comprehensive analysis
        
        Args:
            prediction: 0 for BENIGN, 1 for ATTACK
            confidence: Confidence score (0-1)
            attack_type: Type of attack detected
            features: Optional feature values for detailed analysis
        
        Returns:
            Dictionary with threat analysis
        """
        
        # Get threat intelligence
        threat_info = self.cti.get_threat_info(attack_type)
        
        # Generate analysis
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'detection': {
                'attack_type': attack_type,
                'confidence': confidence,
                'prediction': prediction,
                'is_attack': prediction == 1
            },
            'threat_intelligence': threat_info,
            'analysis': self._generate_analysis(attack_type, confidence, threat_info),
            'recommendations': self._generate_recommendations(attack_type, threat_info)
        }
        
        return analysis
    
    def _generate_analysis(self, attack_type, confidence, threat_info):
        """Generate detailed threat analysis"""
        
        if attack_type == 'BENIGN':
            return {
                'summary': '✅ Normal network traffic detected',
                'risk_level': 'None',
                'action_required': False,
                'details': 'No malicious activity identified in this traffic flow.'
            }
        
        if attack_type == 'DDoS':
            severity = threat_info.get('severity', 'Critical') if threat_info else 'Critical'
            mitre_info = threat_info.get('mitre', {}) if threat_info else {}
            return {
                'summary': f'🚨 DDoS Attack Detected - Severity: {severity}',
                'risk_level': severity,
                'action_required': True,
                'confidence': confidence,
                'details': f"""
Detected a {attack_type} attack with {confidence*100:.1f}% confidence.
This is a {severity} severity threat that requires immediate attention.

MITRE ATT&CK Information:
- Technique ID: {mitre_info.get('technique_id', 'Unknown')}
- Technique Name: {mitre_info.get('technique_name', 'Unknown')}
- Tactic: {mitre_info.get('tactic', 'Unknown')}
                """,
                'mitre_mapping': mitre_info
            }
        
        # Generic attack analysis
        mitre_info = threat_info.get('mitre', {}) if threat_info else {}
        return {
            'summary': f'⚠️ {attack_type} Attack Detected',
            'risk_level': threat_info.get('severity', 'Medium') if threat_info else 'Medium',
            'action_required': True,
            'confidence': confidence,
            'details': f'Detected a {attack_type} attack with {confidence*100:.1f}% confidence.',
            'mitre_mapping': mitre_info
        }
    
    def _generate_recommendations(self, attack_type, threat_info):
        """Generate actionable recommendations"""
        
        if attack_type == 'BENIGN':
            return {
                'immediate': ['✅ No action required - Normal traffic'],
                'investigation': ['Monitor for normal patterns'],
                'long_term': ['Maintain current security posture']
            }
        
        if threat_info and 'recommendations' in threat_info:
            recs = threat_info['recommendations']
            return {
                'immediate': recs.get('immediate', ['Investigate the threat']),
                'investigation': recs.get('investigation', ['Analyze attack patterns']),
                'long_term': recs.get('long_term', ['Review security measures'])
            }
        
        # Fallback recommendations
        return {
            'immediate': ['Isolate affected systems', 'Block suspicious IPs'],
            'investigation': ['Analyze attack vectors', 'Review logs'],
            'long_term': ['Implement security improvements']
        }
    
    def generate_incident_report(self, threat_analysis):
        """
        Generate a structured incident report
        
        Args:
            threat_analysis: Output from analyze_threat()
        
        Returns:
            Dictionary with incident report
        """
        
        self.incident_count += 1
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{self.incident_count:04d}"
        
        is_attack = threat_analysis['detection']['is_attack']
        
        report = {
            'incident_id': incident_id,
            'timestamp': threat_analysis['timestamp'],
            'status': 'Open' if is_attack else 'Closed',
            'priority': 'High' if is_attack else 'Low',
            'detection': threat_analysis['detection'],
            'analysis': threat_analysis['analysis'],
            'recommendations': threat_analysis['recommendations'],
            'summary': {
                'attack_type': threat_analysis['detection']['attack_type'],
                'confidence': threat_analysis['detection']['confidence'],
                'requires_action': is_attack,
                'risk_level': threat_analysis['analysis'].get('risk_level', 'Low')
            }
        }
        
        return report

# Test the AI Agent
if __name__ == "__main__":
    print("="*60)
    print("AI AGENT - THREAT ANALYSIS TEST")
    print("="*60)
    
    # Initialize agent
    agent = CyberAgent()
    
    # Test Case 1: DDoS Attack
    print("\n📊 Test Case 1: DDoS Attack Detection")
    print("-" * 40)
    
    ddos_analysis = agent.analyze_threat(
        prediction=1,
        confidence=0.9997,
        attack_type='DDoS'
    )
    
    print("\nThreat Analysis:")
    print(f"  Attack Type: {ddos_analysis['detection']['attack_type']}")
    print(f"  Confidence: {ddos_analysis['detection']['confidence']*100:.1f}%")
    print(f"  Risk Level: {ddos_analysis['analysis']['risk_level']}")
    print(f"  Summary: {ddos_analysis['analysis']['summary'][:60]}...")
    
    # Generate incident report
    ddos_report = agent.generate_incident_report(ddos_analysis)
    print(f"\nIncident Report:")
    print(f"  Incident ID: {ddos_report['incident_id']}")
    print(f"  Priority: {ddos_report['priority']}")
    print(f"  Status: {ddos_report['status']}")
    print(f"  Immediate Actions: {ddos_report['recommendations']['immediate'][:2]}")
    
    # Test Case 2: BENIGN Traffic
    print("\n📊 Test Case 2: BENIGN Traffic")
    print("-" * 40)
    
    benign_analysis = agent.analyze_threat(
        prediction=0,
        confidence=0.9998,
        attack_type='BENIGN'
    )
    
    print(f"\nThreat Analysis:")
    print(f"  Summary: {benign_analysis['analysis']['summary']}")
    print(f"  Action Required: {benign_analysis['analysis']['action_required']}")
    
    # Generate incident report
    benign_report = agent.generate_incident_report(benign_analysis)
    print(f"\nIncident Report:")
    print(f"  Incident ID: {benign_report['incident_id']}")
    print(f"  Status: {benign_report['status']}")
    print(f"  Priority: {benign_report['priority']}")
    
    # Test Case 3: Full JSON Output
    print("\n📊 Test Case 3: Complete JSON Output")
    print("-" * 40)
    print("\nDDoS Analysis (JSON):")
    print(json.dumps(ddos_analysis, indent=2)[:500] + "...")
    
    print("\n✅ AI Agent Test Complete!")

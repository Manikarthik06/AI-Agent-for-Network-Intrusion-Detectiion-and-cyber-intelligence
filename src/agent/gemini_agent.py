"""
AI Agent with Google Gemini Integration
Using the NEW google.genai package (Python 3.10+ compatible)
"""

import os
import sys
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
load_dotenv()

# ==================================================
# IMPORTANT: Using the NEW package (NOT google.generativeai)
# ==================================================
try:
    from google import genai
    print("✅ Using google.genai package (recommended)")
except ImportError as e:
    print(f"❌ Please install: pip install google-genai")
    print(f"   Error: {e}")
    sys.exit(1)


class GeminiCyberAgent:
    def __init__(self, api_key=None, model_name=None):
        """Initialize Gemini AI Agent"""
        
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file")
        
        # ==================================================
        # NEW: Initialize client (no .configure() needed)
        # ==================================================
        self.client = genai.Client(api_key=self.api_key)
        
        # Use a model that exists in your API
        # From your earlier list: gemini-2.5-flash, gemini-2.5-pro, gemini-3.5-flash
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = "gemini-3.5-flash"
        
        print(f"\n✅ Model set: {self.model_name}")
        
        # Import CTI
        from src.cti.knowledge_base import ThreatIntelligence
        self.cti = ThreatIntelligence()
        self.incident_count = 0
        
        print("\n🤖 Gemini Cyber Agent Initialized!")
        print(f"   Model: {self.model_name}")
        print("   Status: Ready for threat analysis")
    
    def analyze_threat(self, prediction, confidence, attack_type, features=None):
        """Analyze threat using Gemini AI"""
        
        # Get CTI information
        threat_info = self.cti.get_threat_info(attack_type)
        
        # Prepare context
        context = self._prepare_context(attack_type, confidence, threat_info, features)
        
        # Get Gemini analysis
        llm_response = self._call_gemini(context)
        
        # Parse response
        parsed_analysis = self._parse_gemini_response(llm_response)
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'detection': {
                'attack_type': attack_type,
                'confidence': confidence,
                'prediction': prediction,
                'is_attack': prediction == 1
            },
            'threat_intelligence': threat_info,
            'llm_analysis': llm_response,
            'analysis': parsed_analysis,
            'recommendations': self._generate_recommendations(attack_type, threat_info)
        }
        
        return analysis
    
    def _prepare_context(self, attack_type, confidence, threat_info, features):
        """Prepare context for Gemini"""
        
        mitre_info = threat_info.get('mitre', {}) if threat_info else {}
        
        context = f"""
You are a cybersecurity expert analyzing a network threat.

DETECTION DETAILS:
- Attack Type: {attack_type}
- Confidence: {confidence*100:.1f}%
- Status: {'⚠️ ATTACK DETECTED' if confidence > 0.5 else '✅ Normal Traffic'}

MITRE ATT&CK INFORMATION:
- Technique ID: {mitre_info.get('technique_id', 'Unknown')}
- Technique Name: {mitre_info.get('technique_name', 'Unknown')}
- Tactic: {mitre_info.get('tactic', 'Unknown')}
- Description: {mitre_info.get('description', 'No description available')}

Please provide a concise analysis with:
1. Threat severity (Critical/High/Medium/Low/None)
2. Potential impact on the network
3. 3 actionable recommendations
4. Investigation steps

Format your response as JSON:
{{
    "severity": "Critical/High/Medium/Low/None",
    "impact": "Brief impact description",
    "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"],
    "investigation": ["Step 1", "Step 2", "Step 3"],
    "summary": "Brief summary of the threat"
}}
"""
        return context
    
    def _call_gemini(self, context):
        """Call Gemini API using new google.genai package"""
        try:
            # ==================================================
            # NEW: Correct API call format
            # ==================================================
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=context
            )
            return response.text
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️ API Error: {error_msg}")
            
            # Try with "models/" prefix as fallback
            try:
                response = self.client.models.generate_content(
                    model=f"models/{self.model_name}",
                    contents=context
                )
                return response.text
            except Exception as e2:
                return f"⚠️ Gemini API Error: {e2}"
    
    def _parse_gemini_response(self, response):
        """Parse Gemini's JSON response"""
        try:
            # Try to parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'summary': data.get('summary', 'Threat detected'),
                    'risk_level': data.get('severity', 'Unknown'),
                    'impact': data.get('impact', 'Unknown impact'),
                    'recommendations': data.get('recommendations', []),
                    'investigation': data.get('investigation', [])
                }
        except:
            pass
        
        # Fallback if parsing fails
        return {
            'summary': f'{response[:200]}...' if response else 'No analysis available',
            'risk_level': 'Unknown',
            'impact': 'Analysis in progress',
            'recommendations': ['Review Gemini analysis for details'],
            'investigation': ['Review Gemini analysis']
        }
    
    def _generate_recommendations(self, attack_type, threat_info):
        """Generate structured recommendations"""
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
        
        return {
            'immediate': ['Isolate affected systems', 'Block suspicious IPs'],
            'investigation': ['Analyze attack vectors', 'Review logs'],
            'long_term': ['Implement security improvements']
        }
    
    def generate_incident_report(self, threat_analysis):
        """Generate incident report"""
        
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
            'llm_analysis': threat_analysis.get('llm_analysis', 'No LLM analysis'),
            'recommendations': threat_analysis['recommendations'],
            'summary': {
                'attack_type': threat_analysis['detection']['attack_type'],
                'confidence': threat_analysis['detection']['confidence'],
                'risk_level': threat_analysis['analysis'].get('risk_level', 'Low')
            }
        }
        
        return report


# Test function
def test_gemini_agent():
    """Test the Gemini Cyber Agent"""
    print("="*60)
    print("🧪 Testing Gemini Cyber Agent")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ GEMINI_API_KEY not found in .env file")
        print("\n💡 Create .env file with:")
        print("   GEMINI_API_KEY=your-api-key-here")
        return
    
    try:
        agent = GeminiCyberAgent()
        
        # Test DDoS detection
        print("\n📊 Test 1: DDoS Attack Detection")
        analysis = agent.analyze_threat(
            prediction=1,
            confidence=0.9997,
            attack_type='DDoS'
        )
        
        print("\n📋 Gemini Analysis:")
        print(analysis['llm_analysis'])
        
        print("\n📊 Parsed Analysis:")
        print(f"  Severity: {analysis['analysis'].get('risk_level', 'Unknown')}")
        print(f"  Impact: {analysis['analysis'].get('impact', 'Unknown')}")
        print(f"  Summary: {analysis['analysis'].get('summary', 'No summary')}")
        
        # Generate report
        report = agent.generate_incident_report(analysis)
        print(f"\n📄 Incident Report: {report['incident_id']}")
        print(f"  Status: {report['status']}")
        print(f"  Priority: {report['priority']}")
        
        # Test BENIGN traffic
        print("\n📊 Test 2: BENIGN Traffic")
        analysis2 = agent.analyze_threat(
            prediction=0,
            confidence=0.9998,
            attack_type='BENIGN'
        )
        
        if analysis2.get('llm_analysis'):
            print(f"\n📋 Gemini Analysis for BENIGN:")
            print(analysis2['llm_analysis'][:200] + "...")
        else:
            print("No response for BENIGN test")
        
        print("\n✅ Gemini Cyber Agent test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure:")
        print("1. GEMINI_API_KEY is set in .env file")
        print("2. Package installed: pip install google-genai")
        print("3. Your API key is valid and has Gemini access")


if __name__ == "__main__":
    test_gemini_agent()

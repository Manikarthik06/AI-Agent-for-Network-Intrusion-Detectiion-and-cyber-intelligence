"""
Cyber Threat Intelligence Knowledge Base for NIDS
Maps detected attacks to MITRE ATT&CK framework
"""

class ThreatIntelligence:
    """Knowledge base for network intrusion threats"""
    
    # MITRE ATT&CK mappings for common attack types
    MITRE_ATTACK_MAPPINGS = {
        'DDoS': {
            'technique_id': 'T1498',
            'technique_name': 'Network Denial of Service',
            'tactic': 'Impact',
            'description': 'Adversaries may perform Network Denial of Service (DoS) attacks to degrade or block the availability of targeted resources to users.',
            'mitigations': [
                'M1030 - Network Segmentation',
                'M1031 - Network Intrusion Prevention',
                'M1035 - Limit Access to Resource Over Network'
            ]
        },
        'PortScan': {
            'technique_id': 'T1046',
            'technique_name': 'Network Service Scanning',
            'tactic': 'Reconnaissance',
            'description': 'Adversaries may attempt to scan for open ports and services to identify potential targets.',
            'mitigations': [
                'M1030 - Network Segmentation',
                'M1042 - Disable or Remove Feature or Program'
            ]
        },
        'BruteForce': {
            'technique_id': 'T1110',
            'technique_name': 'Brute Force',
            'tactic': 'Credential Access',
            'description': 'Adversaries may use brute force techniques to gain access to accounts.',
            'mitigations': [
                'M1032 - Multi-factor Authentication',
                'M1027 - Password Policies'
            ]
        },
        'BENIGN': {
            'technique_id': 'N/A',
            'technique_name': 'Normal Traffic',
            'tactic': 'N/A',
            'description': 'Normal network traffic that does not exhibit malicious behavior.',
            'mitigations': []
        }
    }
    
    # Severity levels
    SEVERITY = {
        'DDoS': 'Critical',
        'PortScan': 'Medium',
        'BruteForce': 'High',
        'BENIGN': 'None'
    }
    
    # Recommended actions
    RECOMMENDATIONS = {
        'DDoS': {
            'immediate': [
                'Block source IP addresses',
                'Enable rate limiting',
                'Activate DDoS protection services',
                'Notify security team immediately'
            ],
            'investigation': [
                'Analyze traffic patterns',
                'Identify attack vectors',
                'Check for compromised systems',
                'Review logs for similar patterns'
            ],
            'long_term': [
                'Implement DDoS mitigation strategy',
                'Review network architecture',
                'Deploy intrusion prevention systems',
                'Create incident response plan'
            ]
        },
        'PortScan': {
            'immediate': [
                'Monitor source IP for further activity',
                'Review firewall logs',
                'Check for vulnerability scanning'
            ],
            'investigation': [
                'Identify scanning patterns',
                'Check if any ports were exploited',
                'Review system logs for access attempts'
            ],
            'long_term': [
                'Harden network services',
                'Implement network segmentation',
                'Deploy IDS/IPS for scanning detection'
            ]
        },
        'BruteForce': {
            'immediate': [
                'Lock affected accounts',
                'Block source IP addresses',
                'Check for compromised credentials'
            ],
            'investigation': [
                'Identify compromised accounts',
                'Review authentication logs',
                'Check for successful logins'
            ],
            'long_term': [
                'Enforce multi-factor authentication',
                'Implement strong password policies',
                'Deploy account lockout policies'
            ]
        },
        'BENIGN': {
            'immediate': [],
            'investigation': ['Monitor for normal patterns'],
            'long_term': ['Maintain current security posture']
        }
    }
    
    @classmethod
    def get_threat_info(cls, attack_type):
        """Get threat intelligence for an attack type"""
        if attack_type in cls.MITRE_ATTACK_MAPPINGS:
            return {
                'attack_type': attack_type,
                'severity': cls.SEVERITY.get(attack_type, 'Unknown'),
                'mitre': cls.MITRE_ATTACK_MAPPINGS[attack_type],
                'recommendations': cls.RECOMMENDATIONS.get(attack_type, {})
            }
        return None
    
    @classmethod
    def get_all_threats(cls):
        """Get all threat intelligence"""
        threats = {}
        for attack_type in cls.MITRE_ATTACK_MAPPINGS.keys():
            threats[attack_type] = cls.get_threat_info(attack_type)
        return threats

# Test the CTI layer
if __name__ == "__main__":
    print("="*60)
    print("CYBER THREAT INTELLIGENCE - TEST")
    print("="*60)
    
    # Test for DDoS
    print("\nTesting DDoS Threat Intelligence:")
    ddos_info = ThreatIntelligence.get_threat_info('DDoS')
    if ddos_info:
        print(f"  Attack Type: {ddos_info['attack_type']}")
        print(f"  Severity: {ddos_info['severity']}")
        print(f"  MITRE ID: {ddos_info['mitre']['technique_id']}")
        print(f"  Technique: {ddos_info['mitre']['technique_name']}")
        print(f"  Immediate Actions: {ddos_info['recommendations']['immediate'][:2]}")
    
    # Test for BENIGN
    print("\nTesting BENIGN Traffic:")
    benign_info = ThreatIntelligence.get_threat_info('BENIGN')
    if benign_info:
        print(f"  Attack Type: {benign_info['attack_type']}")
        print(f"  Severity: {benign_info['severity']}")
        print(f"  Status: Normal traffic")
    
    print("\n✅ CTI Layer Test Complete!")

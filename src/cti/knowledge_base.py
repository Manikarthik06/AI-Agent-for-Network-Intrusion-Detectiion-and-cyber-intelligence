"""
Cyber Threat Intelligence Knowledge Base for NIDS
Maps all 15 CIC-IDS2017 attack types to MITRE ATT&CK
"""

class ThreatIntelligence:
    MITRE_ATTACK_MAPPINGS = {
        'BENIGN': {
            'technique_id': 'N/A',
            'technique_name': 'Normal Traffic',
            'tactic': 'N/A',
            'description': 'Normal network traffic.',
            'mitigations': []
        },
        'DDoS': {
            'technique_id': 'T1498',
            'technique_name': 'Network Denial of Service',
            'tactic': 'Impact',
            'description': 'Adversaries may perform Network Denial of Service (DoS) attacks to degrade or block availability.',
            'mitigations': ['M1030', 'M1031', 'M1035']
        },
        'DoS GoldenEye': {
            'technique_id': 'T1498',
            'technique_name': 'Network Denial of Service',
            'tactic': 'Impact',
            'description': 'Adversaries may perform a DoS attack using GoldenEye tool.',
            'mitigations': ['M1030', 'M1031']
        },
        'DoS Hulk': {
            'technique_id': 'T1498',
            'technique_name': 'Network Denial of Service',
            'tactic': 'Impact',
            'description': 'Adversaries may perform a DoS attack using Hulk tool.',
            'mitigations': ['M1030', 'M1031']
        },
        'DoS Slowhttptest': {
            'technique_id': 'T1498',
            'technique_name': 'Network Denial of Service',
            'tactic': 'Impact',
            'description': 'Adversaries may perform a DoS attack using Slowhttptest.',
            'mitigations': ['M1030', 'M1031']
        },
        'DoS slowloris': {
            'technique_id': 'T1498',
            'technique_name': 'Network Denial of Service',
            'tactic': 'Impact',
            'description': 'Adversaries may perform a DoS attack using Slowloris.',
            'mitigations': ['M1030', 'M1031']
        },
        'PortScan': {
            'technique_id': 'T1046',
            'technique_name': 'Network Service Scanning',
            'tactic': 'Reconnaissance',
            'description': 'Adversaries may attempt to scan for open ports and services.',
            'mitigations': ['M1030', 'M1042']
        },
        'Bot': {
            'technique_id': 'T1043',
            'technique_name': 'Commonly Used Port',
            'tactic': 'Command and Control',
            'description': 'Adversaries may use commonly used ports to communicate with compromised hosts.',
            'mitigations': ['M1030', 'M1035']
        },
        'FTP-Patator': {
            'technique_id': 'T1110',
            'technique_name': 'Brute Force',
            'tactic': 'Credential Access',
            'description': 'Adversaries may use brute force techniques against FTP services.',
            'mitigations': ['M1032', 'M1027']
        },
        'SSH-Patator': {
            'technique_id': 'T1110',
            'technique_name': 'Brute Force',
            'tactic': 'Credential Access',
            'description': 'Adversaries may use brute force techniques against SSH services.',
            'mitigations': ['M1032', 'M1027']
        },
        'Web Attack – Brute Force': {
            'technique_id': 'T1110',
            'technique_name': 'Brute Force',
            'tactic': 'Credential Access',
            'description': 'Adversaries may use brute force techniques against web applications.',
            'mitigations': ['M1032', 'M1027', 'M1036']
        },
        'Web Attack – Sql Injection': {
            'technique_id': 'T1190',
            'technique_name': 'Exploit Public-Facing Application',
            'tactic': 'Initial Access',
            'description': 'Adversaries may exploit SQL injection vulnerabilities in web applications.',
            'mitigations': ['M1050', 'M1018', 'M1026']
        },
        'Web Attack – XSS': {
            'technique_id': 'T1203',
            'technique_name': 'Exploitation for Client Execution',
            'tactic': 'Execution',
            'description': 'Adversaries may exploit cross-site scripting vulnerabilities.',
            'mitigations': ['M1050', 'M1048']
        },
        'Infiltration': {
            'technique_id': 'T1071',
            'technique_name': 'Application Layer Protocol',
            'tactic': 'Command and Control',
            'description': 'Adversaries may use standard application layer protocols to communicate.',
            'mitigations': ['M1031', 'M1035']
        },
        'Heartbleed': {
            'technique_id': 'T1190',
            'technique_name': 'Exploit Public-Facing Application',
            'tactic': 'Initial Access',
            'description': 'Adversaries may exploit the Heartbleed vulnerability.',
            'mitigations': ['M1050', 'M1018']
        }
    }

    SEVERITY = {
        'BENIGN': 'None',
        'DDoS': 'Critical',
        'DoS GoldenEye': 'Critical',
        'DoS Hulk': 'Critical',
        'DoS Slowhttptest': 'Critical',
        'DoS slowloris': 'Critical',
        'PortScan': 'Medium',
        'Bot': 'Medium',
        'FTP-Patator': 'High',
        'SSH-Patator': 'High',
        'Web Attack – Brute Force': 'High',
        'Web Attack – Sql Injection': 'Critical',
        'Web Attack – XSS': 'High',
        'Infiltration': 'Medium',
        'Heartbleed': 'Critical'
    }

    RECOMMENDATIONS = {
        'Web Attack – Sql Injection': {
            'immediate': ['Block malicious SQL queries', 'Enable Web Application Firewall (WAF)', 'Isolate affected databases'],
            'investigation': ['Analyze SQL query logs', 'Check for data exfiltration', 'Review application logs'],
            'long_term': ['Implement parameterized queries', 'Conduct code review', 'Regular security testing']
        },
        'Web Attack – XSS': {
            'immediate': ['Block malicious scripts', 'Apply input validation', 'Enable Content Security Policy (CSP)'],
            'investigation': ['Analyze web server logs', 'Identify injected scripts', 'Check for cookie theft'],
            'long_term': ['Implement output encoding', 'Use CSP', 'Regular penetration testing']
        },
        'Web Attack – Brute Force': {
            'immediate': ['Lock affected accounts', 'Block source IP addresses', 'Enable CAPTCHA on login pages'],
            'investigation': ['Identify compromised accounts', 'Review authentication logs', 'Check for successful logins'],
            'long_term': ['Enforce multi-factor authentication', 'Implement strong password policies', 'Deploy account lockout policies']
        },
        'PortScan': {
            'immediate': ['Monitor source IP for further activity', 'Review firewall logs'],
            'investigation': ['Identify scanning patterns', 'Check if any ports were exploited'],
            'long_term': ['Harden network services', 'Implement network segmentation']
        },
        'Bot': {
            'immediate': ['Block suspicious IPs', 'Review network logs'],
            'investigation': ['Identify compromised hosts', 'Analyze communication patterns'],
            'long_term': ['Implement threat hunting', 'Update security policies']
        },
        'Infiltration': {
            'immediate': ['Isolate affected systems', 'Block suspicious connections'],
            'investigation': ['Identify entry point', 'Check for data exfiltration'],
            'long_term': ['Review network architecture', 'Implement zero-trust principles']
        },
        'Heartbleed': {
            'immediate': ['Patch OpenSSL vulnerabilities', 'Revoke compromised SSL certificates'],
            'investigation': ['Check for data exposure', 'Review access logs'],
            'long_term': ['Regular vulnerability scanning', 'Keep software updated']
        },
        'FTP-Patator': {
            'immediate': ['Block source IPs', 'Enable account lockout'],
            'investigation': ['Review FTP logs', 'Check for compromised credentials'],
            'long_term': ['Move to SFTP/FTPS', 'Implement strong password policies']
        },
        'SSH-Patator': {
            'immediate': ['Block source IPs', 'Enable account lockout'],
            'investigation': ['Review SSH logs', 'Check for compromised credentials'],
            'long_term': ['Use key-based authentication', 'Implement strong password policies']
        }
    }

    @classmethod
    def get_threat_info(cls, attack_type):
        if attack_type in cls.MITRE_ATTACK_MAPPINGS:
            return {
                'attack_type': attack_type,
                'severity': cls.SEVERITY.get(attack_type, 'Unknown'),
                'mitre': cls.MITRE_ATTACK_MAPPINGS[attack_type],
                'recommendations': cls.RECOMMENDATIONS.get(attack_type, {
                    'immediate': ['Investigate the threat', 'Block suspicious IPs'],
                    'investigation': ['Analyze attack patterns', 'Review logs'],
                    'long_term': ['Update security measures']
                })
            }
        return {
            'attack_type': attack_type,
            'severity': 'Medium',
            'mitre': {
                'technique_id': 'Unknown',
                'technique_name': attack_type,
                'tactic': 'Unknown',
                'description': f'{attack_type} attack detected.',
                'mitigations': ['Monitor traffic', 'Analyze logs']
            },
            'recommendations': {
                'immediate': ['Block suspicious IPs', 'Enable additional monitoring'],
                'investigation': ['Analyze attack patterns', 'Check for known signatures'],
                'long_term': ['Update security policies', 'Implement threat hunting']
            }
        }
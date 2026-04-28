"""
Assignment 9 - Test Suite for Fraud Engine Module
VeriSupport AI-Based Customer Support Automation Platform

Tests the calculate_trust_score function from backend/fraud_engine.py
which is the core risk-scoring module of the platform.
"""

import sys
import os
import unittest
from datetime import datetime

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'neuradesk', 'backend'))

from fraud_engine import calculate_trust_score


class TestFraudEngine(unittest.TestCase):
    """Test suite for the Fraud Engine - calculate_trust_score function"""

    # =========================================================================
    # TC-01: Legitimate ticket with no image - expect Low Risk
    # =========================================================================
    def test_tc01_legitimate_ticket_no_image(self):
        """TC-01: A genuine customer with positive sentiment and no image should get high trust score"""
        ai_analysis = {
            'sentiment': 'neutral',
            'entities': ['order', 'delivery', 'late']
        }
        customer = {
            'tier': 'Gold',
            'orders': 15,
            'ltv': 5000.0
        }
        vision_analysis = None

        result = calculate_trust_score(ai_analysis, customer, vision_analysis)

        self.assertEqual(result['verdict'], 'Low Risk')
        self.assertGreaterEqual(result['trustScore'], 80)
        self.assertEqual(result['anomalies'], 0)
        self.assertEqual(len(result['riskFactors']), 0)
        print(f"  TC-01 PASS | Trust Score: {result['trustScore']} | Verdict: {result['verdict']}")

    # =========================================================================
    # TC-02: Fraudulent image with high tampering score - expect High Risk
    # =========================================================================
    def test_tc02_high_tampering_fraud_image(self):
        """TC-02: An image with high tampering score should result in critical fraud detection"""
        ai_analysis = {
            'sentiment': 'angry',
            'entities': ['cockroach', 'refund']
        }
        customer = {
            'tier': 'Standard',
            'orders': 1,
            'ltv': 50.0
        }
        vision_analysis = {
            'tamperingScore': 0.85,
            'elaAnomaly': True,
            'metadataConsistent': False,
            'aiVision': {'fraudRisk': 'high'},
            'verdict': 'FRAUD DETECTED - Fake Food Image'
        }

        result = calculate_trust_score(ai_analysis, customer, vision_analysis)

        self.assertEqual(result['verdict'], 'High Risk')
        self.assertLess(result['trustScore'], 50)
        self.assertIn('CRITICAL: Fake food image detected', result['riskFactors'])
        self.assertIn('Image metadata shows editing software', result['riskFactors'])
        print(f"  TC-02 PASS | Trust Score: {result['trustScore']} | Verdict: {result['verdict']}")

    # =========================================================================
    # TC-03: Medium tampering image - expect Medium Risk or lower
    # =========================================================================
    def test_tc03_medium_tampering_image(self):
        """TC-03: A medium tampering score should flag suspicious tampering"""
        ai_analysis = {
            'sentiment': 'neutral',
            'entities': ['quality', 'food']
        }
        customer = {
            'tier': 'Premium',
            'orders': 10,
            'ltv': 2000.0
        }
        vision_analysis = {
            'tamperingScore': 0.55,
            'elaAnomaly': True,
            'metadataConsistent': True,
            'aiVision': {'fraudRisk': 'medium'},
            'verdict': 'Medium Risk - Suspicious Patterns'
        }

        result = calculate_trust_score(ai_analysis, customer, vision_analysis)

        self.assertIn(result['verdict'], ['Medium Risk', 'High Risk'])
        self.assertIn('Suspicious image tampering detected', result['riskFactors'])
        self.assertGreater(result['anomalies'], 0)
        print(f"  TC-03 PASS | Trust Score: {result['trustScore']} | Verdict: {result['verdict']}")

    # =========================================================================
    # TC-04: Angry sentiment should deduct score
    # =========================================================================
    def test_tc04_angry_sentiment_deduction(self):
        """TC-04: Angry sentiment should reduce trust score by 5 points"""
        ai_analysis_neutral = {
            'sentiment': 'neutral',
            'entities': []
        }
        ai_analysis_angry = {
            'sentiment': 'angry',
            'entities': []
        }
        customer = {
            'tier': 'Gold',
            'orders': 20,
            'ltv': 10000.0
        }

        result_neutral = calculate_trust_score(ai_analysis_neutral, customer, None)
        result_angry = calculate_trust_score(ai_analysis_angry, customer, None)

        self.assertEqual(result_neutral['trustScore'] - result_angry['trustScore'], 5)
        self.assertIn('High negative sentiment', result_angry['riskFactors'])
        print(f"  TC-04 PASS | Neutral Score: {result_neutral['trustScore']} | Angry Score: {result_angry['trustScore']} | Diff: {result_neutral['trustScore'] - result_angry['trustScore']}")

    # =========================================================================
    # TC-05: New account with low order history
    # =========================================================================
    def test_tc05_new_account_low_history(self):
        """TC-05: A Standard-tier customer with < 3 orders should get 15-point deduction"""
        ai_analysis = {
            'sentiment': 'neutral',
            'entities': []
        }
        customer = {
            'tier': 'Standard',
            'orders': 2,
            'ltv': 80.0
        }

        result = calculate_trust_score(ai_analysis, customer, None)

        self.assertEqual(result['trustScore'], 85)
        self.assertIn('New account with low history', result['riskFactors'])
        print(f"  TC-05 PASS | Trust Score: {result['trustScore']} | Risk Factors: {result['riskFactors']}")

    # =========================================================================
    # TC-06: Suspicious keywords in entities
    # =========================================================================
    def test_tc06_suspicious_keywords_detection(self):
        """TC-06: Entities containing 'sue' or 'lawyer' should trigger suspicious keyword flag"""
        ai_analysis = {
            'sentiment': 'angry',
            'entities': ['lawyer', 'sue', 'refund']
        }
        customer = {
            'tier': 'Standard',
            'orders': 5,
            'ltv': 200.0
        }

        result = calculate_trust_score(ai_analysis, customer, None)

        # Should have deductions for both suspicious keywords and angry sentiment
        self.assertLess(result['trustScore'], 80)
        keyword_flags = [f for f in result['riskFactors'] if 'Suspicious keyword' in f]
        self.assertGreater(len(keyword_flags), 0)
        print(f"  TC-06 PASS | Trust Score: {result['trustScore']} | Keyword Flags: {keyword_flags}")

    # =========================================================================
    # TC-07: Metadata inconsistency in image
    # =========================================================================
    def test_tc07_metadata_inconsistency(self):
        """TC-07: Image with inconsistent metadata should get 15-point deduction"""
        ai_analysis = {
            'sentiment': 'neutral',
            'entities': []
        }
        customer = {
            'tier': 'Gold',
            'orders': 10,
            'ltv': 3000.0
        }
        vision_analysis = {
            'tamperingScore': 0.10,
            'elaAnomaly': False,
            'metadataConsistent': False,
            'aiVision': {'fraudRisk': 'low'},
            'verdict': 'Low Risk - Appears Authentic'
        }

        result = calculate_trust_score(ai_analysis, customer, vision_analysis)

        self.assertIn('Image metadata shows editing software', result['riskFactors'])
        self.assertEqual(result['trustScore'], 85)
        print(f"  TC-07 PASS | Trust Score: {result['trustScore']} | Risk Factors: {result['riskFactors']}")

    # =========================================================================
    # TC-08: Combined worst-case scenario
    # =========================================================================
    def test_tc08_worst_case_combined(self):
        """TC-08: All risk factors combined should produce minimum trust score"""
        ai_analysis = {
            'sentiment': 'angry',
            'entities': ['lawyer', 'sue', 'police', 'scam']
        }
        customer = {
            'tier': 'Standard',
            'orders': 1,
            'ltv': 10.0
        }
        vision_analysis = {
            'tamperingScore': 0.95,
            'elaAnomaly': True,
            'metadataConsistent': False,
            'aiVision': {'fraudRisk': 'high'},
            'verdict': 'FRAUD DETECTED - Fake Food Image'
        }

        result = calculate_trust_score(ai_analysis, customer, vision_analysis)

        self.assertEqual(result['verdict'], 'High Risk')
        self.assertEqual(result['trustScore'], 0)  # Should be clamped to 0
        self.assertGreater(len(result['riskFactors']), 3)
        self.assertGreater(result['anomalies'], 3)
        print(f"  TC-08 PASS | Trust Score: {result['trustScore']} | Verdict: {result['verdict']} | Factors: {len(result['riskFactors'])}")

    # =========================================================================
    # TC-09: Boundary test - minor image inconsistencies
    # =========================================================================
    def test_tc09_minor_image_inconsistency(self):
        """TC-09: Tampering score between 0.2 and 0.4 should flag minor inconsistencies"""
        ai_analysis = {
            'sentiment': 'neutral',
            'entities': []
        }
        customer = {
            'tier': 'Gold',
            'orders': 10,
            'ltv': 3000.0
        }
        vision_analysis = {
            'tamperingScore': 0.25,
            'elaAnomaly': False,
            'metadataConsistent': True,
            'aiVision': {'fraudRisk': 'low'},
            'verdict': 'Low Risk'
        }

        result = calculate_trust_score(ai_analysis, customer, vision_analysis)

        self.assertIn('Minor image inconsistencies', result['riskFactors'])
        self.assertEqual(result['trustScore'], 90)
        print(f"  TC-09 PASS | Trust Score: {result['trustScore']} | Risk Factors: {result['riskFactors']}")

    # =========================================================================
    # TC-10: No vision, no bad signals - perfect score (100)
    # =========================================================================
    def test_tc10_perfect_score(self):
        """TC-10: A perfect customer with no risk signals should get score of 100"""
        ai_analysis = {
            'sentiment': 'positive',
            'entities': ['thank you', 'delivery']
        }
        customer = {
            'tier': 'Premium',
            'orders': 50,
            'ltv': 25000.0
        }

        result = calculate_trust_score(ai_analysis, customer, None)

        self.assertEqual(result['trustScore'], 100)
        self.assertEqual(result['verdict'], 'Low Risk')
        self.assertEqual(len(result['riskFactors']), 0)
        self.assertEqual(result['anomalies'], 0)
        print(f"  TC-10 PASS | Trust Score: {result['trustScore']} | Verdict: {result['verdict']}")


if __name__ == '__main__':
    print("=" * 70)
    print("  VERISUPPORT - FRAUD ENGINE TEST SUITE EXECUTION")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Run tests with verbose output
    unittest.main(verbosity=2)

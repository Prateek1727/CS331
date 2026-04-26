def calculate_trust_score(ai_analysis: dict, customer: dict, vision_analysis: dict = None) -> dict:
    """
    Enhanced fraud engine that considers text analysis, customer history, and image tampering.
    Real-world would use HuggingFace tabular models or XGBoost.
    """
    base_score = 100
    risk_factors = []
    anomalies = 0
    
    # Check for image fraud (highest priority)
    if vision_analysis:
        tampering_score = vision_analysis.get('tamperingScore', 0)
        fraud_risk = vision_analysis.get('aiVision', {}).get('fraudRisk', 'low')
        
        if fraud_risk == 'high' or tampering_score > 0.7:
            base_score -= 60
            risk_factors.append("CRITICAL: Fake food image detected")
            anomalies += 3
        elif fraud_risk == 'medium' or tampering_score > 0.4:
            base_score -= 30
            risk_factors.append("Suspicious image tampering detected")
            anomalies += 2
        elif tampering_score > 0.2:
            base_score -= 10
            risk_factors.append("Minor image inconsistencies")
            anomalies += 1
            
        if not vision_analysis.get('metadataConsistent', True):
            base_score -= 15
            risk_factors.append("Image metadata shows editing software")
            anomalies += 1
    
    # Text sentiment analysis
    if ai_analysis.get('sentiment') == 'angry':
        base_score -= 5
        risk_factors.append("High negative sentiment")
        
    # Check customer standing
    if customer.get('tier') == 'Standard' and customer.get('orders', 0) < 3:
        base_score -= 15
        risk_factors.append("New account with low history")
        
    # Example keyword flags
    suspicious_phrases = ["sue", "lawyer", "police", "scam", "viral", "social media"]
    for phrase in suspicious_phrases:
        if any(phrase in entity.lower() for entity in ai_analysis.get('entities', [])):
            base_score -= 20
            risk_factors.append(f"Suspicious keyword: '{phrase}'")
            anomalies += 1
            
    # Trust verdict
    if base_score >= 80:
        verdict = "Low Risk"
    elif base_score >= 50:
        verdict = "Medium Risk"
    else:
        verdict = "High Risk"
        
    return {
        "trustScore": max(0, base_score),
        "riskFactors": risk_factors,
        "anomalies": anomalies,
        "verdict": verdict
    }

import os
import google.generativeai as genai
import json
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import io
from datetime import datetime

# Placeholder initialization - will load key from settings or env
def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        genai.configure(api_key=api_key)

def analyze_ticket_text(subject: str, message: str) -> dict:
    """
    Uses Gemini to extract intent, sentiment, entities, and confidence.
    """
    # If API key is not set, return simulated AI response
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_gemini_api_key_here":
        return {
            "intent": "general_inquiry",
            "sentiment": "neutral",
            "confidence": 0.85,
            "language": "en",
            "entities": []
        }
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze this customer support ticket.
        Subject: {subject}
        Message: {message}
        
        Respond ONLY with a valid JSON block structured like this:
        {{
            "intent": "string (e.g., refund_request, delivery_issue, technical_issue, product_inquiry)",
            "sentiment": "string (positive, neutral, negative, angry)",
            "confidence": float (between 0.0 and 1.0),
            "language": "string (e.g., en, es, fr)",
            "entities": ["list", "of", "important", "entities", "or", "products"]
        }}
        """
        response = model.generate_content(prompt)
        # Parse JSON from response
        # It's better to strip markdown code blocks if gemini returns them
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text_response)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {
            "intent": "error",
            "sentiment": "neutral",
            "confidence": 0.0,
            "language": "en",
            "entities": []
        }

def analyze_image_forensics(image_bytes: bytes) -> dict:
    """
    Performs Error Level Analysis (ELA) to detect image tampering.
    Detects if insects or foreign objects were added to food images.
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Perform ELA (Error Level Analysis)
        # Save at quality 90 and compare
        temp_buffer = io.BytesIO()
        img.save(temp_buffer, 'JPEG', quality=90)
        temp_buffer.seek(0)
        compressed_img = Image.open(temp_buffer)
        
        # Calculate difference
        ela_img = ImageChops.difference(img, compressed_img)
        
        # Enhance to make differences visible
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
        
        # Convert to numpy for analysis
        ela_array = np.array(ela_img)
        
        # Calculate tampering score based on ELA variance
        ela_variance = np.var(ela_array)
        ela_mean = np.mean(ela_array)
        
        # Normalize tampering score (0-1)
        tampering_score = min(ela_variance / 10000.0, 1.0)
        
        # Detect anomalies (high variance regions indicate potential tampering)
        ela_anomaly = ela_variance > 500 or ela_mean > 30
        
        # Check metadata consistency
        metadata_consistent = True
        if hasattr(img, '_getexif') and img._getexif():
            exif_data = img._getexif()
            # Check for common manipulation software signatures
            if exif_data and any(key in str(exif_data) for key in ['Photoshop', 'GIMP', 'Paint']):
                metadata_consistent = False
        
        # Determine verdict
        if tampering_score > 0.6 or ela_anomaly:
            verdict = "High Tampering Risk - Likely Manipulated"
        elif tampering_score > 0.3:
            verdict = "Medium Risk - Suspicious Patterns"
        else:
            verdict = "Low Risk - Appears Authentic"
        
        return {
            "tamperingScore": round(tampering_score, 3),
            "elaAnomaly": bool(ela_anomaly),
            "metadataConsistent": bool(metadata_consistent),
            "verdict": verdict,
            "elaVariance": round(float(ela_variance), 2),
            "elaMean": round(float(ela_mean), 2)
        }
        
    except Exception as e:
        print(f"Image forensics error: {e}")
        return {
            "tamperingScore": 0.0,
            "elaAnomaly": False,
            "metadataConsistent": True,
            "verdict": "Analysis Failed",
            "error": str(e)
        }

def analyze_food_image_with_gemini(image_bytes: bytes) -> dict:
    """
    Uses Gemini Vision to detect fake food images, insects, or foreign objects.
    Specifically designed to catch fraudulent refund attempts.
    """
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_gemini_api_key_here":
        return {
            "hasFoodIssue": False,
            "issueType": "none",
            "confidence": 0.0,
            "description": "Gemini API key not configured",
            "fraudRisk": "unknown"
        }
    
    try:
        # Load image for Gemini
        img = Image.open(io.BytesIO(image_bytes))
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        Analyze this food delivery image carefully for fraud detection purposes.
        
        Check for:
        1. Insects (cockroaches, flies, ants, etc.) that appear to be digitally added
        2. Foreign objects that seem out of place or manipulated
        3. Signs of image editing or tampering
        4. Unnatural lighting or shadows on specific objects
        5. Inconsistent image quality in different regions
        6. Legitimate food quality issues vs. fabricated ones
        
        Respond ONLY with valid JSON:
        {
            "hasFoodIssue": boolean (true if there's a real or fake issue),
            "issueType": "string (none, insect_added, foreign_object, poor_quality, legitimate_issue)",
            "confidence": float (0.0 to 1.0),
            "description": "string (detailed description of what you see)",
            "fraudRisk": "string (low, medium, high)",
            "suspiciousElements": ["list", "of", "suspicious", "elements"],
            "reasoning": "string (why you think this is real or fake)"
        }
        """
        
        response = model.generate_content([prompt, img])
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text_response)
        
        return result
        
    except Exception as e:
        print(f"Gemini Vision API Error: {e}")
        return {
            "hasFoodIssue": False,
            "issueType": "error",
            "confidence": 0.0,
            "description": f"Analysis error: {str(e)}",
            "fraudRisk": "unknown"
        }

def analyze_image_complete(image_bytes: bytes) -> dict:
    """
    Complete image analysis combining forensics and AI vision.
    """
    if not image_bytes:
        return None
    
    # Run forensic analysis (ELA)
    forensics = analyze_image_forensics(image_bytes)
    
    # Run Gemini Vision analysis
    vision_analysis = analyze_food_image_with_gemini(image_bytes)
    
    # Combine results for final verdict
    combined_tampering_score = float((forensics["tamperingScore"] + 
                                (1.0 if vision_analysis.get("fraudRisk") == "high" else 
                                 0.5 if vision_analysis.get("fraudRisk") == "medium" else 0.0)) / 2)
    
    final_verdict = forensics["verdict"]
    if vision_analysis.get("fraudRisk") == "high":
        final_verdict = "FRAUD DETECTED - Fake Food Image"
    elif vision_analysis.get("hasFoodIssue") and vision_analysis.get("fraudRisk") == "low":
        final_verdict = "Legitimate Issue Detected"
    
    return {
        "tamperingScore": round(float(combined_tampering_score), 3),
        "elaAnomaly": bool(forensics["elaAnomaly"]),
        "metadataConsistent": bool(forensics["metadataConsistent"]),
        "verdict": final_verdict,
        "forensics": forensics,
        "aiVision": vision_analysis
    }

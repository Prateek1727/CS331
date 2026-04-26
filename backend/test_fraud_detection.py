"""
Test script for fake food image detection system
Run this to verify your setup is working correctly
"""

import os
from dotenv import load_dotenv
from ai_engine import analyze_image_forensics, analyze_food_image_with_gemini, analyze_image_complete
from PIL import Image
import io

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test if Gemini API key is configured"""
    print("=" * 60)
    print("TEST 1: Gemini API Configuration")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ FAILED: Gemini API key not configured")
        print("   Please add your API key to backend/.env")
        return False
    else:
        print("✅ PASSED: Gemini API key is configured")
        print(f"   Key starts with: {api_key[:10]}...")
        return True

def test_image_forensics():
    """Test ELA forensics on a sample image"""
    print("\n" + "=" * 60)
    print("TEST 2: Image Forensics (ELA)")
    print("=" * 60)
    
    try:
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        result = analyze_image_forensics(img_bytes)
        
        print("✅ PASSED: Image forensics working")
        print(f"   Tampering Score: {result['tamperingScore']}")
        print(f"   ELA Anomaly: {result['elaAnomaly']}")
        print(f"   Verdict: {result['verdict']}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("\n" + "=" * 60)
    print("TEST 3: Dependencies Check")
    print("=" * 60)
    
    dependencies = {
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'google.generativeai': 'google-generativeai',
        'fastapi': 'fastapi',
        'pydantic': 'pydantic'
    }
    
    all_installed = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} NOT installed - run: pip install {package}")
            all_installed = False
    
    return all_installed

def test_complete_analysis():
    """Test complete image analysis pipeline"""
    print("\n" + "=" * 60)
    print("TEST 4: Complete Analysis Pipeline")
    print("=" * 60)
    
    try:
        # Create a test image
        img = Image.new('RGB', (200, 200), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        result = analyze_image_complete(img_bytes)
        
        if result:
            print("✅ PASSED: Complete analysis pipeline working")
            print(f"   Combined Tampering Score: {result['tamperingScore']}")
            print(f"   Final Verdict: {result['verdict']}")
            
            if 'aiVision' in result:
                print(f"   AI Fraud Risk: {result['aiVision'].get('fraudRisk', 'N/A')}")
            
            return True
        else:
            print("❌ FAILED: Analysis returned None")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🔍" * 30)
    print("FAKE FOOD IMAGE DETECTION - SYSTEM TEST")
    print("🔍" * 30 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Gemini API", test_gemini_api()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Image Forensics", test_image_forensics()))
    results.append(("Complete Pipeline", test_complete_analysis()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Start backend: cd backend && python -m uvicorn main:app --reload")
        print("2. Start frontend: npm run dev")
        print("3. Go to http://localhost:5173/customer-portal")
        print("4. Upload a test image and submit a ticket")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Add Gemini API key to backend/.env")
        print("- Make sure you're in the backend directory")

if __name__ == "__main__":
    main()

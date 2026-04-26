# 🚀 Quick Start Guide - Fake Food Image Detection

## What This Does
This platform detects **fake food images** (like insects photoshopped into food) to prevent fraudulent refund claims from food delivery platforms like Zomato, Swiggy, etc.

## ⚡ Quick Setup (5 minutes)

### Step 1: Get Gemini API Key (FREE)
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

### Step 2: Configure Backend
```bash
cd backend
cp .env.example .env
```

Edit `backend/.env` and paste your API key:
```
GEMINI_API_KEY=paste_your_key_here
```

### Step 3: Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd ..
npm install
```

### Step 4: Test Your Setup
```bash
cd backend
python test_fraud_detection.py
```

If all tests pass ✅, continue to Step 5!

### Step 5: Start the Application

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

### Step 6: Test Fraud Detection

1. Open: http://localhost:5173/customer-portal
2. Fill in the form
3. Upload a food image (try editing one with Photoshop/GIMP first!)
4. Submit and watch the AI analyze it

## 🎯 What to Test

### Test 1: Normal Image
- Upload a regular, unedited food photo
- Should show: ✅ Low tampering score, authentic verdict

### Test 2: Edited Image
- Edit a food photo (add an insect, change colors, etc.)
- Upload the edited version
- Should show: ⚠️ High tampering score, fraud detected

### Test 3: No Image
- Submit without uploading an image
- Should process normally without vision analysis

## 📊 View Results

After submitting tickets:
- **Dashboard**: http://localhost:5173/ - See all tickets
- **AI Brain**: http://localhost:5173/ai-brain - Detailed analysis
- **Decision Engine**: http://localhost:5173/decision-engine - Fraud rules

## 🔍 How It Detects Fraud

1. **Error Level Analysis (ELA)** - Detects compression artifacts from editing
2. **Gemini Vision AI** - Identifies fake insects, foreign objects
3. **Metadata Check** - Looks for editing software signatures
4. **Fraud Scoring** - Combines all signals into risk score

## ⚠️ Troubleshooting

**"Gemini API key not configured"**
→ Add your API key to `backend/.env`

**"Module not found"**
→ Run `pip install -r requirements.txt` in backend folder

**"Connection refused"**
→ Make sure backend is running on port 8000

**Frontend won't start**
→ Run `npm install` first

## 📁 Project Structure

```
├── backend/
│   ├── main.py              # API endpoints
│   ├── ai_engine.py         # Image analysis (ELA + Gemini)
│   ├── fraud_engine.py      # Fraud scoring
│   ├── .env                 # Your API key goes here
│   └── test_fraud_detection.py  # Test script
├── src/
│   ├── pages/
│   │   ├── CustomerPortal.jsx   # Image upload form
│   │   ├── AIBrain.jsx          # Analysis visualization
│   │   └── Dashboard.jsx        # Ticket overview
│   └── services/
│       └── apiService.js        # API calls
└── FRAUD_DETECTION_SETUP.md     # Detailed documentation
```

## 🎓 Understanding the Results

**Tampering Score:**
- 0.0 - 0.3: Low risk (authentic)
- 0.3 - 0.6: Medium risk (suspicious)
- 0.6 - 1.0: High risk (likely fake)

**AI Fraud Risk:**
- Low: Image appears legitimate
- Medium: Some suspicious elements
- High: Clear signs of manipulation

**Verdict:**
- "Low Risk - Appears Authentic" → Auto-approve
- "Medium Risk - Suspicious Patterns" → Manual review
- "FRAUD DETECTED - Fake Food Image" → Escalate to fraud team

## 🚀 Next Steps

1. ✅ Test with various images
2. ✅ Check the AI Brain page for detailed analysis
3. ✅ Review fraud detection rules in Decision Engine
4. ✅ Adjust thresholds in `backend/fraud_engine.py` if needed

## 📚 Full Documentation

See `FRAUD_DETECTION_SETUP.md` for:
- Complete API documentation
- Architecture details
- Advanced configuration
- Production deployment guide

---

**Need Help?**
1. Run the test script: `python backend/test_fraud_detection.py`
2. Check backend logs in Terminal 1
3. Check browser console (F12) for frontend errors
4. Review `FRAUD_DETECTION_SETUP.md` for detailed troubleshooting

**Status: ✅ FULLY IMPLEMENTED - READY TO USE**

# Neuradesk: Fraudulent Image Detection Platform

Neuradesk is an advanced, AI-powered customer support and intelligence platform specifically designed to detect fake or manipulated food images. Built for food delivery platforms, it prevents fraudulent refund claims by automatically analyzing customer-submitted photos for tampering, such as digitally inserted insects or foreign objects.

## Core Architecture

The system is constructed with a modern, decoupled architecture:

### 1. Vision Intelligence Backend (Python/FastAPI)
The backend acts as the core analytical engine. It handles API requests, processes images, and applies sophisticated fraud scoring algorithms.
- **Error Level Analysis (ELA):** Algorithmically detects inconsistencies in image compression artifacts, highlighting areas that have been spliced or photoshopped.
- **Gemini Vision AI Analysis:** Utilizes LLM vision models to semantically interpret the image contents, identify foreign objects, and provide a plain-text reasoning of anomalies.
- **Metadata Inspection:** Extracts and scrutinizes EXIF data to locate traces of image editing software (such as Adobe Photoshop or GIMP).
- **Fraud Scoring Engine:** Aggregates the ELA, AI, and Metadata signals into a definitive Risk Score (0.0 to 1.0) and issues an automated verdict.

### 2. Interactive Frontend (React/Vite)
The frontend provides a real-time dashboard for administrators and a portal for customers.
- **Customer Portal:** A secure interface allowing users to upload photo evidence for their disputes. 
- **AI Brain Dashboard:** Visualizes the backend's analytical thought process, displaying ELA heatmaps and raw Gemini reasoning.
- **Decision Engine Dashboard:** Details the ruleset and thresholds used for auto-approving or flagging tickets.

## Installation and Setup

### Prerequisites
- Node.js (v14 or higher)
- Python (v3.8 or higher)
- Google Gemini API Key

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and insert your Gemini API Key.
5. Boot the FastAPI server:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```

### Frontend Setup

1. Open a new terminal and navigate to the project root directory.
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

## Usage Workflow

1. Navigate to the Customer Portal at `http://localhost:5173/customer-portal`.
2. Upload a deliberately manipulated food image (for instance, an image with an insect added via photo editing software).
3. Submit the ticket. 
4. The backend immediately processes the image. Navigate to the AI Brain interface to see the detection pipeline in action.
5. Tickets flagged as High Risk (Risk Score > 0.6) are automatically escalated to the fraud protection team for manual review.

## API Integration

The platform provides RESTful endpoints. The primary analysis route accepts multipart form-data image uploads and returns a JSON payload containing the tampering score, semantic reasoning, and the final automated verification verdict.

## Acknowledgements

This project was developed for academic and demonstration purposes, specifically targeting the detection of warranty and refund fraud in modern delivery and service ecosystems.

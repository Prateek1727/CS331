# Software Requirements Specification (SRS): VeriSupport
[cite_start]**Project:**AI-based Customer Support Automation Platform [cite: 1, 2]

## 1. Abstract
[cite_start]The rapid adoption of Generative AI has introduced "Adversarial Refund Fraud" in the E-Commerce and Food Delivery sectors[cite: 4]. [cite_start]VeriSupport is a novel Customer Support Automation Platform that integrates Multimodal Large Language Models (LLMs) with Digital Image Forensics[cite: 6]. [cite_start]It acts as an "Active Defense" mechanism, utilizing Error Level Analysis (ELA), Metadata Scrutiny, and Vision-Language reasoning to autonomously verify the physical integrity of user claims in real-time[cite: 7, 8]. [cite_start]The goal is to reduce refund processing latency for genuine customers while neutralizing AI-generated fraud[cite: 9].

## 2. Problem Statement
The system addresses three critical vulnerabilities in current support infrastructure:
1.  [cite_start]**Vulnerability to Synthetic Media:** Existing systems cannot distinguish between authentic photographs and AI-generated or digitally manipulated images[cite: 11].
2.  [cite_start]**Inefficient Triage:** Genuine disputes often face resolution times of 24-48 hours due to the need for manual human review[cite: 12].
3.  [cite_start]**Revenue Leakage:** Companies suffer financial losses due to fraudsters exploiting "no-questions-asked" refund policies using manipulated media[cite: 13].

## 3. Novelty & Innovation (Technical Differentiators)
VeriSupport introduces an "Adversarial Defense Framework" with the following key features:
* [cite_start]**Cryptographic & Forensic Gatekeeping:** Implements pixel-level forensic analysis (Error Level Analysis) before processing support tickets[cite: 16].
* [cite_start]**Live-Constraint Enforcement:** Enforces a "Live-Only" capture mode using HTML5 hardware constraints to disable file system uploads, countering PC-based editing tools[cite: 17].
* [cite_start]**Weighted Ensemble Decision Logic:** Uses a probabilistic approach combining metadata, compression artifacts, and semantic consistency into a single confidence metric[cite: 18].

## 4. Functional Requirements (FR)

### Module 1: The User Interaction & Evidence Interface
* [cite_start]**FR 1.1:** The system shall provide a conversational chat interface capable of collecting context regarding the dispute[cite: 21].
* [cite_start]**FR 1.2:** The system shall enforce Real-Time Image Capture, programmatically disabling "Upload from Gallery" to prevent submission of pre-edited images[cite: 22].

### Module 2: The Forensic Analysis Engine (Microscopic Check)
* **FR 2.1 (Metadata Scrutiny):** The system shall parse EXIF data to validate the source device. [cite_start]Images containing software signatures (e.g., "Adobe", "Stable Diffusion") must be flagged[cite: 24, 25].
* [cite_start]**FR 2.2 (Error Level Analysis - ELA):** The system shall perform a microscopic pixel-level scan to detect JPEG Compression Artifacts[cite: 26].
    * [cite_start]**Algorithm:** The system creates a control image by resaving the input at 90% quality and calculates the pixel difference: `Difference = |Pixel_A - Pixel_B|`[cite: 31, 32].
    * [cite_start]**Amplification:** This difference is multiplied by a scaling factor (x50) to make manipulated regions visible[cite: 34].

### Module 3: The AI Reasoning Agent
* [cite_start]**FR 3.1:** The system shall utilize a Vision-Language Model (Gemini 1.5 Flash) to perform Semantic Consistency Checks (e.g., verifying if text descriptions match visual data)[cite: 37].

### Module 4: Decision Algorithm (The Mathematical Core)
* [cite_start]**FR 4.1:** The system shall compute the final Trust Score ($T$) using a weighted ensemble algorithm[cite: 39]:
    $$T=w_1(S_{meta})+w_2(S_{ela})+w_3(S_{ai})$$
    * [cite_start]$S_{meta}$: Binary score (0 or 1) based on EXIF validity[cite: 43].
    * [cite_start]$S_{ela}$: Normalized score inversely proportional to error level variance[cite: 46].
    * [cite_start]$S_{ai}$: Confidence score from the Vision-Language Model[cite: 47].
* [cite_start]**FR 4.2:** If $T > 0.9$, the system shall trigger the Auto-Refund API[cite: 48].
* [cite_start]**FR 4.3:** If $T < 0.5$, the system shall route the ticket to a human agent with a "Fraud Alert" tag[cite: 49].

## 5. Non-Functional Requirements (NFR)
* [cite_start]**NFR 1 (Latency):** The complete forensic audit (Metadata + ELA + AI Analysis) must complete within 5 seconds[cite: 51].
* [cite_start]**NFR 2 (Scalability):** The backend must utilize serverless architecture (FastAPI/Lambda) to handle concurrent requests[cite: 52].
* [cite_start]**NFR 3 (Privacy):** User images must be processed in ephemeral memory and permanently deleted post-resolution unless archived for confirmed fraud[cite: 53].

## 6. Technology Stack
* [cite_start]**Frontend:** Streamlit / React.js (with HTML5 Media Capture API)[cite: 67].
* [cite_start]**Backend:** Python 3.x (FastAPI)[cite: 68].
* [cite_start]**AI Models:** Google Gemini 1.5 Flash[cite: 69].
* [cite_start]**Forensics:** OpenCV, Pillow (PIL), ExifRead[cite: 70].
* [cite_start]**Database:** Supabase (PostgreSQL), Pinecone (Vector DB)[cite: 71].

## 7. Evaluation Methodology
The system will be stress-tested using a "Red Teaming" approach with 50 adversarial images generated via:
1.  [cite_start]**Generative Fill Attack:** Adding foreign objects via Photoshop[cite: 58].
2.  [cite_start]**Inpainting Attack:** Erasing/replacing items via Stable Diffusion[cite: 59].
3.  [cite_start]**Metadata Scrubbing:** Programmatically stripping EXIF data[cite: 60].

**Scoring Metrics:**
* [cite_start]**True Positive:** Correctly flagging "FRAUD"[cite: 63].
* [cite_start]**False Negative:** Incorrectly granting "REFUND APPROVED"[cite: 64].
* [cite_start]**Robustness Score:** Calculated as the Recall Rate[cite: 65].

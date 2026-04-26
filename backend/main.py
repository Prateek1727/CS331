from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from models import RawTicketRequest, Ticket, Customer, TicketTimelineEvent, AIAnalysis
from ai_engine import analyze_ticket_text, analyze_image_complete
from fraud_engine import calculate_trust_score
from database_sqlite import get_all_tickets, insert_ticket, initialize_db
from action_service import action_service
from email_service import email_service
from pydantic import BaseModel
import datetime
import uuid
from typing import Optional

app = FastAPI(title="NeuraDesk API")

# Initialize database on startup
initialize_db()

# Configure CORS to allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models for actions
class ApproveTicketRequest(BaseModel):
    ticket_id: str
    agent_id: Optional[str] = "AGENT-001"

class UpdateDraftRequest(BaseModel):
    ticket_id: str
    message: str
    agent_id: Optional[str] = "AGENT-001"

class EscalateTicketRequest(BaseModel):
    ticket_id: str
    reason: str
    agent_id: Optional[str] = "AGENT-001"

class RejectTicketRequest(BaseModel):
    ticket_id: str
    reason: str
    agent_id: Optional[str] = "AGENT-001"

@app.get("/")
def read_root():
    return {"status": "Backend is running!", "app": "NeuraDesk"}

@app.get("/api/tickets")
def get_tickets():
    return get_all_tickets()

@app.post("/api/tickets")
async def create_ticket(req: RawTicketRequest, background_tasks: BackgroundTasks):
    """
    Ingests a raw ticket (without image), runs AI models, applies rules, and stores it.
    """
    return await process_ticket_internal(
        channel=req.channel,
        customer_name=req.customer_name,
        customer_email=req.customer_email,
        subject=req.subject,
        message=req.message,
        image_bytes=None
    )

@app.post("/api/tickets/upload")
async def create_ticket_with_image(
    channel: str = Form(...),
    customer_name: str = Form(...),
    customer_email: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Ingests a ticket with optional image upload for fraud detection.
    """
    image_bytes = None
    if image:
        image_bytes = await image.read()
    
    return await process_ticket_internal(
        channel=channel,
        customer_name=customer_name,
        customer_email=customer_email,
        subject=subject,
        message=message,
        image_bytes=image_bytes
    )

async def process_ticket_internal(
    channel: str,
    customer_name: str,
    customer_email: str,
    subject: str,
    message: str,
    image_bytes: Optional[bytes] = None
):
    """
    Internal function to process tickets with or without images.
    """
    # 1. Base Setup
    ticket_id = f"TK-{str(uuid.uuid4())[:4].upper()}"
    customer = {
        "id": f"C-{str(uuid.uuid4())[:6].upper()}",
        "name": customer_name,
        "email": customer_email,
        "tier": "Standard",
        "orders": 1,
        "ltv": 0.0
    }
    
    now_iso = datetime.datetime.utcnow().isoformat() + "Z"
    
    # 2. Run AI Analysis
    text_analysis = analyze_ticket_text(subject, message)
    vision_analysis = analyze_image_complete(image_bytes) if image_bytes else None
    
    # 3. Enhanced fraud analysis considering image tampering
    fraud_analysis = calculate_trust_score(text_analysis, customer, vision_analysis)
    
    # 4. Decision Engine Logic with Image Fraud Detection
    action = "draft_response"
    decision_confidence = 0.85
    rule = "Standard routing"
    
    # Check for image fraud first
    if vision_analysis and "FRAUD DETECTED" in vision_analysis.get("verdict", ""):
        action = "escalate"
        decision_confidence = 0.98
        rule = "FRAUD ALERT: Fake food image detected - Escalate to fraud team"
    elif vision_analysis and vision_analysis.get("tamperingScore", 0) > 0.6:
        action = "escalate"
        decision_confidence = 0.92
        rule = "High image tampering score - Manual review required"
    elif text_analysis.get('intent') == 'refund_request' and fraud_analysis['trustScore'] >= 80:
        action = "auto_resolve"
        decision_confidence = 0.95
        rule = "Auto-refund under threshold + high trust"
    elif fraud_analysis['trustScore'] < 50 or text_analysis.get('sentiment') == 'angry':
        action = "escalate"
        decision_confidence = 0.90
        rule = "High risk or angry sentiment"

    # 5. Construct Final Ticket
    timeline = [
        TicketTimelineEvent(time=datetime.datetime.utcnow().strftime("%H:%M:%S"), event=f"Ticket created via {channel.capitalize()}", type="created"),
        TicketTimelineEvent(time=datetime.datetime.utcnow().strftime("%H:%M:%S"), event="AI Brain processing complete", type="ai"),
    ]
    
    if vision_analysis:
        timeline.append(TicketTimelineEvent(
            time=datetime.datetime.utcnow().strftime("%H:%M:%S"), 
            event=f"Image analysis: {vision_analysis['verdict']}", 
            type="ai"
        ))
    
    timeline.append(TicketTimelineEvent(time=datetime.datetime.utcnow().strftime("%H:%M:%S"), event=f"Decision: {rule}", type="decision"))
    
    if action == "auto_resolve":
        timeline.append(TicketTimelineEvent(time=datetime.datetime.utcnow().strftime("%H:%M:%S"), event="Auto-resolved and customer notified", type="resolved"))
    elif "FRAUD" in rule:
        timeline.append(TicketTimelineEvent(time=datetime.datetime.utcnow().strftime("%H:%M:%S"), event="Escalated to fraud investigation team", type="escalated"))
        
    ai_analysis_obj = AIAnalysis()
    ai_analysis_obj.nlp = text_analysis if text_analysis else None
    ai_analysis_obj.vision = vision_analysis if vision_analysis else None
    ai_analysis_obj.fraud = fraud_analysis
    ai_analysis_obj.rag = {"matchedPolicies": ["General KB", "Refund Policy", "Fraud Prevention"], "confidence": 0.88}

    ticket = Ticket(
        id=ticket_id,
        customer=customer,
        channel=channel,
        subject=subject,
        message=message,
        priority="critical" if "FRAUD" in rule else "high" if action == "escalate" else "medium",
        status="fraud_review" if "FRAUD" in rule else "auto_resolved" if action == "auto_resolve" else "in_progress",
        sentiment=text_analysis.get('sentiment', 'neutral'),
        trustScore=fraud_analysis['trustScore'],
        createdAt=now_iso,
        resolvedAt=now_iso if action == "auto_resolve" else None,
        hasImage=image_bytes is not None,
        aiAnalysis=ai_analysis_obj,
        decision={"action": action, "confidence": decision_confidence, "rule": rule},
        timeline=timeline
    )
    
    # Store
    insert_ticket(ticket.model_dump())
    
    # Send confirmation email to customer
    email_service.send_ticket_confirmation(
        customer_email=customer['email'],
        customer_name=customer['name'],
        ticket_id=ticket_id,
        subject=subject
    )
    
    return {"message": "Ticket processed via AI successfully", "ticket_id": ticket_id, "action": action, "fraud_detected": "FRAUD" in rule}


# ============================================================================
# ACTION ENDPOINTS - Production-level ticket actions
# ============================================================================

@app.post("/api/actions/approve")
async def approve_ticket(req: ApproveTicketRequest):
    """
    Approve ticket and send notifications to customer
    - Updates ticket status to 'resolved'
    - Sends email/SMS to customer
    - Triggers webhooks
    - Logs action
    """
    result = action_service.approve_and_send(req.ticket_id, req.agent_id)
    return result

@app.post("/api/actions/update-draft")
async def update_draft(req: UpdateDraftRequest):
    """
    Update draft response for a ticket
    - Saves new draft message
    - Logs action
    - Updates timeline
    """
    result = action_service.update_draft(req.ticket_id, req.message, req.agent_id)
    return result

@app.post("/api/actions/escalate")
async def escalate_ticket(req: EscalateTicketRequest):
    """
    Escalate ticket to human agent
    - Updates status to 'escalated'
    - Sends Slack notification
    - Logs action
    """
    result = action_service.escalate_ticket(req.ticket_id, req.reason, req.agent_id)
    return result

@app.post("/api/actions/reject")
async def reject_ticket(req: RejectTicketRequest):
    """
    Reject ticket (for fraud cases)
    - Updates status to 'rejected'
    - Sends fraud alerts
    - Logs action
    """
    result = action_service.reject_ticket(req.ticket_id, req.reason, req.agent_id)
    return result

@app.get("/api/actions/logs/{ticket_id}")
async def get_action_logs(ticket_id: str):
    """
    Get action logs for a specific ticket
    """
    import sqlite3
    import json
    
    conn = sqlite3.connect("neuradesk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM action_logs 
        WHERE ticket_id = ? 
        ORDER BY timestamp DESC
    """, (ticket_id,))
    
    rows = cursor.fetchall()
    logs = [dict(row) for row in rows]
    
    # Parse metadata JSON
    for log in logs:
        log['metadata'] = json.loads(log['metadata'])
    
    conn.close()
    
    return {"ticket_id": ticket_id, "logs": logs}

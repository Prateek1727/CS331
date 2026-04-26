from pydantic import BaseModel
from typing import List, Optional

# Basic models
class Customer(BaseModel):
    id: str
    name: str
    email: str
    tier: str
    orders: int
    ltv: float

class AIAnalysisEntity(BaseModel):
    intent: str
    sentiment: str
    confidence: float
    language: str
    entities: List[str]

class VisionAnalysisEntity(BaseModel):
    tamperingScore: float
    elaAnomaly: bool
    metadataConsistent: bool
    verdict: str

class AIAnalysis(BaseModel):
    nlp: Optional[dict] = None
    vision: Optional[dict] = None
    rag: Optional[dict] = None
    fraud: Optional[dict] = None

class TicketTimelineEvent(BaseModel):
    time: str
    event: str
    type: str

class Ticket(BaseModel):
    id: str
    customer: Customer
    channel: str
    subject: str
    priority: str
    status: str
    sentiment: str
    trustScore: int
    createdAt: str
    resolvedAt: Optional[str] = None
    message: str
    hasImage: bool
    aiAnalysis: Optional[AIAnalysis] = None
    decision: Optional[dict] = None
    timeline: List[TicketTimelineEvent]

# Incoming raw request model
class RawTicketRequest(BaseModel):
    channel: str
    customer_name: str
    customer_email: str
    subject: str
    message: str
    hasImage: bool = False

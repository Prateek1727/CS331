"""
Simple script to view database contents
"""
from database_sqlite import get_all_tickets, get_ticket_count
import json

print("\n" + "="*80)
print("📊 NEURADESK DATABASE CONTENTS")
print("="*80)

count = get_ticket_count()
print(f"\n✅ Total Tickets in Database: {count}\n")

if count == 0:
    print("❌ No tickets found. Submit a ticket from the Customer Portal first!")
    print("\nGo to: http://localhost:5174/customer-portal")
else:
    tickets = get_all_tickets()
    
    for i, ticket in enumerate(tickets, 1):
        print(f"\n{'='*80}")
        print(f"TICKET #{i}")
        print(f"{'='*80}")
        print(f"ID: {ticket['id']}")
        print(f"Customer: {ticket['customer']['name']} ({ticket['customer']['email']})")
        print(f"Channel: {ticket['channel']}")
        print(f"Subject: {ticket['subject']}")
        print(f"Message: {ticket['message'][:100]}...")
        print(f"Status: {ticket['status']}")
        print(f"Priority: {ticket['priority']}")
        print(f"Sentiment: {ticket['sentiment']}")
        print(f"Trust Score: {ticket['trustScore']}/100")
        print(f"Has Image: {ticket['hasImage']}")
        print(f"Created: {ticket['createdAt']}")
        
        # Show AI Analysis
        if ticket.get('aiAnalysis'):
            print(f"\n🤖 AI ANALYSIS:")
            
            # NLP
            if ticket['aiAnalysis'].get('nlp'):
                nlp = ticket['aiAnalysis']['nlp']
                print(f"  Intent: {nlp.get('intent', 'N/A')}")
                print(f"  Sentiment: {nlp.get('sentiment', 'N/A')}")
                print(f"  Confidence: {nlp.get('confidence', 0):.2f}")
            
            # Vision (Fraud Detection)
            if ticket['aiAnalysis'].get('vision'):
                vision = ticket['aiAnalysis']['vision']
                print(f"\n  🔍 FRAUD DETECTION:")
                print(f"  Tampering Score: {vision.get('tamperingScore', 0):.3f}")
                print(f"  ELA Anomaly: {vision.get('elaAnomaly', False)}")
                print(f"  Metadata Consistent: {vision.get('metadataConsistent', True)}")
                print(f"  Verdict: {vision.get('verdict', 'N/A')}")
                
                if vision.get('aiVision'):
                    ai_vision = vision['aiVision']
                    print(f"  AI Fraud Risk: {ai_vision.get('fraudRisk', 'N/A').upper()}")
                    if ai_vision.get('description'):
                        print(f"  AI Description: {ai_vision['description'][:150]}...")
            
            # Fraud Score
            if ticket['aiAnalysis'].get('fraud'):
                fraud = ticket['aiAnalysis']['fraud']
                print(f"\n  ⚠️  FRAUD ANALYSIS:")
                print(f"  Trust Score: {fraud.get('trustScore', 0)}/100")
                print(f"  Verdict: {fraud.get('verdict', 'N/A')}")
                print(f"  Anomalies: {fraud.get('anomalies', 0)}")
                if fraud.get('riskFactors'):
                    print(f"  Risk Factors:")
                    for factor in fraud['riskFactors']:
                        print(f"    - {factor}")
        
        # Show Decision
        if ticket.get('decision'):
            decision = ticket['decision']
            print(f"\n⚖️  DECISION:")
            print(f"  Action: {decision.get('action', 'N/A')}")
            print(f"  Confidence: {decision.get('confidence', 0):.2f}")
            print(f"  Rule: {decision.get('rule', 'N/A')}")

print(f"\n{'='*80}")
print(f"📍 Database Location: backend/neuradesk.db")
print(f"{'='*80}\n")

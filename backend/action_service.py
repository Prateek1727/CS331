"""
Production-level Action Service
Handles ticket actions: approve, reject, escalate, etc.
"""
from datetime import datetime
from typing import Dict, Optional
from database_sqlite import get_ticket_by_id, insert_ticket
from email_service import email_service
import sqlite3
import json

class ActionService:
    """
    Production action service for ticket operations
    """
    
    def __init__(self):
        self.db_path = "neuradesk.db"
    
    def approve_and_send(self, ticket_id: str, agent_id: str = "AI-AGENT") -> Dict:
        """
        Approve ticket and send notifications to customer
        
        Steps:
        1. Update ticket status to 'resolved'
        2. Add resolution timestamp
        3. Log action in timeline
        4. Send email/SMS to customer
        5. Trigger webhooks
        6. Return confirmation
        """
        try:
            # Get ticket from database
            ticket = get_ticket_by_id(ticket_id)
            if not ticket:
                return {"success": False, "error": "Ticket not found"}
            
            # Update ticket status
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            resolved_at = datetime.utcnow().isoformat() + "Z"
            
            cursor.execute("""
                UPDATE tickets 
                SET status = 'resolved', 
                    resolved_at = ?
                WHERE id = ?
            """, (resolved_at, ticket_id))
            
            conn.commit()
            conn.close()
            
            # Update ticket object
            ticket['status'] = 'resolved'
            ticket['resolvedAt'] = resolved_at
            
            # Add timeline event
            self._add_timeline_event(
                ticket_id,
                f"Ticket approved and resolved by {agent_id}",
                "resolved"
            )
            
            # Send notifications
            email_sent = email_service.send_ticket_resolved(
                customer_email=ticket['customer']['email'],
                customer_name=ticket['customer']['name'],
                ticket_id=ticket_id,
                subject=ticket['subject']
            )
            
            notification_result = {
                "email_sent": email_sent,
                "sms_sent": False,
                "webhook_sent": False,
                "slack_sent": False
            }
            
            # Log action
            self._log_action(ticket_id, "approve_and_send", agent_id, {
                "notifications": notification_result,
                "resolved_at": resolved_at
            })
            
            print(f"✅ Ticket {ticket_id} approved and customer notified")
            
            return {
                "success": True,
                "ticket_id": ticket_id,
                "status": "resolved",
                "resolved_at": resolved_at,
                "notifications": notification_result,
                "message": "Ticket approved and customer notified successfully"
            }
            
        except Exception as e:
            print(f"❌ Error approving ticket {ticket_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def update_draft(self, ticket_id: str, new_message: str, agent_id: str = "AI-AGENT") -> Dict:
        """
        Update draft response for a ticket
        """
        try:
            ticket = get_ticket_by_id(ticket_id)
            if not ticket:
                return {"success": False, "error": "Ticket not found"}
            
            # Update draft in database (store in decision field)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current decision
            cursor.execute("SELECT decision FROM tickets WHERE id = ?", (ticket_id,))
            decision_json = cursor.fetchone()[0]
            decision = json.loads(decision_json)
            
            # Update draft message
            decision['draft_message'] = new_message
            decision['draft_updated_at'] = datetime.utcnow().isoformat()
            decision['draft_updated_by'] = agent_id
            
            cursor.execute("""
                UPDATE tickets 
                SET decision = ?
                WHERE id = ?
            """, (json.dumps(decision), ticket_id))
            
            conn.commit()
            conn.close()
            
            # Add timeline event
            self._add_timeline_event(
                ticket_id,
                f"Draft response updated by {agent_id}",
                "draft_updated"
            )
            
            # Log action
            self._log_action(ticket_id, "update_draft", agent_id, {
                "message_length": len(new_message)
            })
            
            print(f"✅ Draft updated for ticket {ticket_id}")
            
            return {
                "success": True,
                "ticket_id": ticket_id,
                "message": "Draft updated successfully"
            }
            
        except Exception as e:
            print(f"❌ Error updating draft for {ticket_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def escalate_ticket(self, ticket_id: str, reason: str, agent_id: str = "AI-AGENT") -> Dict:
        """
        Escalate ticket to human agent
        """
        try:
            ticket = get_ticket_by_id(ticket_id)
            if not ticket:
                return {"success": False, "error": "Ticket not found"}
            
            # Update status
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE tickets 
                SET status = 'escalated',
                    priority = 'high'
                WHERE id = ?
            """, (ticket_id,))
            
            conn.commit()
            conn.close()
            
            # Add timeline event
            self._add_timeline_event(
                ticket_id,
                f"Escalated to human agent: {reason}",
                "escalated"
            )
            
            # Send Slack notification to team (optional)
            # notification_service.send_slack_notification(
            #     f"⚠️ Ticket {ticket_id} escalated: {reason}",
            #     channel="#support-escalations"
            # )
            
            # Log action
            self._log_action(ticket_id, "escalate", agent_id, {
                "reason": reason
            })
            
            print(f"✅ Ticket {ticket_id} escalated")
            
            return {
                "success": True,
                "ticket_id": ticket_id,
                "status": "escalated",
                "message": "Ticket escalated successfully"
            }
            
        except Exception as e:
            print(f"❌ Error escalating ticket {ticket_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def reject_ticket(self, ticket_id: str, reason: str, agent_id: str = "AI-AGENT") -> Dict:
        """
        Reject ticket (for fraud cases)
        """
        try:
            ticket = get_ticket_by_id(ticket_id)
            if not ticket:
                return {"success": False, "error": "Ticket not found"}
            
            # Update status
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            resolved_at = datetime.utcnow().isoformat() + "Z"
            
            cursor.execute("""
                UPDATE tickets 
                SET status = 'rejected',
                    resolved_at = ?
                WHERE id = ?
            """, (resolved_at, ticket_id))
            
            conn.commit()
            conn.close()
            
            # Add timeline event
            self._add_timeline_event(
                ticket_id,
                f"Ticket rejected: {reason}",
                "rejected"
            )
            
            # Send fraud alert (optional)
            # if "fraud" in reason.lower():
            #     notification_service.notify_fraud_detected(ticket)
            
            # Log action
            self._log_action(ticket_id, "reject", agent_id, {
                "reason": reason,
                "resolved_at": resolved_at
            })
            
            print(f"✅ Ticket {ticket_id} rejected")
            
            return {
                "success": True,
                "ticket_id": ticket_id,
                "status": "rejected",
                "message": "Ticket rejected successfully"
            }
            
        except Exception as e:
            print(f"❌ Error rejecting ticket {ticket_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _add_timeline_event(self, ticket_id: str, event: str, event_type: str):
        """
        Add event to ticket timeline
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current timeline
            cursor.execute("SELECT timeline FROM tickets WHERE id = ?", (ticket_id,))
            timeline_json = cursor.fetchone()[0]
            timeline = json.loads(timeline_json)
            
            # Add new event
            timeline.append({
                "time": datetime.utcnow().strftime("%H:%M:%S"),
                "event": event,
                "type": event_type
            })
            
            # Update database
            cursor.execute("""
                UPDATE tickets 
                SET timeline = ?
                WHERE id = ?
            """, (json.dumps(timeline), ticket_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Error adding timeline event: {e}")
    
    def _log_action(self, ticket_id: str, action: str, agent_id: str, metadata: Dict):
        """
        Log action to action_logs table
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create action_logs table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    action TEXT,
                    agent_id TEXT,
                    metadata TEXT,
                    timestamp TEXT
                )
            """)
            
            # Insert log
            cursor.execute("""
                INSERT INTO action_logs (ticket_id, action, agent_id, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                ticket_id,
                action,
                agent_id,
                json.dumps(metadata),
                datetime.utcnow().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Error logging action: {e}")

# Global instance
action_service = ActionService()

"""
Production-level Notification Service
Handles email, SMS, and webhook notifications
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import requests
from typing import Dict, List
import json

class NotificationService:
    """
    Production notification service supporting:
    - Email (SMTP)
    - SMS (Twilio)
    - Webhooks
    - Slack notifications
    """
    
    def __init__(self):
        # Email configuration (SMTP)
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "support@neuradesk.com")
        
        # SMS configuration (Twilio)
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_phone = os.getenv("TWILIO_PHONE", "")
        
        # Webhook configuration
        self.webhook_urls = os.getenv("WEBHOOK_URLS", "").split(",")
        
        # Slack configuration
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    
    def send_email(self, to_email: str, subject: str, body: str, html: bool = True) -> bool:
        """
        Send email notification via SMTP
        """
        try:
            if not self.smtp_user or not self.smtp_password:
                print(f"⚠️  Email not configured. Would send to {to_email}: {subject}")
                return False
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            print(f"✅ Email sent to {to_email}: {subject}")
            return True
            
        except Exception as e:
            print(f"❌ Email error: {e}")
            return False
    
    def send_sms(self, to_phone: str, message: str) -> bool:
        """
        Send SMS notification via Twilio
        """
        try:
            if not self.twilio_account_sid or not self.twilio_auth_token:
                print(f"⚠️  SMS not configured. Would send to {to_phone}: {message}")
                return False
            
            from twilio.rest import Client
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            message = client.messages.create(
                body=message,
                from_=self.twilio_phone,
                to=to_phone
            )
            
            print(f"✅ SMS sent to {to_phone}: {message.sid}")
            return True
            
        except Exception as e:
            print(f"❌ SMS error: {e}")
            return False
    
    def send_webhook(self, event_type: str, data: Dict) -> List[bool]:
        """
        Send webhook notifications to configured endpoints
        """
        results = []
        
        for webhook_url in self.webhook_urls:
            if not webhook_url or webhook_url.strip() == "":
                continue
            
            try:
                payload = {
                    "event": event_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data
                }
                
                response = requests.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"✅ Webhook sent to {webhook_url}: {event_type}")
                    results.append(True)
                else:
                    print(f"⚠️  Webhook failed ({response.status_code}): {webhook_url}")
                    results.append(False)
                    
            except Exception as e:
                print(f"❌ Webhook error for {webhook_url}: {e}")
                results.append(False)
        
        return results
    
    def send_slack_notification(self, message: str, channel: str = None) -> bool:
        """
        Send notification to Slack
        """
        try:
            if not self.slack_webhook:
                print(f"⚠️  Slack not configured. Would send: {message}")
                return False
            
            payload = {
                "text": message,
                "username": "NeuraDesk Bot",
                "icon_emoji": ":robot_face:"
            }
            
            if channel:
                payload["channel"] = channel
            
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ Slack notification sent")
                return True
            else:
                print(f"⚠️  Slack failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Slack error: {e}")
            return False
    
    def notify_ticket_approved(self, ticket: Dict) -> Dict:
        """
        Send all notifications when ticket is approved
        """
        customer = ticket['customer']
        ticket_id = ticket['id']
        
        # Email notification
        email_subject = f"Your Support Request {ticket_id} Has Been Resolved"
        email_body = self._generate_approval_email(ticket)
        email_sent = self.send_email(customer['email'], email_subject, email_body)
        
        # SMS notification (if phone available)
        sms_sent = False
        if customer.get('phone'):
            sms_message = f"NeuraDesk: Your ticket {ticket_id} has been resolved. Check your email for details."
            sms_sent = self.send_sms(customer['phone'], sms_message)
        
        # Webhook notification
        webhook_sent = self.send_webhook("ticket.approved", {
            "ticket_id": ticket_id,
            "customer_email": customer['email'],
            "status": "resolved"
        })
        
        # Slack notification for internal team
        slack_message = f"✅ Ticket {ticket_id} approved and customer notified ({customer['name']})"
        slack_sent = self.send_slack_notification(slack_message)
        
        return {
            "email_sent": email_sent,
            "sms_sent": sms_sent,
            "webhook_sent": any(webhook_sent) if webhook_sent else False,
            "slack_sent": slack_sent,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def notify_fraud_detected(self, ticket: Dict) -> Dict:
        """
        Send fraud alert notifications
        """
        ticket_id = ticket['id']
        customer = ticket['customer']
        
        # Internal Slack alert
        slack_message = f"🚨 FRAUD ALERT: Ticket {ticket_id} - Fake food image detected from {customer['name']} ({customer['email']})"
        slack_sent = self.send_slack_notification(slack_message, channel="#fraud-alerts")
        
        # Webhook for fraud system
        webhook_sent = self.send_webhook("fraud.detected", {
            "ticket_id": ticket_id,
            "customer_email": customer['email'],
            "trust_score": ticket['trustScore'],
            "tampering_score": ticket.get('aiAnalysis', {}).get('vision', {}).get('tamperingScore', 0)
        })
        
        # Email to fraud team
        fraud_email = os.getenv("FRAUD_TEAM_EMAIL", "fraud@neuradesk.com")
        email_subject = f"FRAUD ALERT: {ticket_id}"
        email_body = self._generate_fraud_alert_email(ticket)
        email_sent = self.send_email(fraud_email, email_subject, email_body)
        
        return {
            "email_sent": email_sent,
            "webhook_sent": any(webhook_sent) if webhook_sent else False,
            "slack_sent": slack_sent,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_approval_email(self, ticket: Dict) -> str:
        """
        Generate HTML email for ticket approval
        """
        customer = ticket['customer']
        ticket_id = ticket['id']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #10b981; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 8px 8px; }}
                .ticket-id {{ font-size: 24px; font-weight: bold; color: #10b981; }}
                .footer {{ text-align: center; margin-top: 20px; color: #6b7280; font-size: 12px; }}
                .button {{ background: #10b981; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>✅ Your Request Has Been Resolved</h1>
                </div>
                <div class="content">
                    <p>Dear {customer['name']},</p>
                    
                    <p>Thank you for contacting us. We're pleased to inform you that your support request has been resolved.</p>
                    
                    <p><strong>Ticket ID:</strong> <span class="ticket-id">{ticket_id}</span></p>
                    <p><strong>Subject:</strong> {ticket['subject']}</p>
                    
                    <p>Our AI-powered support system has processed your request and taken the appropriate action. If this was a refund request, the amount will be credited to your account within 3-5 business days.</p>
                    
                    <p>If you have any additional questions or concerns, please don't hesitate to reach out.</p>
                    
                    <a href="https://neuradesk.com/tickets/{ticket_id}" class="button">View Ticket Details</a>
                    
                    <p style="margin-top: 30px;">Best regards,<br>The NeuraDesk Support Team</p>
                </div>
                <div class="footer">
                    <p>This is an automated message from NeuraDesk AI Support Platform</p>
                    <p>© 2026 NeuraDesk. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_fraud_alert_email(self, ticket: Dict) -> str:
        """
        Generate HTML email for fraud alert
        """
        customer = ticket['customer']
        ticket_id = ticket['id']
        vision = ticket.get('aiAnalysis', {}).get('vision', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #ef4444; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #fef2f2; padding: 30px; border: 2px solid #ef4444; border-radius: 0 0 8px 8px; }}
                .alert {{ background: #fee2e2; border-left: 4px solid #ef4444; padding: 15px; margin: 20px 0; }}
                .metric {{ background: white; padding: 10px; margin: 10px 0; border-radius: 6px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 FRAUD ALERT</h1>
                </div>
                <div class="content">
                    <div class="alert">
                        <strong>Fake Food Image Detected</strong>
                    </div>
                    
                    <p><strong>Ticket ID:</strong> {ticket_id}</p>
                    <p><strong>Customer:</strong> {customer['name']} ({customer['email']})</p>
                    <p><strong>Subject:</strong> {ticket['subject']}</p>
                    
                    <h3>Fraud Analysis:</h3>
                    <div class="metric">
                        <strong>Trust Score:</strong> {ticket['trustScore']}/100
                    </div>
                    <div class="metric">
                        <strong>Tampering Score:</strong> {vision.get('tamperingScore', 0):.3f}
                    </div>
                    <div class="metric">
                        <strong>Verdict:</strong> {vision.get('verdict', 'N/A')}
                    </div>
                    <div class="metric">
                        <strong>ELA Anomaly:</strong> {'Yes' if vision.get('elaAnomaly') else 'No'}
                    </div>
                    
                    <h3>Recommended Actions:</h3>
                    <ul>
                        <li>Review ticket manually</li>
                        <li>Flag customer account</li>
                        <li>Deny refund request</li>
                        <li>Consider account suspension</li>
                    </ul>
                    
                    <p style="margin-top: 30px;"><strong>Fraud Investigation Team</strong></p>
                </div>
            </div>
        </body>
        </html>
        """
        return html

# Global instance
notification_service = NotificationService()

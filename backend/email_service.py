"""
Simple Email Service for Customer Notifications
Sends email to customer when ticket is created
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class EmailService:
    """
    Simple email service for customer notifications
    """
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "support@neuradesk.com")
        
        # Check if email is configured
        self.is_configured = bool(self.smtp_user and self.smtp_password)
        
        if self.is_configured:
            print(f"✅ Email service configured: {self.smtp_user}")
        else:
            print("⚠️  Email service not configured. Add SMTP_USER and SMTP_PASSWORD to .env")
    
    def send_ticket_confirmation(self, customer_email: str, customer_name: str, ticket_id: str, subject: str) -> bool:
        """
        Send ticket confirmation email to customer
        """
        if not self.is_configured:
            print(f"⚠️  Email not configured. Would send to: {customer_email}")
            return False
        
        try:
            email_subject = f"Ticket Received: {ticket_id} - {subject}"
            email_body = self._generate_confirmation_email(customer_name, ticket_id, subject)
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = email_subject
            msg['From'] = self.from_email
            msg['To'] = customer_email
            
            msg.attach(MIMEText(email_body, 'html'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            print(f"✅ Confirmation email sent to {customer_email} for ticket {ticket_id}")
            return True
            
        except Exception as e:
            print(f"❌ Email error: {e}")
            return False
    
    def send_ticket_resolved(self, customer_email: str, customer_name: str, ticket_id: str, subject: str) -> bool:
        """
        Send ticket resolution email to customer
        """
        if not self.is_configured:
            print(f"⚠️  Email not configured. Would send to: {customer_email}")
            return False
        
        try:
            email_subject = f"Ticket Resolved: {ticket_id} - {subject}"
            email_body = self._generate_resolution_email(customer_name, ticket_id, subject)
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = email_subject
            msg['From'] = self.from_email
            msg['To'] = customer_email
            
            msg.attach(MIMEText(email_body, 'html'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            print(f"✅ Resolution email sent to {customer_email} for ticket {ticket_id}")
            return True
            
        except Exception as e:
            print(f"❌ Email error: {e}")
            return False
    
    def _generate_confirmation_email(self, customer_name: str, ticket_id: str, subject: str) -> str:
        """
        Generate HTML email for ticket confirmation
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #3b82f6; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9fafb; padding: 30px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }}
                .ticket-id {{ font-size: 24px; font-weight: bold; color: #3b82f6; }}
                .info-box {{ background: white; padding: 15px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #3b82f6; }}
                .footer {{ text-align: center; margin-top: 20px; color: #6b7280; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>✅ Ticket Received</h1>
                </div>
                <div class="content">
                    <p>Dear {customer_name},</p>
                    
                    <p>Thank you for contacting us. We have received your support request and our AI-powered system is analyzing it.</p>
                    
                    <div class="info-box">
                        <p><strong>Ticket ID:</strong> <span class="ticket-id">{ticket_id}</span></p>
                        <p><strong>Subject:</strong> {subject}</p>
                        <p><strong>Status:</strong> Processing</p>
                    </div>
                    
                    <p>Our AI system is currently:</p>
                    <ul>
                        <li>Analyzing your request with NLP</li>
                        <li>Checking for fraud patterns (if image attached)</li>
                        <li>Matching with our knowledge base</li>
                        <li>Determining the best resolution</li>
                    </ul>
                    
                    <p>You will receive another email once your ticket has been processed. This typically takes 10-15 seconds.</p>
                    
                    <p style="margin-top: 30px;">Best regards,<br><strong>NeuraDesk AI Support Team</strong></p>
                </div>
                <div class="footer">
                    <p>This is an automated message from NeuraDesk</p>
                    <p>Ticket ID: {ticket_id} | © 2026 NeuraDesk</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_resolution_email(self, customer_name: str, ticket_id: str, subject: str) -> str:
        """
        Generate HTML email for ticket resolution
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #10b981; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9fafb; padding: 30px; border: 1px solid #e5e7eb; border-radius: 0 0 8px 8px; }}
                .ticket-id {{ font-size: 24px; font-weight: bold; color: #10b981; }}
                .success-box {{ background: #d1fae5; padding: 15px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #10b981; }}
                .footer {{ text-align: center; margin-top: 20px; color: #6b7280; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>✅ Ticket Resolved</h1>
                </div>
                <div class="content">
                    <p>Dear {customer_name},</p>
                    
                    <p>Great news! Your support request has been resolved by our AI system.</p>
                    
                    <div class="success-box">
                        <p><strong>Ticket ID:</strong> <span class="ticket-id">{ticket_id}</span></p>
                        <p><strong>Subject:</strong> {subject}</p>
                        <p><strong>Status:</strong> ✅ Resolved</p>
                    </div>
                    
                    <p>Our AI-powered support system has processed your request and taken the appropriate action.</p>
                    
                    <p><strong>What happens next:</strong></p>
                    <ul>
                        <li>If this was a refund request, the amount will be credited within 3-5 business days</li>
                        <li>If this was a complaint, appropriate action has been taken</li>
                        <li>If you need further assistance, simply reply to this email</li>
                    </ul>
                    
                    <p>We appreciate your patience and thank you for choosing our service.</p>
                    
                    <p style="margin-top: 30px;">Best regards,<br><strong>NeuraDesk AI Support Team</strong></p>
                </div>
                <div class="footer">
                    <p>This is an automated message from NeuraDesk</p>
                    <p>Ticket ID: {ticket_id} | © 2026 NeuraDesk</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html

# Global instance
email_service = EmailService()

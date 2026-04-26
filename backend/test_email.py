"""
Test email notification system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*60)
print("📧 EMAIL NOTIFICATION TEST")
print("="*60)
print()

# Check configuration
smtp_user = os.getenv("SMTP_USER", "")
smtp_password = os.getenv("SMTP_PASSWORD", "")

if not smtp_user or not smtp_password:
    print("❌ Email not configured!")
    print()
    print("To enable email notifications:")
    print("1. Edit backend/.env")
    print("2. Add your Gmail credentials:")
    print()
    print("   SMTP_USER=your.email@gmail.com")
    print("   SMTP_PASSWORD=your_app_password")
    print("   FROM_EMAIL=your.email@gmail.com")
    print()
    print("See SETUP_EMAIL_NOTIFICATIONS.md for detailed instructions")
    exit(1)

print(f"✅ SMTP configured: {smtp_user}")
print()

# Ask for test email
test_email = input("Enter email address to send test to (or press Enter to use configured email): ").strip()
if not test_email:
    test_email = smtp_user

print()
print(f"Sending test email to: {test_email}")
print()

# Import notification service
from notification_service import notification_service

# Send test email
html_body = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #10b981; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
        .content { background: #f9fafb; padding: 30px; border-radius: 0 0 8px 8px; }
        .success { color: #10b981; font-size: 48px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✅ Email Test Successful!</h1>
        </div>
        <div class="content">
            <p class="success">🎉</p>
            <h2>Congratulations!</h2>
            <p>Your NeuraDesk email notification system is working correctly.</p>
            <p>When customers receive ticket approvals, they'll get professional emails like this one.</p>
            <hr>
            <p><strong>Configuration:</strong></p>
            <ul>
                <li>SMTP Server: Connected ✅</li>
                <li>Email Delivery: Working ✅</li>
                <li>HTML Formatting: Enabled ✅</li>
            </ul>
            <p style="margin-top: 30px;">Best regards,<br>The NeuraDesk Team</p>
        </div>
    </div>
</body>
</html>
"""

result = notification_service.send_email(
    to_email=test_email,
    subject="✅ NeuraDesk Email Test - Success!",
    body=html_body,
    html=True
)

print()
if result:
    print("="*60)
    print("✅ SUCCESS! Email sent successfully!")
    print("="*60)
    print()
    print(f"Check the inbox for: {test_email}")
    print("(Also check spam folder if you don't see it)")
    print()
    print("Your email notifications are now fully configured! 🎉")
    print()
    print("Next steps:")
    print("1. Go to: http://localhost:5174/customer-portal")
    print("2. Submit a ticket with your email")
    print("3. Go to: http://localhost:5174/actions")
    print("4. Click 'Approve & Send'")
    print("5. Customer will receive email notification!")
else:
    print("="*60)
    print("❌ FAILED! Email could not be sent")
    print("="*60)
    print()
    print("Common issues:")
    print("1. Wrong App Password - Generate a new one")
    print("2. 2FA not enabled - Enable it first")
    print("3. Wrong email address - Check SMTP_USER")
    print()
    print("See SETUP_EMAIL_NOTIFICATIONS.md for help")

print()

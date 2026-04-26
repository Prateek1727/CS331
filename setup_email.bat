@echo off
cls
echo ========================================
echo 📧 Email Notification Setup
echo ========================================
echo.
echo This will help you configure email notifications.
echo.
echo You need:
echo 1. A Gmail account
echo 2. 2-Factor Authentication enabled
echo 3. An App Password generated
echo.
echo ========================================
echo.

set /p email="Enter your Gmail address: "
set /p password="Enter your App Password (16 characters): "

echo.
echo Updating backend\.env file...
echo.

cd backend

(
echo GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
echo.
echo # Email Configuration ^(SMTP^) - Gmail
echo SMTP_HOST=smtp.gmail.com
echo SMTP_PORT=587
echo SMTP_USER=%email%
echo SMTP_PASSWORD=%password%
echo FROM_EMAIL=%email%
echo.
echo # SMS Configuration ^(Twilio^) - Optional
echo TWILIO_ACCOUNT_SID=
echo TWILIO_AUTH_TOKEN=
echo TWILIO_PHONE=
echo.
echo # Webhook Configuration - Optional
echo WEBHOOK_URLS=
echo.
echo # Slack Configuration - Optional
echo SLACK_WEBHOOK_URL=
echo.
echo # Fraud Team Email
echo FRAUD_TEAM_EMAIL=fraud@neuradesk.com
) > .env

cd ..

echo ✅ Configuration saved!
echo.
echo ========================================
echo Testing email...
echo ========================================
echo.

cd backend
python test_email.py
cd ..

echo.
pause

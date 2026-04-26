# 📧 Email Notification - Complete Setup

## 🎯 What You Want

When a customer enters their email in the Customer Portal and you click "Approve & Send", they should receive an email notification.

## ✅ How It Works Now

1. Customer fills form at: http://localhost:5174/customer-portal
2. Customer enters their email (e.g., `customer@example.com`)
3. Ticket is created with that email
4. You go to: http://localhost:5174/actions
5. You click "Approve & Send"
6. **Email is sent to the customer's email address** ✅

---

## 🚀 Quick Setup (Choose One Method)

### **Method 1: Automated Setup (Easiest)**

1. Double-click: `setup_email.bat`
2. Enter your Gmail address
3. Enter your App Password
4. Done! It will test automatically

### **Method 2: Manual Setup**

1. Get Gmail App Password:
   - Go to: https://myaccount.google.com/apppasswords
   - Generate password for "Mail"
   
2. Edit `backend/.env`:
   ```env
   SMTP_USER=your.email@gmail.com
   SMTP_PASSWORD=abcdefghijklmnop
   FROM_EMAIL=your.email@gmail.com
   ```

3. Test it:
   ```bash
   cd backend
   python test_email.py
   ```

---

## 📧 Complete Flow Example

### **Step 1: Customer Submits Ticket**

Customer goes to: http://localhost:5174/customer-portal

Fills form:
- Name: John Doe
- **Email: john.doe@example.com** ← This is where email will be sent
- Subject: Found insect in food
- Message: I want a refund
- Uploads image

### **Step 2: Ticket Created**

System creates ticket TK-1234 with:
- Customer email: `john.doe@example.com`
- Status: `in_progress`
- AI analysis completed

### **Step 3: Agent Approves**

You go to: http://localhost:5174/actions

Click "Approve & Send" on ticket TK-1234

### **Step 4: Email Sent**

System automatically:
1. ✅ Updates ticket status to "resolved"
2. ✅ Sends email to `john.doe@example.com`
3. ✅ Logs action
4. ✅ Updates timeline

### **Step 5: Customer Receives Email**

Customer receives:

```
From: support@neuradesk.com
To: john.doe@example.com
Subject: Your Support Request TK-1234 Has Been Resolved

Dear John Doe,

Thank you for contacting us. We're pleased to inform you 
that your support request has been resolved.

Ticket ID: TK-1234
Subject: Found insect in food

Our AI-powered support system has processed your request 
and taken the appropriate action. If this was a refund 
request, the amount will be credited to your account 
within 3-5 business days.

Best regards,
The NeuraDesk Support Team
```

---

## 🧪 Test With Your Own Email

### **Test 1: Send to Yourself**

1. Go to: http://localhost:5174/customer-portal
2. Enter YOUR email address
3. Fill rest of form
4. Submit ticket
5. Go to: http://localhost:5174/actions
6. Click "Approve & Send"
7. Check YOUR inbox!

### **Test 2: Send to Different Email**

1. Use any email address in the form
2. That email will receive the notification
3. Works with Gmail, Yahoo, Outlook, etc.

---

## 📊 Email Configuration Status

### **Before Configuration:**
```
⚠️  Email not configured. Would send to customer@example.com: Your Support Request...
```

### **After Configuration:**
```
✅ Email sent to customer@example.com: Your Support Request TK-1234 Has Been Resolved
```

---

## 🔍 Verify Email is Working

### **Check Backend Logs:**

Look for:
```
✅ Email sent to customer@example.com: Your Support Request TK-1234 Has Been Resolved
```

If you see:
```
⚠️  Email not configured. Would send to...
```
Then SMTP is not configured yet.

### **Check Frontend Response:**

After clicking "Approve & Send", you'll see:
```
✅ SUCCESS!

Ticket TK-1234 approved and customer notified!

Status: resolved
Notifications sent:
- Email: ✅  ← Should be green checkmark
- SMS: ❌
- Webhook: ❌
- Slack: ❌
```

---

## 🎯 Current Configuration

Your `.env` file should look like:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# Email Configuration (SMTP) - Gmail
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your.email@gmail.com          ← Add this
SMTP_PASSWORD=your_app_password         ← Add this
FROM_EMAIL=your.email@gmail.com         ← Add this

# Other services (optional)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE=
WEBHOOK_URLS=
SLACK_WEBHOOK_URL=
FRAUD_TEAM_EMAIL=fraud@neuradesk.com
```

---

## ⚠️ Important Notes

### **Email Goes To:**
- The email address the **customer entered** in the form
- NOT your email (unless you test with your own email)
- NOT a fixed email
- **Dynamic based on customer input** ✅

### **From Address:**
- Shows as: `FROM_EMAIL` from your `.env`
- Example: `support@neuradesk.com`
- Or your Gmail: `your.email@gmail.com`

### **Email Content:**
- Professional HTML template
- Includes ticket ID
- Includes customer name
- Includes subject
- Branded with NeuraDesk

---

## 🚀 Quick Start Commands

```bash
# Setup email (automated)
setup_email.bat

# Test email manually
cd backend
python test_email.py

# View backend logs
# (Check terminal where backend is running)

# Check database
view_database.bat
```

---

## 📝 Troubleshooting

### **"Email not configured" in logs**
→ Add SMTP_USER and SMTP_PASSWORD to `.env`
→ Restart backend (it auto-reloads)

### **"Authentication failed"**
→ Use App Password, not regular password
→ Enable 2FA on Gmail first

### **"Email not arriving"**
→ Check spam folder
→ Verify customer email is correct
→ Check backend logs for errors

### **"Connection refused"**
→ Check internet connection
→ Verify SMTP_HOST and SMTP_PORT

---

## ✅ Checklist

- [ ] Gmail account ready
- [ ] 2FA enabled on Gmail
- [ ] App Password generated
- [ ] Updated `backend/.env`
- [ ] Ran `test_email.py` successfully
- [ ] Backend restarted (auto-reloads)
- [ ] Tested with own email
- [ ] Verified email received
- [ ] Checked spam folder

---

## 🎉 Success Criteria

You'll know it's working when:

1. ✅ `test_email.py` sends successfully
2. ✅ You receive test email in inbox
3. ✅ Backend logs show: `✅ Email sent to...`
4. ✅ Frontend shows: `Email: ✅`
5. ✅ Customer receives professional email

---

## 📞 Need Help?

1. Run `setup_email.bat` for guided setup
2. Run `test_email.py` to test configuration
3. Check `SETUP_EMAIL_NOTIFICATIONS.md` for detailed guide
4. Check backend terminal for error messages
5. Verify `.env` file has correct credentials

---

**Once configured, every customer will receive email notifications automatically!** 📧✨

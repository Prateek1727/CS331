# 📊 Database Information

## Database Type: SQLite

Your platform now uses **SQLite** for persistent data storage.

### ✅ Benefits:
- **Persistent**: Data survives server restarts
- **No Setup**: No database server needed
- **File-based**: Stored in `backend/neuradesk.db`
- **Fast**: Excellent for small to medium applications
- **Portable**: Single file, easy to backup

---

## 📁 Database Location

**File**: `backend/neuradesk.db`

This file contains all your tickets and will be created automatically when you start the backend.

---

## 🗄️ Database Schema

### **Tickets Table**

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Unique ticket ID (e.g., TK-1234) |
| customer_id | TEXT | Customer unique ID |
| customer_name | TEXT | Customer name |
| customer_email | TEXT | Customer email |
| customer_tier | TEXT | Customer tier (Standard, VIP, etc.) |
| customer_orders | INTEGER | Number of previous orders |
| customer_ltv | REAL | Customer lifetime value |
| channel | TEXT | Source channel (web, email, social) |
| subject | TEXT | Ticket subject |
| message | TEXT | Ticket message/description |
| priority | TEXT | Priority level (low, medium, high, critical) |
| status | TEXT | Current status (in_progress, auto_resolved, fraud_review) |
| sentiment | TEXT | Customer sentiment (positive, neutral, negative, angry) |
| trust_score | INTEGER | Fraud trust score (0-100) |
| created_at | TEXT | Creation timestamp |
| resolved_at | TEXT | Resolution timestamp (if resolved) |
| has_image | BOOLEAN | Whether ticket has image attached |
| ai_analysis | TEXT | JSON string of AI analysis results |
| decision | TEXT | JSON string of decision engine output |
| timeline | TEXT | JSON array of timeline events |

---

## 🛠️ Database Management

### **View Database Contents**

Run the database manager:
```bash
cd backend
python db_manager.py
```

This provides a menu to:
1. View all tickets
2. View ticket count
3. View tickets by status
4. Delete all tickets
5. Initialize/reset database

### **Direct SQL Access**

You can also query the database directly:
```bash
cd backend
sqlite3 neuradesk.db
```

Example queries:
```sql
-- View all tickets
SELECT id, subject, status, trust_score FROM tickets;

-- Count tickets by status
SELECT status, COUNT(*) FROM tickets GROUP BY status;

-- Find fraud cases
SELECT id, subject, trust_score FROM tickets WHERE status = 'fraud_review';

-- View recent tickets
SELECT id, subject, created_at FROM tickets ORDER BY created_at DESC LIMIT 10;

-- Exit
.quit
```

---

## 📊 Channel Types Explained

The **channel** field indicates where the customer complaint originated:

### **1. Web** 🌐
- Customer Portal form submissions
- Website contact forms
- Self-service portals

### **2. Email** 📧
- Customer emails to support@company.com
- Email replies to tickets
- Forwarded complaints

### **3. Social** 📱
- Twitter mentions/DMs
- Facebook messages
- Instagram comments
- LinkedIn messages

### **4. Live Chat** 💬
- Website chat widget
- In-app messaging
- WhatsApp Business

### **5. Voice** 📞
- Phone calls
- Voice messages
- IVR transcriptions

### **6. Mobile** 📱
- Mobile app submissions
- Push notification responses
- In-app feedback

### **7. API** 🔌
- Third-party integrations
- Automated systems
- Partner platforms

---

## 🔄 How Channels Work

When a ticket is created, the system:

1. **Records the channel** (web, email, social, etc.)
2. **Normalizes the data** into a standard format
3. **Routes to AI analysis** regardless of source
4. **Applies channel-specific rules** if needed

Example:
```python
# Customer submits via Twitter
channel = "social"

# System processes it the same way as web/email
# But can apply special rules like:
if channel == "social":
    priority = "high"  # Public complaints get higher priority
```

---

## 💾 Backup & Restore

### **Backup Database**
```bash
# Copy the database file
cp backend/neuradesk.db backend/neuradesk_backup_2026-04-18.db
```

### **Restore Database**
```bash
# Replace with backup
cp backend/neuradesk_backup_2026-04-18.db backend/neuradesk.db
```

### **Export to JSON**
```python
import json
from database_sqlite import get_all_tickets

tickets = get_all_tickets()
with open('tickets_export.json', 'w') as f:
    json.dump(tickets, f, indent=2)
```

---

## 🚀 Upgrading to Production Database

For production, you can easily switch to PostgreSQL or MySQL:

### **PostgreSQL** (Recommended)
```python
# Install: pip install psycopg2-binary sqlalchemy
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://user:password@localhost/neuradesk"
engine = create_engine(DATABASE_URL)
```

### **MySQL**
```python
# Install: pip install pymysql sqlalchemy
DATABASE_URL = "mysql+pymysql://user:password@localhost/neuradesk"
engine = create_engine(DATABASE_URL)
```

### **MongoDB** (NoSQL)
```python
# Install: pip install pymongo
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.neuradesk
```

---

## 📈 Database Statistics

View statistics using the database manager or SQL:

```sql
-- Total tickets
SELECT COUNT(*) as total_tickets FROM tickets;

-- Tickets by status
SELECT status, COUNT(*) as count FROM tickets GROUP BY status;

-- Tickets by channel
SELECT channel, COUNT(*) as count FROM tickets GROUP BY channel;

-- Average trust score
SELECT AVG(trust_score) as avg_trust FROM tickets;

-- Fraud cases
SELECT COUNT(*) as fraud_cases FROM tickets WHERE status = 'fraud_review';

-- Tickets with images
SELECT COUNT(*) as with_images FROM tickets WHERE has_image = 1;
```

---

## 🔍 Querying Fraud Cases

Find all fraud-detected tickets:

```sql
SELECT 
    id,
    customer_name,
    subject,
    trust_score,
    status,
    created_at
FROM tickets
WHERE status = 'fraud_review'
ORDER BY created_at DESC;
```

---

## 🎯 Summary

- ✅ **Database**: SQLite (file-based)
- ✅ **Location**: `backend/neuradesk.db`
- ✅ **Persistent**: Data survives restarts
- ✅ **Manageable**: Use `db_manager.py` tool
- ✅ **Queryable**: Direct SQL access available
- ✅ **Channels**: Multi-channel support (web, email, social, etc.)
- ✅ **Upgradeable**: Easy migration to PostgreSQL/MySQL

Your tickets are now safely stored in a database! 🎉

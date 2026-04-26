"""
SQLite Database for persistent ticket storage
"""
import sqlite3
import json
from typing import List, Dict
from datetime import datetime
import os

DB_PATH = "neuradesk.db"

def initialize_db():
    """Create database and tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tickets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            customer_id TEXT,
            customer_name TEXT,
            customer_email TEXT,
            customer_tier TEXT,
            customer_orders INTEGER,
            customer_ltv REAL,
            channel TEXT,
            subject TEXT,
            message TEXT,
            priority TEXT,
            status TEXT,
            sentiment TEXT,
            trust_score INTEGER,
            created_at TEXT,
            resolved_at TEXT,
            has_image BOOLEAN,
            ai_analysis TEXT,
            decision TEXT,
            timeline TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {DB_PATH}")

def insert_ticket(ticket: dict):
    """Insert a new ticket into the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Extract customer data
    customer = ticket.get('customer', {})
    
    # Convert complex objects to JSON strings
    ai_analysis_json = json.dumps(ticket.get('aiAnalysis', {}))
    decision_json = json.dumps(ticket.get('decision', {}))
    timeline_json = json.dumps(ticket.get('timeline', []))
    
    cursor.execute("""
        INSERT INTO tickets (
            id, customer_id, customer_name, customer_email, customer_tier,
            customer_orders, customer_ltv, channel, subject, message,
            priority, status, sentiment, trust_score, created_at,
            resolved_at, has_image, ai_analysis, decision, timeline
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticket['id'],
        customer.get('id'),
        customer.get('name'),
        customer.get('email'),
        customer.get('tier'),
        customer.get('orders'),
        customer.get('ltv'),
        ticket['channel'],
        ticket['subject'],
        ticket['message'],
        ticket['priority'],
        ticket['status'],
        ticket['sentiment'],
        ticket['trustScore'],
        ticket['createdAt'],
        ticket.get('resolvedAt'),
        ticket['hasImage'],
        ai_analysis_json,
        decision_json,
        timeline_json
    ))
    
    conn.commit()
    conn.close()
    print(f"✅ Ticket {ticket['id']} saved to database")

def get_all_tickets() -> List[dict]:
    """Retrieve all tickets from the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    
    tickets = []
    for row in rows:
        # Convert row to dictionary
        ticket = dict(row)
        
        # Reconstruct customer object
        ticket['customer'] = {
            'id': ticket.pop('customer_id'),
            'name': ticket.pop('customer_name'),
            'email': ticket.pop('customer_email'),
            'tier': ticket.pop('customer_tier'),
            'orders': ticket.pop('customer_orders'),
            'ltv': ticket.pop('customer_ltv')
        }
        
        # Parse JSON fields
        ticket['aiAnalysis'] = json.loads(ticket.pop('ai_analysis'))
        ticket['decision'] = json.loads(ticket.pop('decision'))
        ticket['timeline'] = json.loads(ticket.pop('timeline'))
        
        # Rename fields to match frontend expectations
        ticket['trustScore'] = ticket.pop('trust_score')
        ticket['createdAt'] = ticket.pop('created_at')
        ticket['resolvedAt'] = ticket.pop('resolved_at')
        ticket['hasImage'] = bool(ticket.pop('has_image'))
        
        tickets.append(ticket)
    
    conn.close()
    return tickets

def get_ticket_by_id(ticket_id: str) -> dict:
    """Retrieve a specific ticket by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return None
    
    ticket = dict(row)
    
    # Reconstruct customer object
    ticket['customer'] = {
        'id': ticket.pop('customer_id'),
        'name': ticket.pop('customer_name'),
        'email': ticket.pop('customer_email'),
        'tier': ticket.pop('customer_tier'),
        'orders': ticket.pop('customer_orders'),
        'ltv': ticket.pop('customer_ltv')
    }
    
    # Parse JSON fields
    ticket['aiAnalysis'] = json.loads(ticket.pop('ai_analysis'))
    ticket['decision'] = json.loads(ticket.pop('decision'))
    ticket['timeline'] = json.loads(ticket.pop('timeline'))
    
    # Rename fields
    ticket['trustScore'] = ticket.pop('trust_score')
    ticket['createdAt'] = ticket.pop('created_at')
    ticket['resolvedAt'] = ticket.pop('resolved_at')
    ticket['hasImage'] = bool(ticket.pop('has_image'))
    
    conn.close()
    return ticket

def get_ticket_count() -> int:
    """Get total number of tickets"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tickets")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_tickets_by_status(status: str) -> List[dict]:
    """Get tickets filtered by status"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC", (status,))
    rows = cursor.fetchall()
    
    tickets = []
    for row in rows:
        ticket = dict(row)
        ticket['customer'] = {
            'id': ticket.pop('customer_id'),
            'name': ticket.pop('customer_name'),
            'email': ticket.pop('customer_email'),
            'tier': ticket.pop('customer_tier'),
            'orders': ticket.pop('customer_orders'),
            'ltv': ticket.pop('customer_ltv')
        }
        ticket['aiAnalysis'] = json.loads(ticket.pop('ai_analysis'))
        ticket['decision'] = json.loads(ticket.pop('decision'))
        ticket['timeline'] = json.loads(ticket.pop('timeline'))
        ticket['trustScore'] = ticket.pop('trust_score')
        ticket['createdAt'] = ticket.pop('created_at')
        ticket['resolvedAt'] = ticket.pop('resolved_at')
        ticket['hasImage'] = bool(ticket.pop('has_image'))
        tickets.append(ticket)
    
    conn.close()
    return tickets

def delete_all_tickets():
    """Delete all tickets (for testing)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tickets")
    conn.commit()
    conn.close()
    print("✅ All tickets deleted")

# Initialize database on module import
initialize_db()

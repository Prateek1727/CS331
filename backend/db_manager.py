"""
Database Management Utility
Run this script to manage the database
"""
import sys
from database_sqlite import (
    initialize_db, 
    get_all_tickets, 
    get_ticket_count,
    get_tickets_by_status,
    delete_all_tickets
)

def show_menu():
    print("\n" + "="*60)
    print("📊 NeuraDesk Database Manager")
    print("="*60)
    print("1. View all tickets")
    print("2. View ticket count")
    print("3. View tickets by status")
    print("4. Delete all tickets (CAUTION!)")
    print("5. Initialize/Reset database")
    print("6. Exit")
    print("="*60)

def view_all_tickets():
    tickets = get_all_tickets()
    print(f"\n📋 Total Tickets: {len(tickets)}\n")
    
    if not tickets:
        print("No tickets found.")
        return
    
    for ticket in tickets:
        print(f"ID: {ticket['id']}")
        print(f"  Customer: {ticket['customer']['name']} ({ticket['customer']['email']})")
        print(f"  Subject: {ticket['subject']}")
        print(f"  Status: {ticket['status']}")
        print(f"  Priority: {ticket['priority']}")
        print(f"  Trust Score: {ticket['trustScore']}")
        print(f"  Has Image: {ticket['hasImage']}")
        print(f"  Created: {ticket['createdAt']}")
        
        # Show fraud detection verdict if image was analyzed
        if ticket['hasImage'] and ticket.get('aiAnalysis', {}).get('vision'):
            verdict = ticket['aiAnalysis']['vision'].get('verdict', 'N/A')
            print(f"  🔍 Fraud Verdict: {verdict}")
        
        print("-" * 60)

def view_ticket_count():
    count = get_ticket_count()
    print(f"\n📊 Total Tickets in Database: {count}")

def view_by_status():
    print("\nEnter status (auto_resolved, in_progress, fraud_review, escalated): ", end="")
    status = input().strip()
    
    tickets = get_tickets_by_status(status)
    print(f"\n📋 Tickets with status '{status}': {len(tickets)}\n")
    
    if not tickets:
        print(f"No tickets found with status '{status}'.")
        return
    
    for ticket in tickets:
        print(f"ID: {ticket['id']} - {ticket['subject']} (Trust: {ticket['trustScore']})")

def delete_all():
    print("\n⚠️  WARNING: This will delete ALL tickets!")
    print("Type 'DELETE' to confirm: ", end="")
    confirm = input().strip()
    
    if confirm == "DELETE":
        delete_all_tickets()
        print("✅ All tickets deleted successfully!")
    else:
        print("❌ Deletion cancelled.")

def main():
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            view_all_tickets()
        elif choice == "2":
            view_ticket_count()
        elif choice == "3":
            view_by_status()
        elif choice == "4":
            delete_all()
        elif choice == "5":
            initialize_db()
            print("✅ Database initialized!")
        elif choice == "6":
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

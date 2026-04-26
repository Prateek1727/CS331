from typing import List, Dict
import datetime
from models import Ticket
# In-memory database just for demoing live ingestion
import json
import os

# Start with a strictly empty database to force real ingestion only
DB_TICKETS = []

def get_all_tickets() -> List[dict]:
    return DB_TICKETS

def insert_ticket(ticket: dict):
    # insert at the top (newest first)
    DB_TICKETS.insert(0, ticket)

def initialize_db():
    pass

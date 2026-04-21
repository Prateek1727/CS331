"""
Dispute Data Access Layer
Assignment 8 - Part A: Data Access Layer

Handles all database operations for disputes.
"""

from typing import List, Dict, Any, Optional
from .base_dal import BaseDAL
import logging

logger = logging.getLogger(__name__)


class DisputeDAL(BaseDAL):
    """
    Data Access Layer for Dispute operations.
    
    Provides CRUD operations and business-specific queries for disputes.
    """
    
    def __init__(self):
        """Initialize Dispute DAL."""
        super().__init__('disputes')
    
    def create_dispute(self, dispute_code: str, user_id: int, order_id: str, 
                      amount: float, description: str, image_path: str = None) -> Optional[int]:
        """
        Create a new dispute.
        
        Args:
            dispute_code: Unique dispute code (e.g., DISP-000001)
            user_id: ID of the user creating the dispute
            order_id: Order identifier
            amount: Dispute amount
            description: Dispute description
            image_path: Path to evidence image
        
        Returns:
            ID of created dispute
        """
        data = {
            'dispute_code': dispute_code,
            'user_id': user_id,
            'order_id': order_id,
            'amount': amount,
            'description': description,
            'status': 'pending'
        }
        
        if image_path:
            data['image_path'] = image_path
        
        try:
            dispute_id = self.insert(data)
            logger.info(f"Created dispute {dispute_code} with ID {dispute_id}")
            return dispute_id
        except Exception as e:
            logger.error(f"Error creating dispute: {e}")
            raise
    
    def get_dispute_by_code(self, dispute_code: str) -> Optional[Dict]:
        """
        Get dispute by dispute code.
        
        Args:
            dispute_code: Dispute code
        
        Returns:
            Dispute dictionary or None
        """
        query = "SELECT * FROM disputes WHERE dispute_code = %s"
        
        try:
            result = self.execute_query(query, (dispute_code,), fetch_one=True)
            if result:
                return self._row_to_dict(result)
            return None
        except Exception as e:
            logger.error(f"Error getting dispute by code: {e}")
            raise
    
    def get_disputes_by_user(self, user_id: int, limit: int = 20) -> List[Dict]:
        """
        Get all disputes for a specific user.
        
        Args:
            user_id: User ID
            limit: Maximum number of results
        
        Returns:
            List of dispute dictionaries
        """
        query = """
            SELECT * FROM disputes 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT %s
        """
        
        try:
            results = self.execute_query(query, (user_id, limit), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting disputes by user: {e}")
            raise
    
    def get_disputes_by_status(self, status: str, limit: int = 50) -> List[Dict]:
        """
        Get disputes by status.
        
        Args:
            status: Dispute status (pending, under_review, approved, rejected, fraud_alert)
            limit: Maximum number of results
        
        Returns:
            List of dispute dictionaries
        """
        query = """
            SELECT * FROM disputes 
            WHERE status = %s 
            ORDER BY created_at DESC 
            LIMIT %s
        """
        
        try:
            results = self.execute_query(query, (status, limit), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting disputes by status: {e}")
            raise
    
    def update_dispute_status(self, dispute_id: int, status: str, 
                             decision: str = None, trust_score: float = None,
                             confidence: str = None, agent_id: int = None,
                             agent_notes: str = None) -> bool:
        """
        Update dispute status and related fields.
        
        Args:
            dispute_id: Dispute ID
            status: New status
            decision: Decision type
            trust_score: Trust score
            confidence: Confidence level
            agent_id: Agent ID who updated
            agent_notes: Agent notes
        
        Returns:
            True if successful
        """
        data = {'status': status}
        
        if decision:
            data['decision'] = decision
        if trust_score is not None:
            data['trust_score'] = trust_score
        if confidence:
            data['confidence'] = confidence
        if agent_id:
            data['agent_id'] = agent_id
        if agent_notes:
            data['agent_notes'] = agent_notes
        
        # Set resolution timestamp for final statuses
        if status in ['approved', 'rejected', 'fraud_alert']:
            data['resolution_timestamp'] = 'CURRENT_TIMESTAMP'
        
        try:
            return self.update(dispute_id, data)
        except Exception as e:
            logger.error(f"Error updating dispute status: {e}")
            raise
    
    def get_dispute_statistics(self) -> Dict[str, Any]:
        """
        Get dispute statistics.
        
        Returns:
            Dictionary with statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_disputes,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                COUNT(CASE WHEN status = 'under_review' THEN 1 END) as under_review,
                COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved,
                COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected,
                COUNT(CASE WHEN status = 'fraud_alert' THEN 1 END) as fraud_alert,
                AVG(trust_score) as avg_trust_score,
                SUM(CASE WHEN status = 'approved' THEN amount ELSE 0 END) as total_refunded
            FROM disputes
        """
        
        try:
            result = self.execute_query(query, fetch_one=True)
            if result:
                return {
                    'total_disputes': result[0],
                    'pending': result[1],
                    'under_review': result[2],
                    'approved': result[3],
                    'rejected': result[4],
                    'fraud_alert': result[5],
                    'avg_trust_score': float(result[6]) if result[6] else 0.0,
                    'total_refunded': float(result[7]) if result[7] else 0.0
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting dispute statistics: {e}")
            raise
    
    def _row_to_dict(self, row: tuple) -> Dict:
        """Convert database row to dictionary."""
        if not row:
            return {}
        
        return {
            'dispute_id': row[0],
            'dispute_code': row[1],
            'user_id': row[2],
            'order_id': row[3],
            'amount': float(row[4]),
            'description': row[5],
            'image_path': row[6],
            'status': row[7],
            'decision': row[8],
            'trust_score': float(row[9]) if row[9] else None,
            'confidence': row[10],
            'agent_id': row[11],
            'agent_notes': row[12],
            'created_at': row[13],
            'updated_at': row[14],
            'resolution_timestamp': row[15]
        }

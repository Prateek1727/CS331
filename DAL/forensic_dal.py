"""
Forensic Results Data Access Layer
Assignment 8 - Part A: Data Access Layer

Handles all database operations for forensic analysis results.
"""

from typing import List, Dict, Any, Optional
from .base_dal import BaseDAL
import logging

logger = logging.getLogger(__name__)


class ForensicDAL(BaseDAL):
    """
    Data Access Layer for Forensic Results operations.
    
    Provides CRUD operations for forensic analysis results.
    """
    
    def __init__(self):
        """Initialize Forensic DAL."""
        super().__init__('forensic_results')
    
    def create_forensic_result(self, dispute_id: int, metadata_score: float,
                              ela_score: float, ai_score: float, risk_level: str,
                              risk_color: str, flags: List[str] = None,
                              metadata_label: str = None, ela_label: str = None,
                              processing_time_ms: int = None) -> Optional[int]:
        """
        Create a new forensic analysis result.
        
        Args:
            dispute_id: ID of the dispute
            metadata_score: Metadata analysis score (0.0-1.0)
            ela_score: ELA analysis score (0.0-1.0)
            ai_score: AI analysis score (0.0-1.0)
            risk_level: Risk level (low, medium, high, critical)
            risk_color: Risk color code
            flags: List of detected fraud indicators
            metadata_label: Metadata analysis label
            ela_label: ELA analysis label
            processing_time_ms: Processing time in milliseconds
        
        Returns:
            ID of created forensic result
        """
        data = {
            'dispute_id': dispute_id,
            'metadata_score': metadata_score,
            'ela_score': ela_score,
            'ai_score': ai_score,
            'risk_level': risk_level,
            'risk_color': risk_color
        }
        
        if flags:
            # Convert list to PostgreSQL array format
            data['flags'] = flags
        if metadata_label:
            data['metadata_label'] = metadata_label
        if ela_label:
            data['ela_label'] = ela_label
        if processing_time_ms:
            data['processing_time_ms'] = processing_time_ms
        
        try:
            forensic_id = self.insert(data)
            logger.info(f"Created forensic result for dispute {dispute_id} with ID {forensic_id}")
            return forensic_id
        except Exception as e:
            logger.error(f"Error creating forensic result: {e}")
            raise
    
    def get_forensic_by_dispute(self, dispute_id: int) -> Optional[Dict]:
        """
        Get forensic result for a specific dispute.
        
        Args:
            dispute_id: Dispute ID
        
        Returns:
            Forensic result dictionary or None
        """
        query = "SELECT * FROM forensic_results WHERE dispute_id = %s"
        
        try:
            result = self.execute_query(query, (dispute_id,), fetch_one=True)
            if result:
                return self._row_to_dict(result)
            return None
        except Exception as e:
            logger.error(f"Error getting forensic result by dispute: {e}")
            raise
    
    def get_high_risk_results(self, limit: int = 20) -> List[Dict]:
        """
        Get high-risk forensic results.
        
        Args:
            limit: Maximum number of results
        
        Returns:
            List of forensic result dictionaries
        """
        query = """
            SELECT * FROM forensic_results 
            WHERE risk_level IN ('high', 'critical')
            ORDER BY analysis_timestamp DESC 
            LIMIT %s
        """
        
        try:
            results = self.execute_query(query, (limit,), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting high risk results: {e}")
            raise
    
    def get_results_by_risk_level(self, risk_level: str, limit: int = 50) -> List[Dict]:
        """
        Get forensic results by risk level.
        
        Args:
            risk_level: Risk level (low, medium, high, critical)
            limit: Maximum number of results
        
        Returns:
            List of forensic result dictionaries
        """
        query = """
            SELECT * FROM forensic_results 
            WHERE risk_level = %s
            ORDER BY analysis_timestamp DESC 
            LIMIT %s
        """
        
        try:
            results = self.execute_query(query, (risk_level, limit), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting results by risk level: {e}")
            raise
    
    def get_results_with_flags(self, flag: str, limit: int = 20) -> List[Dict]:
        """
        Get forensic results containing a specific flag.
        
        Args:
            flag: Flag to search for
            limit: Maximum number of results
        
        Returns:
            List of forensic result dictionaries
        """
        query = """
            SELECT * FROM forensic_results 
            WHERE %s = ANY(flags)
            ORDER BY analysis_timestamp DESC 
            LIMIT %s
        """
        
        try:
            results = self.execute_query(query, (flag, limit), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting results with flag: {e}")
            raise
    
    def get_average_scores(self) -> Dict[str, float]:
        """
        Get average scores across all forensic results.
        
        Returns:
            Dictionary with average scores
        """
        query = """
            SELECT 
                AVG(metadata_score) as avg_metadata,
                AVG(ela_score) as avg_ela,
                AVG(ai_score) as avg_ai
            FROM forensic_results
        """
        
        try:
            result = self.execute_query(query, fetch_one=True)
            if result:
                return {
                    'avg_metadata_score': float(result[0]) if result[0] else 0.0,
                    'avg_ela_score': float(result[1]) if result[1] else 0.0,
                    'avg_ai_score': float(result[2]) if result[2] else 0.0
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting average scores: {e}")
            raise
    
    def _row_to_dict(self, row: tuple) -> Dict:
        """Convert database row to dictionary."""
        if not row:
            return {}
        
        return {
            'forensic_id': row[0],
            'dispute_id': row[1],
            'metadata_score': float(row[2]),
            'ela_score': float(row[3]),
            'ai_score': float(row[4]),
            'risk_level': row[5],
            'risk_color': row[6],
            'flags': list(row[7]) if row[7] else [],
            'metadata_label': row[8],
            'ela_label': row[9],
            'analysis_timestamp': row[10],
            'processing_time_ms': row[11]
        }

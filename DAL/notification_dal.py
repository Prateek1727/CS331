"""
Notification Data Access Layer
Assignment 8 - Part A: Data Access Layer

Handles all database operations for notifications.
"""

from typing import List, Dict, Any, Optional
from .base_dal import BaseDAL
import logging

logger = logging.getLogger(__name__)


class NotificationDAL(BaseDAL):
    """
    Data Access Layer for Notification operations.
    
    Provides CRUD operations for user notifications.
    """
    
    def __init__(self):
        """Initialize Notification DAL."""
        super().__init__('notifications')
    
    def create_notification(self, notification_code: str, user_id: int, 
                          subject: str, message: str, notification_type: str = 'email',
                          dispute_id: int = None, priority: str = 'medium') -> Optional[int]:
        """
        Create a new notification.
        
        Args:
            notification_code: Unique notification code
            user_id: ID of the user to notify
            subject: Notification subject
            message: Notification message
            notification_type: Type (email, sms, push)
            dispute_id: Related dispute ID (optional)
            priority: Priority level (low, medium, high, urgent)
        
        Returns:
            ID of created notification
        """
        data = {
            'notification_code': notification_code,
            'user_id': user_id,
            'subject': subject,
            'message': message,
            'type': notification_type,
            'priority': priority,
            'status': 'sent'
        }
        
        if dispute_id:
            data['dispute_id'] = dispute_id
        
        try:
            notification_id = self.insert(data)
            logger.info(f"Created notification {notification_code} with ID {notification_id}")
            return notification_id
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
            raise
    
    def get_notifications_by_user(self, user_id: int, limit: int = 20, 
                                 unread_only: bool = False) -> List[Dict]:
        """
        Get notifications for a specific user.
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            unread_only: If True, only return unread notifications
        
        Returns:
            List of notification dictionaries
        """
        if unread_only:
            query = """
                SELECT * FROM notifications 
                WHERE user_id = %s AND read_at IS NULL
                ORDER BY sent_at DESC 
                LIMIT %s
            """
        else:
            query = """
                SELECT * FROM notifications 
                WHERE user_id = %s
                ORDER BY sent_at DESC 
                LIMIT %s
            """
        
        try:
            results = self.execute_query(query, (user_id, limit), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting notifications by user: {e}")
            raise
    
    def get_notifications_by_dispute(self, dispute_id: int) -> List[Dict]:
        """
        Get all notifications related to a specific dispute.
        
        Args:
            dispute_id: Dispute ID
        
        Returns:
            List of notification dictionaries
        """
        query = """
            SELECT * FROM notifications 
            WHERE dispute_id = %s
            ORDER BY sent_at DESC
        """
        
        try:
            results = self.execute_query(query, (dispute_id,), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting notifications by dispute: {e}")
            raise
    
    def mark_as_read(self, notification_id: int) -> bool:
        """
        Mark a notification as read.
        
        Args:
            notification_id: Notification ID
        
        Returns:
            True if successful
        """
        query = "UPDATE notifications SET read_at = CURRENT_TIMESTAMP WHERE notification_id = %s"
        
        try:
            self.execute_query(query, (notification_id,), commit=True)
            logger.info(f"Marked notification {notification_id} as read")
            return True
        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")
            raise
    
    def mark_all_as_read(self, user_id: int) -> bool:
        """
        Mark all notifications for a user as read.
        
        Args:
            user_id: User ID
        
        Returns:
            True if successful
        """
        query = """
            UPDATE notifications 
            SET read_at = CURRENT_TIMESTAMP 
            WHERE user_id = %s AND read_at IS NULL
        """
        
        try:
            self.execute_query(query, (user_id,), commit=True)
            logger.info(f"Marked all notifications as read for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error marking all notifications as read: {e}")
            raise
    
    def get_unread_count(self, user_id: int) -> int:
        """
        Get count of unread notifications for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            Number of unread notifications
        """
        query = "SELECT COUNT(*) FROM notifications WHERE user_id = %s AND read_at IS NULL"
        
        try:
            result = self.execute_query(query, (user_id,), fetch_one=True)
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting unread count: {e}")
            raise
    
    def get_notifications_by_priority(self, priority: str, limit: int = 50) -> List[Dict]:
        """
        Get notifications by priority level.
        
        Args:
            priority: Priority level (low, medium, high, urgent)
            limit: Maximum number of results
        
        Returns:
            List of notification dictionaries
        """
        query = """
            SELECT * FROM notifications 
            WHERE priority = %s
            ORDER BY sent_at DESC 
            LIMIT %s
        """
        
        try:
            results = self.execute_query(query, (priority, limit), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting notifications by priority: {e}")
            raise
    
    def delete_old_notifications(self, days: int = 30) -> int:
        """
        Delete notifications older than specified days.
        
        Args:
            days: Number of days to keep
        
        Returns:
            Number of deleted notifications
        """
        query = """
            DELETE FROM notifications 
            WHERE sent_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
        """
        
        try:
            # Get count first
            count_query = """
                SELECT COUNT(*) FROM notifications 
                WHERE sent_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
            """
            result = self.execute_query(count_query, (days,), fetch_one=True)
            count = result[0] if result else 0
            
            # Delete
            self.execute_query(query, (days,), commit=True)
            logger.info(f"Deleted {count} old notifications")
            return count
        except Exception as e:
            logger.error(f"Error deleting old notifications: {e}")
            raise
    
    def _row_to_dict(self, row: tuple) -> Dict:
        """Convert database row to dictionary."""
        if not row:
            return {}
        
        return {
            'notification_id': row[0],
            'notification_code': row[1],
            'user_id': row[2],
            'dispute_id': row[3],
            'type': row[4],
            'subject': row[5],
            'message': row[6],
            'priority': row[7],
            'status': row[8],
            'sent_at': row[9],
            'read_at': row[10]
        }

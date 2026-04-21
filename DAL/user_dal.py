"""
User Data Access Layer
Assignment 8 - Part A: Data Access Layer

Handles all database operations for users.
"""

from typing import List, Dict, Any, Optional
from .base_dal import BaseDAL
import logging
import hashlib

logger = logging.getLogger(__name__)


class UserDAL(BaseDAL):
    """
    Data Access Layer for User operations.
    
    Provides CRUD operations and authentication for users.
    """
    
    def __init__(self):
        """Initialize User DAL."""
        super().__init__('users')
    
    def create_user(self, username: str, email: str, password: str, 
                   full_name: str = None, phone: str = None, 
                   role: str = 'customer') -> Optional[int]:
        """
        Create a new user.
        
        Args:
            username: Unique username
            email: User email
            password: Plain text password (will be hashed)
            full_name: User's full name
            phone: Phone number
            role: User role (customer, agent, admin)
        
        Returns:
            ID of created user
        """
        # Hash password
        password_hash = self._hash_password(password)
        
        data = {
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'role': role
        }
        
        if full_name:
            data['full_name'] = full_name
        if phone:
            data['phone'] = phone
        
        try:
            user_id = self.insert(data)
            logger.info(f"Created user {username} with ID {user_id}")
            return user_id
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        Get user by email.
        
        Args:
            email: User email
        
        Returns:
            User dictionary or None
        """
        query = "SELECT * FROM users WHERE email = %s"
        
        try:
            result = self.execute_query(query, (email,), fetch_one=True)
            if result:
                return self._row_to_dict(result)
            return None
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            raise
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """
        Get user by username.
        
        Args:
            username: Username
        
        Returns:
            User dictionary or None
        """
        query = "SELECT * FROM users WHERE username = %s"
        
        try:
            result = self.execute_query(query, (username,), fetch_one=True)
            if result:
                return self._row_to_dict(result)
            return None
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            raise
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """
        Authenticate user with email and password.
        
        Args:
            email: User email
            password: Plain text password
        
        Returns:
            User dictionary if authenticated, None otherwise
        """
        user = self.get_user_by_email(email)
        
        if not user:
            logger.warning(f"Authentication failed: User not found for email {email}")
            return None
        
        password_hash = self._hash_password(password)
        
        if user['password_hash'] != password_hash:
            logger.warning(f"Authentication failed: Invalid password for {email}")
            return None
        
        # Update last login
        self.update_last_login(user['user_id'])
        
        logger.info(f"User {email} authenticated successfully")
        return user
    
    def update_user(self, user_id: int, **kwargs) -> bool:
        """
        Update user information.
        
        Args:
            user_id: User ID
            **kwargs: Fields to update (full_name, phone, etc.)
        
        Returns:
            True if successful
        """
        # Remove None values
        data = {k: v for k, v in kwargs.items() if v is not None}
        
        if not data:
            logger.warning("No data to update")
            return False
        
        # Hash password if provided
        if 'password' in data:
            data['password_hash'] = self._hash_password(data.pop('password'))
        
        try:
            return self.update(user_id, data)
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            raise
    
    def update_last_login(self, user_id: int) -> bool:
        """
        Update user's last login timestamp.
        
        Args:
            user_id: User ID
        
        Returns:
            True if successful
        """
        query = "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = %s"
        
        try:
            self.execute_query(query, (user_id,), commit=True)
            return True
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
            raise
    
    def get_users_by_role(self, role: str, limit: int = 50) -> List[Dict]:
        """
        Get all users with a specific role.
        
        Args:
            role: User role (customer, agent, admin)
            limit: Maximum number of results
        
        Returns:
            List of user dictionaries
        """
        query = """
            SELECT * FROM users 
            WHERE role = %s AND is_active = TRUE
            ORDER BY created_at DESC 
            LIMIT %s
        """
        
        try:
            results = self.execute_query(query, (role, limit), fetch_all=True)
            return [self._row_to_dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting users by role: {e}")
            raise
    
    def deactivate_user(self, user_id: int) -> bool:
        """
        Deactivate a user account.
        
        Args:
            user_id: User ID
        
        Returns:
            True if successful
        """
        return self.update(user_id, {'is_active': False})
    
    def activate_user(self, user_id: int) -> bool:
        """
        Activate a user account.
        
        Args:
            user_id: User ID
        
        Returns:
            True if successful
        """
        return self.update(user_id, {'is_active': True})
    
    def _hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256.
        
        Args:
            password: Plain text password
        
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _row_to_dict(self, row: tuple) -> Dict:
        """Convert database row to dictionary."""
        if not row:
            return {}
        
        return {
            'user_id': row[0],
            'username': row[1],
            'email': row[2],
            'password_hash': row[3],
            'full_name': row[4],
            'phone': row[5],
            'role': row[6],
            'is_active': row[7],
            'created_at': row[8],
            'updated_at': row[9],
            'last_login': row[10]
        }

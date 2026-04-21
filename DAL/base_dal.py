"""
Base Data Access Layer
Assignment 8 - Part A: Data Access Layer

Provides base class for all DAL implementations with common CRUD operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

from ..database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class BaseDAL(ABC):
    """
    Abstract base class for Data Access Layer implementations.
    
    Provides common CRUD operations and database interaction patterns.
    All specific DAL classes should inherit from this base class.
    """
    
    def __init__(self, table_name: str):
        """
        Initialize the DAL.
        
        Args:
            table_name: Name of the database table
        """
        self.table_name = table_name
    
    def execute_query(self, query: str, params: tuple = None, fetch_one=False, fetch_all=False, commit=False):
        """
        Execute a database query.
        
        Args:
            query: SQL query to execute
            params: Query parameters (for parameterized queries)
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results
            commit: Whether to commit the transaction
        
        Returns:
            Query results or None
        """
        try:
            with DatabaseConnection.get_cursor(commit=commit) as cursor:
                cursor.execute(query, params)
                
                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                elif not commit:
                    return cursor.fetchall()
                
                return None
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
    
    def insert(self, data: Dict[str, Any]) -> Optional[int]:
        """
        Insert a record into the table.
        
        Args:
            data: Dictionary of column names and values
        
        Returns:
            ID of the inserted record
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders}) RETURNING *"
        
        try:
            result = self.execute_query(query, tuple(data.values()), fetch_one=True, commit=True)
            logger.info(f"Inserted record into {self.table_name}")
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error inserting into {self.table_name}: {e}")
            raise
    
    def update(self, record_id: int, data: Dict[str, Any], id_column: str = None) -> bool:
        """
        Update a record in the table.
        
        Args:
            record_id: ID of the record to update
            data: Dictionary of column names and values to update
            id_column: Name of the ID column (defaults to table_name + '_id')
        
        Returns:
            True if update was successful
        """
        if id_column is None:
            id_column = f"{self.table_name.rstrip('s')}_id"
        
        set_clause = ', '.join([f"{key} = %s" for key in data.keys()])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE {id_column} = %s"
        params = tuple(data.values()) + (record_id,)
        
        try:
            self.execute_query(query, params, commit=True)
            logger.info(f"Updated record {record_id} in {self.table_name}")
            return True
        except Exception as e:
            logger.error(f"Error updating {self.table_name}: {e}")
            raise
    
    def delete(self, record_id: int, id_column: str = None) -> bool:
        """
        Delete a record from the table.
        
        Args:
            record_id: ID of the record to delete
            id_column: Name of the ID column (defaults to table_name + '_id')
        
        Returns:
            True if deletion was successful
        """
        if id_column is None:
            id_column = f"{self.table_name.rstrip('s')}_id"
        
        query = f"DELETE FROM {self.table_name} WHERE {id_column} = %s"
        
        try:
            self.execute_query(query, (record_id,), commit=True)
            logger.info(f"Deleted record {record_id} from {self.table_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting from {self.table_name}: {e}")
            raise
    
    def get_by_id(self, record_id: int, id_column: str = None) -> Optional[tuple]:
        """
        Get a record by ID.
        
        Args:
            record_id: ID of the record
            id_column: Name of the ID column (defaults to table_name + '_id')
        
        Returns:
            Record tuple or None
        """
        if id_column is None:
            id_column = f"{self.table_name.rstrip('s')}_id"
        
        query = f"SELECT * FROM {self.table_name} WHERE {id_column} = %s"
        
        try:
            result = self.execute_query(query, (record_id,), fetch_one=True)
            return result
        except Exception as e:
            logger.error(f"Error getting record from {self.table_name}: {e}")
            raise
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[tuple]:
        """
        Get all records from the table.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
        
        Returns:
            List of record tuples
        """
        query = f"SELECT * FROM {self.table_name} LIMIT %s OFFSET %s"
        
        try:
            results = self.execute_query(query, (limit, offset), fetch_all=True)
            return results if results else []
        except Exception as e:
            logger.error(f"Error getting all records from {self.table_name}: {e}")
            raise
    
    def count(self, where_clause: str = None, params: tuple = None) -> int:
        """
        Count records in the table.
        
        Args:
            where_clause: Optional WHERE clause
            params: Parameters for WHERE clause
        
        Returns:
            Number of records
        """
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        try:
            result = self.execute_query(query, params, fetch_one=True)
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error counting records in {self.table_name}: {e}")
            raise
    
    def exists(self, record_id: int, id_column: str = None) -> bool:
        """
        Check if a record exists.
        
        Args:
            record_id: ID of the record
            id_column: Name of the ID column
        
        Returns:
            True if record exists
        """
        if id_column is None:
            id_column = f"{self.table_name.rstrip('s')}_id"
        
        query = f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE {id_column} = %s)"
        
        try:
            result = self.execute_query(query, (record_id,), fetch_one=True)
            return result[0] if result else False
        except Exception as e:
            logger.error(f"Error checking existence in {self.table_name}: {e}")
            raise

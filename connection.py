"""
Database Connection Management
Assignment 8 - Part A: Data Access Layer

Provides connection pooling and database connection management.
"""

import psycopg2
from psycopg2 import pool, Error
from contextlib import contextmanager
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Database connection manager with connection pooling.
    
    Features:
    - Connection pooling for performance
    - Automatic connection management
    - Error handling and logging
    - Context manager support
    """
    
    _connection_pool = None
    
    @classmethod
    def initialize_pool(cls, minconn=1, maxconn=10):
        """
        Initialize the connection pool.
        
        Args:
            minconn: Minimum number of connections
            maxconn: Maximum number of connections
        """
        try:
            if cls._connection_pool is None:
                cls._connection_pool = psycopg2.pool.SimpleConnectionPool(
                    minconn,
                    maxconn,
                    host=os.getenv('DB_HOST', 'localhost'),
                    port=os.getenv('DB_PORT', '5432'),
                    database=os.getenv('DB_NAME', 'verisupport_db'),
                    user=os.getenv('DB_USER', 'postgres'),
                    password=os.getenv('DB_PASSWORD', 'postgres')
                )
                logger.info("Database connection pool initialized successfully")
        except Error as e:
            logger.error(f"Error initializing connection pool: {e}")
            raise
    
    @classmethod
    def get_connection(cls):
        """
        Get a connection from the pool.
        
        Returns:
            Database connection
        """
        if cls._connection_pool is None:
            cls.initialize_pool()
        
        try:
            connection = cls._connection_pool.getconn()
            logger.debug("Connection retrieved from pool")
            return connection
        except Error as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise
    
    @classmethod
    def return_connection(cls, connection):
        """
        Return a connection to the pool.
        
        Args:
            connection: Database connection to return
        """
        if cls._connection_pool is not None and connection is not None:
            cls._connection_pool.putconn(connection)
            logger.debug("Connection returned to pool")
    
    @classmethod
    def close_all_connections(cls):
        """Close all connections in the pool."""
        if cls._connection_pool is not None:
            cls._connection_pool.closeall()
            cls._connection_pool = None
            logger.info("All database connections closed")
    
    @classmethod
    @contextmanager
    def get_cursor(cls, commit=False):
        """
        Context manager for database cursor.
        
        Args:
            commit: Whether to commit the transaction
        
        Yields:
            Database cursor
        
        Example:
            with DatabaseConnection.get_cursor(commit=True) as cursor:
                cursor.execute("INSERT INTO users ...")
        """
        connection = None
        cursor = None
        
        try:
            connection = cls.get_connection()
            cursor = connection.cursor()
            yield cursor
            
            if commit:
                connection.commit()
                logger.debug("Transaction committed")
        except Error as e:
            if connection:
                connection.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                cls.return_connection(connection)


def test_connection():
    """Test database connection."""
    try:
        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"✓ Database connection successful!")
            print(f"  PostgreSQL version: {version[0]}")
            return True
    except Error as e:
        print(f"✗ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing database connection...")
    test_connection()
    DatabaseConnection.close_all_connections()

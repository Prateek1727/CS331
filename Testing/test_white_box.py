"""
White Box Testing
Assignment 8 - Part B: Testing

Tests internal code structure, logic, and paths.
Includes: Statement Coverage, Branch Coverage, Path Coverage, Loop Testing
"""

import sys
import os
import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Part_A_DAL.dal.user_dal import UserDAL
from Part_A_DAL.dal.dispute_dal import DisputeDAL
from Part_A_DAL.dal.forensic_dal import ForensicDAL
from Part_A_DAL.dal.notification_dal import NotificationDAL


class TestUserDALWhiteBox:
    """White box tests for User DAL."""
    
    def test_create_user_statement_coverage(self):
        """Test that create_user executes all statements."""
        dal = UserDAL()
        
        # Test with all parameters (covers all assignment statements)
        user_id = dal.create_user(
            username=f"test_user_{os.urandom(4).hex()}",
            email=f"test_{os.urandom(4).hex()}@test.com",
            password="testpass123",
            full_name="Test User",
            phone="+1234567890",
            role="customer"
        )
        
        assert user_id is not None
        assert isinstance(user_id, int)
    
    def test_create_user_branch_coverage(self):
        """Test all branches in create_user."""
        dal = UserDAL()
        
        # Branch 1: With optional parameters
        user_id1 = dal.create_user(
            username=f"test_user_{os.urandom(4).hex()}",
            email=f"test_{os.urandom(4).hex()}@test.com",
            password="testpass123",
            full_name="Test User",
            phone="+1234567890"
        )
        assert user_id1 is not None
        
        # Branch 2: Without optional parameters
        user_id2 = dal.create_user(
            username=f"test_user_{os.urandom(4).hex()}",
            email=f"test_{os.urandom(4).hex()}@test.com",
            password="testpass123"
        )
        assert user_id2 is not None
    
    def test_authenticate_user_path_coverage(self):
        """Test all execution paths in authenticate_user."""
        dal = UserDAL()
        
        # Create test user
        email = f"auth_test_{os.urandom(4).hex()}@test.com"
        dal.create_user(
            username=f"auth_user_{os.urandom(4).hex()}",
            email=email,
            password="correctpass"
        )
        
        # Path 1: User exists, correct password
        result = dal.authenticate_user(email, "correctpass")
        assert result is not None
        assert result['email'] == email
        
        # Path 2: User exists, wrong password
        result = dal.authenticate_user(email, "wrongpass")
        assert result is None
        
        # Path 3: User doesn't exist
        result = dal.authenticate_user("nonexistent@test.com", "anypass")
        assert result is None
    
    def test_update_user_branch_coverage(self):
        """Test branches in update_user."""
        dal = UserDAL()
        
        # Create test user
        user_id = dal.create_user(
            username=f"update_test_{os.urandom(4).hex()}",
            email=f"update_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
        
        # Branch 1: Update with data
        result = dal.update_user(user_id, full_name="Updated Name")
        assert result is True
        
        # Branch 2: Update with no data (all None)
        result = dal.update_user(user_id)
        assert result is False
        
        # Branch 3: Update with password (triggers password hashing branch)
        result = dal.update_user(user_id, password="newpass123")
        assert result is True
    
    def test_get_users_by_role_loop_coverage(self):
        """Test loop execution in get_users_by_role."""
        dal = UserDAL()
        
        # Test with limit (loop executes)
        users = dal.get_users_by_role("customer", limit=5)
        assert isinstance(users, list)
        
        # Test with no results (loop doesn't execute)
        users = dal.get_users_by_role("nonexistent_role", limit=5)
        assert users == []


class TestDisputeDALWhiteBox:
    """White box tests for Dispute DAL."""
    
    def test_create_dispute_statement_coverage(self):
        """Test all statements in create_dispute."""
        dal = DisputeDAL()
        user_dal = UserDAL()
        
        # Create test user
        user_id = user_dal.create_user(
            username=f"dispute_user_{os.urandom(4).hex()}",
            email=f"dispute_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
        
        # Test with all parameters
        dispute_id = dal.create_dispute(
            dispute_code=f"DISP-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            order_id=f"ORD-TEST-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test dispute for statement coverage testing",
            image_path="/test/image.jpg"
        )
        
        assert dispute_id is not None
    
    def test_create_dispute_branch_coverage(self):
        """Test branches in create_dispute."""
        dal = DisputeDAL()
        user_dal = UserDAL()
        
        user_id = user_dal.create_user(
            username=f"dispute_user_{os.urandom(4).hex()}",
            email=f"dispute_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
        
        # Branch 1: With image_path
        dispute_id1 = dal.create_dispute(
            dispute_code=f"DISP-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            order_id=f"ORD-TEST-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test dispute with image path",
            image_path="/test/image.jpg"
        )
        assert dispute_id1 is not None
        
        # Branch 2: Without image_path
        dispute_id2 = dal.create_dispute(
            dispute_code=f"DISP-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            order_id=f"ORD-TEST-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test dispute without image path"
        )
        assert dispute_id2 is not None
    
    def test_update_dispute_status_path_coverage(self):
        """Test all paths in update_dispute_status."""
        dal = DisputeDAL()
        user_dal = UserDAL()
        
        user_id = user_dal.create_user(
            username=f"dispute_user_{os.urandom(4).hex()}",
            email=f"dispute_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
        
        dispute_id = dal.create_dispute(
            dispute_code=f"DISP-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            order_id=f"ORD-TEST-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test dispute for path coverage"
        )
        
        # Path 1: Update with all optional parameters
        result = dal.update_dispute_status(
            dispute_id=dispute_id,
            status="under_review",
            decision="manual_review",
            trust_score=0.75,
            confidence="medium",
            agent_id=1,
            agent_notes="Test notes"
        )
        assert result is True
        
        # Path 2: Update with only status (final status)
        result = dal.update_dispute_status(
            dispute_id=dispute_id,
            status="approved"
        )
        assert result is True
        
        # Path 3: Update with non-final status
        dispute_id2 = dal.create_dispute(
            dispute_code=f"DISP-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            order_id=f"ORD-TEST-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test dispute 2"
        )
        result = dal.update_dispute_status(
            dispute_id=dispute_id2,
            status="pending"
        )
        assert result is True


class TestForensicDALWhiteBox:
    """White box tests for Forensic DAL."""
    
    def test_create_forensic_result_branch_coverage(self):
        """Test all branches in create_forensic_result."""
        forensic_dal = ForensicDAL()
        dispute_dal = DisputeDAL()
        user_dal = UserDAL()
        
        # Create test dispute
        user_id = user_dal.create_user(
            username=f"forensic_user_{os.urandom(4).hex()}",
            email=f"forensic_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
        
        dispute_id = dispute_dal.create_dispute(
            dispute_code=f"DISP-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            order_id=f"ORD-TEST-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test dispute for forensic"
        )
        
        # Branch 1: With all optional parameters
        forensic_id1 = forensic_dal.create_forensic_result(
            dispute_id=dispute_id,
            metadata_score=0.85,
            ela_score=0.72,
            ai_score=0.68,
            risk_level="medium",
            risk_color="#FFA500",
            flags=["flag1", "flag2"],
            metadata_label="Suspicious",
            ela_label="Moderate",
            processing_time_ms=1000
        )
        assert forensic_id1 is not None
        
        # Branch 2: Without optional parameters
        dispute_id2 = dispute_dal.create_dispute(
            dispute_code=f"DISP-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            order_id=f"ORD-TEST-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test dispute 2"
        )
        
        forensic_id2 = forensic_dal.create_forensic_result(
            dispute_id=dispute_id2,
            metadata_score=0.90,
            ela_score=0.85,
            ai_score=0.80,
            risk_level="low",
            risk_color="#00FF00"
        )
        assert forensic_id2 is not None
    
    def test_get_forensic_by_dispute_path_coverage(self):
        """Test paths in get_forensic_by_dispute."""
        forensic_dal = ForensicDAL()
        
        # Path 1: Forensic result exists
        # (Assuming there's at least one forensic result from previous tests)
        result = forensic_dal.get_forensic_by_dispute(1)
        # Result may or may not exist, just testing the path
        
        # Path 2: Forensic result doesn't exist
        result = forensic_dal.get_forensic_by_dispute(999999)
        assert result is None


class TestNotificationDALWhiteBox:
    """White box tests for Notification DAL."""
    
    def test_create_notification_branch_coverage(self):
        """Test branches in create_notification."""
        notif_dal = NotificationDAL()
        user_dal = UserDAL()
        
        user_id = user_dal.create_user(
            username=f"notif_user_{os.urandom(4).hex()}",
            email=f"notif_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
        
        # Branch 1: With dispute_id
        notif_id1 = notif_dal.create_notification(
            notification_code=f"NOTIF-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            subject="Test Notification",
            message="Test message",
            dispute_id=1
        )
        assert notif_id1 is not None
        
        # Branch 2: Without dispute_id
        notif_id2 = notif_dal.create_notification(
            notification_code=f"NOTIF-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            subject="Test Notification 2",
            message="Test message 2"
        )
        assert notif_id2 is not None
    
    def test_get_notifications_by_user_branch_coverage(self):
        """Test branches in get_notifications_by_user."""
        notif_dal = NotificationDAL()
        user_dal = UserDAL()
        
        user_id = user_dal.create_user(
            username=f"notif_user_{os.urandom(4).hex()}",
            email=f"notif_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
        
        # Create test notification
        notif_dal.create_notification(
            notification_code=f"NOTIF-TEST-{os.urandom(4).hex()}",
            user_id=user_id,
            subject="Test",
            message="Test"
        )
        
        # Branch 1: unread_only=True
        unread = notif_dal.get_notifications_by_user(user_id, unread_only=True)
        assert isinstance(unread, list)
        
        # Branch 2: unread_only=False
        all_notifs = notif_dal.get_notifications_by_user(user_id, unread_only=False)
        assert isinstance(all_notifs, list)
        assert len(all_notifs) >= len(unread)


class TestBaseDALWhiteBox:
    """White box tests for Base DAL methods."""
    
    def test_count_with_where_clause_branch(self):
        """Test count method with and without WHERE clause."""
        dal = UserDAL()
        
        # Branch 1: Without WHERE clause
        count1 = dal.count()
        assert count1 >= 0
        
        # Branch 2: With WHERE clause
        count2 = dal.count("role = %s", ("customer",))
        assert count2 >= 0
    
    def test_exists_method_paths(self):
        """Test exists method paths."""
        dal = UserDAL()
        
        # Path 1: Record exists
        result = dal.exists(1)
        # Result depends on database state
        
        # Path 2: Record doesn't exist
        result = dal.exists(999999)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

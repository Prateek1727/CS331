"""
Black Box Testing
Assignment 8 - Part B: Testing

Tests functionality without knowledge of internal implementation.
Includes: Equivalence Partitioning, Boundary Value Analysis, Decision Table Testing
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


class TestDisputeAmountBoundaryValues:
    """Boundary Value Analysis for dispute amounts."""
    
    def setup_method(self):
        """Setup test user."""
        self.user_dal = UserDAL()
        self.dispute_dal = DisputeDAL()
        
        self.user_id = self.user_dal.create_user(
            username=f"boundary_user_{os.urandom(4).hex()}",
            email=f"boundary_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
    
    def test_amount_at_minimum_boundary(self):
        """Test amount at minimum boundary (1.00)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-BVA-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-BVA-{os.urandom(4).hex()}",
            amount=1.00,
            description="Test at minimum boundary value"
        )
        assert dispute_id is not None
    
    def test_amount_just_above_minimum(self):
        """Test amount just above minimum (1.01)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-BVA-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-BVA-{os.urandom(4).hex()}",
            amount=1.01,
            description="Test just above minimum boundary"
        )
        assert dispute_id is not None
    
    def test_amount_at_maximum_boundary(self):
        """Test amount at maximum boundary (10000.00)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-BVA-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-BVA-{os.urandom(4).hex()}",
            amount=10000.00,
            description="Test at maximum boundary value"
        )
        assert dispute_id is not None
    
    def test_amount_just_below_maximum(self):
        """Test amount just below maximum (9999.99)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-BVA-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-BVA-{os.urandom(4).hex()}",
            amount=9999.99,
            description="Test just below maximum boundary"
        )
        assert dispute_id is not None
    
    def test_amount_in_middle_range(self):
        """Test amount in middle of valid range (5000.00)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-BVA-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-BVA-{os.urandom(4).hex()}",
            amount=5000.00,
            description="Test in middle of valid range"
        )
        assert dispute_id is not None


class TestDisputeDescriptionBoundaryValues:
    """Boundary Value Analysis for dispute descriptions."""
    
    def setup_method(self):
        """Setup test user."""
        self.user_dal = UserDAL()
        self.dispute_dal = DisputeDAL()
        
        self.user_id = self.user_dal.create_user(
            username=f"desc_user_{os.urandom(4).hex()}",
            email=f"desc_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
    
    def test_description_at_minimum_length(self):
        """Test description at minimum length (10 characters)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-DESC-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-DESC-{os.urandom(4).hex()}",
            amount=100.00,
            description="1234567890"  # Exactly 10 characters
        )
        assert dispute_id is not None
    
    def test_description_just_above_minimum(self):
        """Test description just above minimum (11 characters)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-DESC-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-DESC-{os.urandom(4).hex()}",
            amount=100.00,
            description="12345678901"  # 11 characters
        )
        assert dispute_id is not None
    
    def test_description_at_maximum_length(self):
        """Test description at maximum length (500 characters)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-DESC-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-DESC-{os.urandom(4).hex()}",
            amount=100.00,
            description="A" * 500  # Exactly 500 characters
        )
        assert dispute_id is not None
    
    def test_description_just_below_maximum(self):
        """Test description just below maximum (499 characters)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-DESC-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-DESC-{os.urandom(4).hex()}",
            amount=100.00,
            description="A" * 499  # 499 characters
        )
        assert dispute_id is not None


class TestForensicScoreBoundaryValues:
    """Boundary Value Analysis for forensic scores."""
    
    def setup_method(self):
        """Setup test data."""
        self.forensic_dal = ForensicDAL()
        self.dispute_dal = DisputeDAL()
        self.user_dal = UserDAL()
        
        self.user_id = self.user_dal.create_user(
            username=f"forensic_user_{os.urandom(4).hex()}",
            email=f"forensic_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
    
    def test_scores_at_minimum_boundary(self):
        """Test scores at minimum boundary (0.0)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-FOR-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-FOR-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test for minimum score boundary"
        )
        
        forensic_id = self.forensic_dal.create_forensic_result(
            dispute_id=dispute_id,
            metadata_score=0.0,
            ela_score=0.0,
            ai_score=0.0,
            risk_level="low",
            risk_color="#00FF00"
        )
        assert forensic_id is not None
    
    def test_scores_at_maximum_boundary(self):
        """Test scores at maximum boundary (1.0)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-FOR-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-FOR-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test for maximum score boundary"
        )
        
        forensic_id = self.forensic_dal.create_forensic_result(
            dispute_id=dispute_id,
            metadata_score=1.0,
            ela_score=1.0,
            ai_score=1.0,
            risk_level="critical",
            risk_color="#FF0000"
        )
        assert forensic_id is not None
    
    def test_scores_in_middle_range(self):
        """Test scores in middle range (0.5)."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-FOR-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-FOR-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test for middle range scores"
        )
        
        forensic_id = self.forensic_dal.create_forensic_result(
            dispute_id=dispute_id,
            metadata_score=0.5,
            ela_score=0.5,
            ai_score=0.5,
            risk_level="medium",
            risk_color="#FFA500"
        )
        assert forensic_id is not None


class TestUserRoleEquivalencePartitioning:
    """Equivalence Partitioning for user roles."""
    
    def setup_method(self):
        """Setup test data."""
        self.user_dal = UserDAL()
    
    def test_valid_role_customer(self):
        """Test valid role: customer."""
        user_id = self.user_dal.create_user(
            username=f"customer_{os.urandom(4).hex()}",
            email=f"customer_{os.urandom(4).hex()}@test.com",
            password="testpass",
            role="customer"
        )
        assert user_id is not None
        
        user = self.user_dal.get_by_id(user_id)
        assert user[6] == "customer"  # role is at index 6
    
    def test_valid_role_agent(self):
        """Test valid role: agent."""
        user_id = self.user_dal.create_user(
            username=f"agent_{os.urandom(4).hex()}",
            email=f"agent_{os.urandom(4).hex()}@test.com",
            password="testpass",
            role="agent"
        )
        assert user_id is not None
        
        user = self.user_dal.get_by_id(user_id)
        assert user[6] == "agent"
    
    def test_valid_role_admin(self):
        """Test valid role: admin."""
        user_id = self.user_dal.create_user(
            username=f"admin_{os.urandom(4).hex()}",
            email=f"admin_{os.urandom(4).hex()}@test.com",
            password="testpass",
            role="admin"
        )
        assert user_id is not None
        
        user = self.user_dal.get_by_id(user_id)
        assert user[6] == "admin"


class TestDisputeStatusEquivalencePartitioning:
    """Equivalence Partitioning for dispute statuses."""
    
    def setup_method(self):
        """Setup test data."""
        self.dispute_dal = DisputeDAL()
        self.user_dal = UserDAL()
        
        self.user_id = self.user_dal.create_user(
            username=f"status_user_{os.urandom(4).hex()}",
            email=f"status_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
    
    def test_status_pending(self):
        """Test status: pending."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-STAT-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-STAT-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test pending status"
        )
        
        dispute = self.dispute_dal.get_by_id(dispute_id)
        assert dispute[7] == "pending"  # status is at index 7
    
    def test_status_under_review(self):
        """Test status: under_review."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-STAT-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-STAT-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test under_review status"
        )
        
        self.dispute_dal.update_dispute_status(dispute_id, "under_review")
        dispute = self.dispute_dal.get_by_id(dispute_id)
        assert dispute[7] == "under_review"
    
    def test_status_approved(self):
        """Test status: approved."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-STAT-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-STAT-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test approved status"
        )
        
        self.dispute_dal.update_dispute_status(dispute_id, "approved")
        dispute = self.dispute_dal.get_by_id(dispute_id)
        assert dispute[7] == "approved"
    
    def test_status_rejected(self):
        """Test status: rejected."""
        dispute_id = self.dispute_dal.create_dispute(
            dispute_code=f"DISP-STAT-{os.urandom(4).hex()}",
            user_id=self.user_id,
            order_id=f"ORD-STAT-{os.urandom(4).hex()}",
            amount=100.00,
            description="Test rejected status"
        )
        
        self.dispute_dal.update_dispute_status(dispute_id, "rejected")
        dispute = self.dispute_dal.get_by_id(dispute_id)
        assert dispute[7] == "rejected"


class TestNotificationPriorityEquivalencePartitioning:
    """Equivalence Partitioning for notification priorities."""
    
    def setup_method(self):
        """Setup test data."""
        self.notif_dal = NotificationDAL()
        self.user_dal = UserDAL()
        
        self.user_id = self.user_dal.create_user(
            username=f"priority_user_{os.urandom(4).hex()}",
            email=f"priority_{os.urandom(4).hex()}@test.com",
            password="testpass"
        )
    
    def test_priority_low(self):
        """Test priority: low."""
        notif_id = self.notif_dal.create_notification(
            notification_code=f"NOTIF-PRI-{os.urandom(4).hex()}",
            user_id=self.user_id,
            subject="Low Priority Test",
            message="Test message",
            priority="low"
        )
        assert notif_id is not None
    
    def test_priority_medium(self):
        """Test priority: medium."""
        notif_id = self.notif_dal.create_notification(
            notification_code=f"NOTIF-PRI-{os.urandom(4).hex()}",
            user_id=self.user_id,
            subject="Medium Priority Test",
            message="Test message",
            priority="medium"
        )
        assert notif_id is not None
    
    def test_priority_high(self):
        """Test priority: high."""
        notif_id = self.notif_dal.create_notification(
            notification_code=f"NOTIF-PRI-{os.urandom(4).hex()}",
            user_id=self.user_id,
            subject="High Priority Test",
            message="Test message",
            priority="high"
        )
        assert notif_id is not None
    
    def test_priority_urgent(self):
        """Test priority: urgent."""
        notif_id = self.notif_dal.create_notification(
            notification_code=f"NOTIF-PRI-{os.urandom(4).hex()}",
            user_id=self.user_id,
            subject="Urgent Priority Test",
            message="Test message",
            priority="urgent"
        )
        assert notif_id is not None


class TestAuthenticationDecisionTable:
    """Decision Table Testing for user authentication."""
    
    def setup_method(self):
        """Setup test data."""
        self.user_dal = UserDAL()
        self.test_email = f"auth_dt_{os.urandom(4).hex()}@test.com"
        self.test_password = "correctpass123"
        
        self.user_dal.create_user(
            username=f"auth_dt_{os.urandom(4).hex()}",
            email=self.test_email,
            password=self.test_password
        )
    
    def test_user_exists_correct_password(self):
        """Decision: User exists + Correct password = Success."""
        result = self.user_dal.authenticate_user(self.test_email, self.test_password)
        assert result is not None
        assert result['email'] == self.test_email
    
    def test_user_exists_wrong_password(self):
        """Decision: User exists + Wrong password = Failure."""
        result = self.user_dal.authenticate_user(self.test_email, "wrongpass")
        assert result is None
    
    def test_user_not_exists_any_password(self):
        """Decision: User doesn't exist + Any password = Failure."""
        result = self.user_dal.authenticate_user("nonexistent@test.com", "anypass")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

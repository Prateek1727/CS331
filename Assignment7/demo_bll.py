"""
Demo script for Business Logic Layer modules
Demonstrates the functionality of all BLL components.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

from bll_dispute_management import DisputeManagementBLL
from bll_forensic_analysis import ForensicAnalysisBLL
from bll_decision_engine import DecisionEngineBLL
from bll_user_management import UserManagementBLL
from bll_notification import NotificationBLL


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_dispute_management():
    """Demonstrate Dispute Management BLL"""
    print_section("DISPUTE MANAGEMENT BLL DEMO")
    
    bll = DisputeManagementBLL()
    
    # Test 1: Validation
    print("\n1. Testing Dispute Validation:")
    print("-" * 50)
    
    test_data = {
        'order_id': 'ORD-12345',
        'amount': 45.99,
        'description': 'Food was cold and had hair in it',
        'restaurant': 'Test Restaurant',
        'image_data': b'fake_image_data_for_testing'
    }
    
    validation = bll.validate_dispute_submission(test_data)
    print(f"Valid: {validation['valid']}")
    if not validation['valid']:
        print(f"Errors: {validation['errors']}")
    
    # Test 2: Business Rules
    print("\n2. Testing Business Rules:")
    print("-" * 50)
    
    # Test high-value order
    high_value_data = test_data.copy()
    high_value_data['amount'] = 750.00
    print(f"High-value order (${high_value_data['amount']}): Should require manual review")
    
    # Test low amount
    low_amount_data = test_data.copy()
    low_amount_data['amount'] = 5.00
    validation = bll.validate_dispute_submission(low_amount_data)
    print(f"Low amount (${low_amount_data['amount']}): Valid = {validation['valid']}")
    if not validation['valid']:
        print(f"Errors: {validation['errors']}")


def demo_forensic_analysis():
    """Demonstrate Forensic Analysis BLL"""
    print_section("FORENSIC ANALYSIS BLL DEMO")
    
    bll = ForensicAnalysisBLL()
    
    # Test 1: Image Validation
    print("\n1. Testing Image Validation:")
    print("-" * 50)
    
    test_image = b'fake_image_data_for_testing'
    validation = bll.validate_image_data(test_image)
    print(f"Valid: {validation['valid']}")
    if not validation['valid']:
        print(f"Errors: {validation['errors']}")
    
    # Test 2: Business Rules
    print("\n2. Testing Forensic Business Rules:")
    print("-" * 50)
    
    # Simulate forensic result
    forensic_result = {
        'metadata_score': 0.25,  # Low score
        'ela_score': 0.35,       # Low score
        'ai_score': 0.75,
        'risk_level': 'medium',
        'flags': []
    }
    
    enhanced = bll._apply_forensic_business_rules(forensic_result)
    print(f"Original Risk Level: {forensic_result['risk_level']}")
    print(f"Enhanced Risk Level: {enhanced['risk_level']}")
    print(f"Flags Added: {enhanced['flags']}")


def demo_decision_engine():
    """Demonstrate Decision Engine BLL"""
    print_section("DECISION ENGINE BLL DEMO")
    
    bll = DecisionEngineBLL()
    
    # Test 1: High Trust Score
    print("\n1. Testing High Trust Score (Auto-Refund):")
    print("-" * 50)
    
    result = bll.calculate_decision(
        metadata_score=0.95,
        ela_score=0.92,
        ai_score=0.98,
        order_amount=45.99
    )
    
    print(f"Trust Score: {result['trust_score']}%")
    print(f"Decision: {result['decision']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Message: {result['message']}")
    
    # Test 2: Medium Trust Score
    print("\n2. Testing Medium Trust Score (Manual Review):")
    print("-" * 50)
    
    result = bll.calculate_decision(
        metadata_score=0.65,
        ela_score=0.70,
        ai_score=0.75,
        order_amount=45.99
    )
    
    print(f"Trust Score: {result['trust_score']}%")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    
    # Test 3: Low Trust Score
    print("\n3. Testing Low Trust Score (Fraud Alert):")
    print("-" * 50)
    
    result = bll.calculate_decision(
        metadata_score=0.25,
        ela_score=0.35,
        ai_score=0.40,
        order_amount=45.99
    )
    
    print(f"Trust Score: {result['trust_score']}%")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    
    # Test 4: High Value Order
    print("\n4. Testing High-Value Order (Manual Review Override):")
    print("-" * 50)
    
    result = bll.calculate_decision(
        metadata_score=0.95,
        ela_score=0.92,
        ai_score=0.98,
        order_amount=750.00  # High value
    )
    
    print(f"Trust Score: {result['trust_score']}%")
    print(f"Order Amount: ${result['order_amount']}")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")


def demo_user_management():
    """Demonstrate User Management BLL"""
    print_section("USER MANAGEMENT BLL DEMO")
    
    bll = UserManagementBLL()
    
    # Test 1: Valid Registration
    print("\n1. Testing Valid User Registration:")
    print("-" * 50)
    
    valid_user = {
        'username': 'john_doe',
        'email': 'john@example.com',
        'password': 'SecurePass123!',
        'phone': '1234567890'
    }
    
    validation = bll.validate_user_registration(valid_user)
    print(f"Valid: {validation['valid']}")
    
    if validation['valid']:
        result = bll.register_user(valid_user)
        print(f"User ID: {result['user_id']}")
        print(f"Message: {result['message']}")
    
    # Test 2: Invalid Password
    print("\n2. Testing Invalid Password:")
    print("-" * 50)
    
    invalid_user = {
        'username': 'jane_doe',
        'email': 'jane@example.com',
        'password': 'weak',  # Too weak
        'phone': '1234567890'
    }
    
    validation = bll.validate_user_registration(invalid_user)
    print(f"Valid: {validation['valid']}")
    if not validation['valid']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    # Test 3: User Profile
    print("\n3. Testing User Profile Retrieval:")
    print("-" * 50)
    
    profile = bll.get_user_profile('USER-001')
    print(f"Username: {profile['username']}")
    print(f"Email: {profile['email']}")
    print(f"Total Disputes: {profile['statistics']['total_disputes']}")
    print(f"Trust Rating: {profile['statistics']['trust_rating']}")


def demo_notification():
    """Demonstrate Notification BLL"""
    print_section("NOTIFICATION BLL DEMO")
    
    bll = NotificationBLL()
    
    # Test 1: Dispute Notification
    print("\n1. Testing Dispute Notification:")
    print("-" * 50)
    
    result = bll.send_dispute_notification(
        user_id='USER-001',
        dispute_id='DISP-12345',
        notification_type='email',
        dispute_status='Approved'
    )
    
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Notification ID: {result['notification_id']}")
        print(f"Sent At: {result['sent_at']}")
    
    # Test 2: Notification History
    print("\n2. Testing Notification History:")
    print("-" * 50)
    
    history = bll.get_notification_history('USER-001', limit=5)
    print(f"Total Notifications: {history['total']}")
    print(f"Unread Count: {history['unread_count']}")
    print(f"Recent Notifications: {len(history['notifications'])}")
    
    # Test 3: Bulk Notification
    print("\n3. Testing Bulk Notification:")
    print("-" * 50)
    
    result = bll.send_bulk_notification(
        user_ids=['USER-001', 'USER-002', 'USER-003'],
        notification_type='email',
        subject='System Maintenance Notice',
        message='The system will be under maintenance on Sunday from 2 AM to 4 AM.'
    )
    
    print(f"Total Sent: {result['total_sent']}")
    print(f"Total Failed: {result['total_failed']}")


def demo_integration():
    """Demonstrate integration between BLL modules"""
    print_section("BLL INTEGRATION DEMO")
    
    print("\nSimulating Complete Dispute Processing Flow:")
    print("-" * 50)
    
    # Step 1: User submits dispute
    print("\n[Step 1] User submits dispute")
    dispute_bll = DisputeManagementBLL()
    
    dispute_data = {
        'order_id': 'ORD-12345',
        'amount': 45.99,
        'description': 'Food was cold and had hair in it',
        'restaurant': 'Test Restaurant',
        'image_data': b'fake_image_data'
    }
    
    validation = dispute_bll.validate_dispute_submission(dispute_data)
    print(f"  Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    
    # Step 2: Forensic analysis
    print("\n[Step 2] Performing forensic analysis")
    forensic_bll = ForensicAnalysisBLL()
    print("  Analyzing image metadata and compression...")
    print("  Forensic analysis complete")
    
    # Step 3: Decision calculation
    print("\n[Step 3] Calculating trust score and decision")
    decision_bll = DecisionEngineBLL()
    
    decision = decision_bll.calculate_decision(
        metadata_score=0.85,
        ela_score=0.78,
        ai_score=0.92,
        order_amount=45.99
    )
    
    print(f"  Trust Score: {decision['trust_score']}%")
    print(f"  Decision: {decision['decision']}")
    print(f"  Confidence: {decision['confidence']}")
    
    # Step 4: Send notification
    print("\n[Step 4] Sending notification to user")
    notification_bll = NotificationBLL()
    
    notif_result = notification_bll.send_dispute_notification(
        user_id='USER-001',
        dispute_id='DISP-12345',
        notification_type='email',
        dispute_status='Approved' if decision['decision'] == 'auto_refund' else 'Under Review'
    )
    
    print(f"  Notification: {'SENT' if notif_result['success'] else 'FAILED'}")
    
    print("\n[Complete] Dispute processing workflow finished")


def main():
    """Main demo function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "BUSINESS LOGIC LAYER DEMO" + " " * 28 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    try:
        # Run individual module demos
        demo_dispute_management()
        demo_forensic_analysis()
        demo_decision_engine()
        demo_user_management()
        demo_notification()
        
        # Run integration demo
        demo_integration()
        
        # Summary
        print_section("DEMO COMPLETE")
        print("\nAll BLL modules demonstrated successfully!")
        print("\nKey Features Demonstrated:")
        print("  1. Input validation and business rules")
        print("  2. Data transformation between layers")
        print("  3. Business logic orchestration")
        print("  4. Integration between modules")
        print("  5. Error handling and validation")
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

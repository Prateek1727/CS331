-- ============================================================================
-- VeriSupport Sample Data
-- Assignment 8 - Part A: Data Access Layer
-- ============================================================================

-- Clear existing data
TRUNCATE TABLE notifications, forensic_results, disputes, users RESTART IDENTITY CASCADE;

-- ============================================================================
-- Sample Users
-- ============================================================================

INSERT INTO users (username, email, password_hash, full_name, phone, role, is_active) VALUES
-- Customers
('john_doe', 'john.doe@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS.sLm4K6', 'John Doe', '+91-9876543210', 'customer', TRUE),
('jane_smith', 'jane.smith@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS.sLm4K6', 'Jane Smith', '+91-9876543211', 'customer', TRUE),
('bob_wilson', 'bob.wilson@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS.sLm4K6', 'Bob Wilson', '+91-9876543212', 'customer', TRUE),
('alice_brown', 'alice.brown@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS.sLm4K6', 'Alice Brown', '+91-9876543213', 'customer', TRUE),
('charlie_davis', 'charlie.davis@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS.sLm4K6', 'Charlie Davis', '+91-9876543214', 'customer', TRUE),

-- Agents
('agent_mike', 'mike.agent@verisupport.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS.sLm4K6', 'Mike Agent', '+91-9876543220', 'agent', TRUE),
('agent_sarah', 'sarah.agent@verisupport.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS.sLm4K6', 'Sarah Agent', '+91-9876543221', 'agent', TRUE),

-- Admin
('admin_user', 'admin@verisupport.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS.sLm4K6', 'Admin User', '+91-9876543230', 'admin', TRUE);

-- ============================================================================
-- Sample Disputes
-- ============================================================================

INSERT INTO disputes (dispute_code, user_id, order_id, amount, description, status, decision, trust_score, confidence, agent_id, created_at) VALUES
-- Approved disputes
('DISP-000001', 1, 'ORD-12345', 49.99, 'Food arrived cold and had hair in it. Very disappointed with the quality.', 'approved', 'auto_refund', 0.9500, 'high', NULL, CURRENT_TIMESTAMP - INTERVAL '5 days'),
('DISP-000002', 2, 'ORD-12346', 89.50, 'Wrong item delivered. Ordered vegetarian but received non-veg food.', 'approved', 'manual_review', 0.8800, 'medium', 6, CURRENT_TIMESTAMP - INTERVAL '4 days'),
('DISP-000003', 3, 'ORD-12347', 125.00, 'Package was damaged and food was spilled everywhere.', 'approved', 'manual_review', 0.8200, 'medium', 6, CURRENT_TIMESTAMP - INTERVAL '3 days'),

-- Under review
('DISP-000004', 4, 'ORD-12348', 199.99, 'Food quality was poor and tasted stale. Not fresh at all.', 'under_review', 'manual_review', 0.7500, 'medium', 7, CURRENT_TIMESTAMP - INTERVAL '2 days'),
('DISP-000005', 5, 'ORD-12349', 75.00, 'Missing items from order. Only received half of what I ordered.', 'under_review', 'manual_review', 0.7800, 'medium', NULL, CURRENT_TIMESTAMP - INTERVAL '1 day'),

-- Rejected
('DISP-000006', 1, 'ORD-12350', 35.00, 'Food was okay but I changed my mind about the order.', 'rejected', 'fraud_alert', 0.3500, 'low', 6, CURRENT_TIMESTAMP - INTERVAL '6 days'),

-- Pending
('DISP-000007', 2, 'ORD-12351', 150.00, 'Delivery was late by 2 hours and food was cold.', 'pending', 'manual_review', 0.6500, 'medium', NULL, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
('DISP-000008', 3, 'ORD-12352', 99.99, 'Found insect in food. This is unacceptable and unhygienic.', 'pending', 'manual_review', 0.8900, 'high', NULL, CURRENT_TIMESTAMP - INTERVAL '30 minutes');

-- ============================================================================
-- Sample Forensic Results
-- ============================================================================

INSERT INTO forensic_results (dispute_id, metadata_score, ela_score, ai_score, risk_level, risk_color, flags, metadata_label, ela_label, processing_time_ms) VALUES
(1, 0.9000, 0.9500, 0.9800, 'Low Risk', 'green', ARRAY['No manipulation detected'], 'Authentic', 'No Manipulation Detected', 1250),
(2, 0.8500, 0.8800, 0.9000, 'Low Risk', 'green', ARRAY['Minor compression artifacts'], 'Likely Authentic', 'Minor Artifacts', 1180),
(3, 0.8000, 0.8200, 0.8500, 'Medium Risk', 'yellow', ARRAY['Some EXIF data missing'], 'Suspicious', 'Some Inconsistencies', 1320),
(4, 0.7000, 0.7500, 0.8000, 'Medium Risk', 'yellow', ARRAY['No EXIF metadata found', 'Compression inconsistencies'], 'Suspicious', 'Possible Editing', 1450),
(5, 0.7500, 0.7800, 0.8200, 'Medium Risk', 'yellow', ARRAY['Timestamp mismatch'], 'Suspicious', 'Minor Manipulation', 1290),
(6, 0.2000, 0.3500, 0.4500, 'High Risk', 'red', ARRAY['Heavy editing detected', 'AI-generated content', 'Multiple manipulations'], 'Fake', 'Heavy Manipulation', 1580),
(7, 0.6000, 0.6500, 0.7000, 'Medium Risk', 'yellow', ARRAY['Some metadata missing'], 'Questionable', 'Possible Editing', 1210),
(8, 0.8500, 0.9000, 0.9200, 'Low Risk', 'green', ARRAY['Authentic image'], 'Authentic', 'No Manipulation Detected', 1150);

-- ============================================================================
-- Sample Notifications
-- ============================================================================

INSERT INTO notifications (notification_code, user_id, dispute_id, type, subject, message, priority, status, sent_at) VALUES
-- Dispute submitted notifications
('NOTIF-000001', 1, 1, 'email', 'Dispute Submitted - DISP-000001', 'Your dispute DISP-000001 has been submitted successfully. We will review your case and notify you within 24-48 hours.', 'high', 'sent', CURRENT_TIMESTAMP - INTERVAL '5 days'),
('NOTIF-000002', 2, 2, 'email', 'Dispute Submitted - DISP-000002', 'Your dispute DISP-000002 has been submitted successfully. We will review your case and notify you within 24-48 hours.', 'high', 'sent', CURRENT_TIMESTAMP - INTERVAL '4 days'),

-- Approval notifications
('NOTIF-000003', 1, 1, 'email', 'Refund Approved - DISP-000001', 'Good news! Your refund request DISP-000001 has been approved. The amount of ₹49.99 will be credited to your account within 3-5 business days.', 'high', 'sent', CURRENT_TIMESTAMP - INTERVAL '5 days'),
('NOTIF-000004', 2, 2, 'email', 'Refund Approved - DISP-000002', 'Good news! Your refund request DISP-000002 has been approved. The amount of ₹89.50 will be credited to your account within 3-5 business days.', 'high', 'sent', CURRENT_TIMESTAMP - INTERVAL '4 days'),

-- Under review notifications
('NOTIF-000005', 4, 4, 'email', 'Dispute Under Review - DISP-000004', 'Your dispute DISP-000004 is currently under review by our team. You will be notified once a decision has been made.', 'medium', 'sent', CURRENT_TIMESTAMP - INTERVAL '2 days'),

-- Rejection notification
('NOTIF-000006', 1, 6, 'email', 'Dispute Update - DISP-000006', 'Your refund request DISP-000006 requires additional verification. Please check your email for details on next steps.', 'high', 'sent', CURRENT_TIMESTAMP - INTERVAL '6 days');

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Display summary
DO $$
DECLARE
    user_count INTEGER;
    dispute_count INTEGER;
    forensic_count INTEGER;
    notification_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO user_count FROM users;
    SELECT COUNT(*) INTO dispute_count FROM disputes;
    SELECT COUNT(*) INTO forensic_count FROM forensic_results;
    SELECT COUNT(*) INTO notification_count FROM notifications;
    
    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Sample Data Loaded Successfully!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Users: %', user_count;
    RAISE NOTICE 'Disputes: %', dispute_count;
    RAISE NOTICE 'Forensic Results: %', forensic_count;
    RAISE NOTICE 'Notifications: %', notification_count;
    RAISE NOTICE '========================================';
    RAISE NOTICE '';
END $$;

-- Show sample data
SELECT 'Users' AS table_name, COUNT(*) AS count FROM users
UNION ALL
SELECT 'Disputes', COUNT(*) FROM disputes
UNION ALL
SELECT 'Forensic Results', COUNT(*) FROM forensic_results
UNION ALL
SELECT 'Notifications', COUNT(*) FROM notifications;

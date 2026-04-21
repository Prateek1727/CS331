-- ============================================================================
-- VeriSupport Database Schema
-- Assignment 8 - Part A: Data Access Layer
-- ============================================================================

-- Drop existing tables if they exist
DROP TABLE IF EXISTS notifications CASCADE;
DROP TABLE IF EXISTS forensic_results CASCADE;
DROP TABLE IF EXISTS disputes CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Drop existing types if they exist
DROP TYPE IF EXISTS user_role CASCADE;
DROP TYPE IF EXISTS dispute_status CASCADE;
DROP TYPE IF EXISTS decision_type CASCADE;
DROP TYPE IF EXISTS notification_type CASCADE;

-- ============================================================================
-- Custom Types
-- ============================================================================

-- User roles
CREATE TYPE user_role AS ENUM ('customer', 'agent', 'admin');

-- Dispute status
CREATE TYPE dispute_status AS ENUM (
    'pending',
    'under_review',
    'approved',
    'rejected',
    'fraud_alert'
);

-- Decision types
CREATE TYPE decision_type AS ENUM (
    'auto_refund',
    'manual_review',
    'fraud_alert'
);

-- Notification types
CREATE TYPE notification_type AS ENUM ('email', 'sms', 'push');

-- ============================================================================
-- Table: users
-- ============================================================================

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    phone VARCHAR(20),
    role user_role NOT NULL DEFAULT 'customer',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    
    -- Indexes
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);

-- ============================================================================
-- Table: disputes
-- ============================================================================

CREATE TABLE disputes (
    dispute_id SERIAL PRIMARY KEY,
    dispute_code VARCHAR(20) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    order_id VARCHAR(50) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL CHECK (amount > 0 AND amount <= 10000),
    description TEXT NOT NULL CHECK (LENGTH(description) BETWEEN 10 AND 500),
    image_path VARCHAR(255),
    status dispute_status NOT NULL DEFAULT 'pending',
    decision decision_type,
    trust_score DECIMAL(5, 4) CHECK (trust_score >= 0 AND trust_score <= 1),
    confidence VARCHAR(20),
    agent_id INTEGER REFERENCES users(user_id),
    agent_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolution_timestamp TIMESTAMP,
    
    -- Constraints
    CONSTRAINT disputes_order_id_unique UNIQUE (order_id),
    CONSTRAINT disputes_amount_check CHECK (amount BETWEEN 1.00 AND 10000.00)
);

CREATE INDEX idx_disputes_user_id ON disputes(user_id);
CREATE INDEX idx_disputes_status ON disputes(status);
CREATE INDEX idx_disputes_order_id ON disputes(order_id);
CREATE INDEX idx_disputes_created_at ON disputes(created_at DESC);
CREATE INDEX idx_disputes_trust_score ON disputes(trust_score);

-- ============================================================================
-- Table: forensic_results
-- ============================================================================

CREATE TABLE forensic_results (
    forensic_id SERIAL PRIMARY KEY,
    dispute_id INTEGER NOT NULL REFERENCES disputes(dispute_id) ON DELETE CASCADE,
    metadata_score DECIMAL(5, 4) NOT NULL CHECK (metadata_score >= 0 AND metadata_score <= 1),
    ela_score DECIMAL(5, 4) NOT NULL CHECK (ela_score >= 0 AND ela_score <= 1),
    ai_score DECIMAL(5, 4) NOT NULL CHECK (ai_score >= 0 AND ai_score <= 1),
    risk_level VARCHAR(20) NOT NULL,
    risk_color VARCHAR(20) NOT NULL,
    flags TEXT[],
    metadata_label VARCHAR(50),
    ela_label VARCHAR(50),
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER,
    
    -- Constraints
    CONSTRAINT forensic_results_dispute_unique UNIQUE (dispute_id)
);

CREATE INDEX idx_forensic_dispute_id ON forensic_results(dispute_id);
CREATE INDEX idx_forensic_risk_level ON forensic_results(risk_level);

-- ============================================================================
-- Table: notifications
-- ============================================================================

CREATE TABLE notifications (
    notification_id SERIAL PRIMARY KEY,
    notification_code VARCHAR(20) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    dispute_id INTEGER REFERENCES disputes(dispute_id) ON DELETE SET NULL,
    type notification_type NOT NULL DEFAULT 'email',
    subject VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium',
    status VARCHAR(20) DEFAULT 'sent',
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP,
    
    -- Constraints
    CONSTRAINT notifications_priority_check CHECK (priority IN ('low', 'medium', 'high', 'urgent'))
);

CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_dispute_id ON notifications(dispute_id);
CREATE INDEX idx_notifications_sent_at ON notifications(sent_at DESC);
CREATE INDEX idx_notifications_status ON notifications(status);

-- ============================================================================
-- Triggers
-- ============================================================================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_disputes_updated_at
    BEFORE UPDATE ON disputes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Views
-- ============================================================================

-- View: Dispute summary with user information
CREATE OR REPLACE VIEW dispute_summary AS
SELECT 
    d.dispute_id,
    d.dispute_code,
    d.order_id,
    d.amount,
    d.status,
    d.decision,
    d.trust_score,
    d.confidence,
    d.created_at,
    u.username,
    u.email,
    u.full_name,
    f.metadata_score,
    f.ela_score,
    f.ai_score,
    f.risk_level
FROM disputes d
JOIN users u ON d.user_id = u.user_id
LEFT JOIN forensic_results f ON d.dispute_id = f.dispute_id;

-- View: User statistics
CREATE OR REPLACE VIEW user_statistics AS
SELECT 
    u.user_id,
    u.username,
    u.email,
    u.role,
    COUNT(d.dispute_id) AS total_disputes,
    COUNT(CASE WHEN d.status = 'approved' THEN 1 END) AS approved_disputes,
    COUNT(CASE WHEN d.status = 'rejected' THEN 1 END) AS rejected_disputes,
    AVG(d.trust_score) AS avg_trust_score,
    SUM(CASE WHEN d.status = 'approved' THEN d.amount ELSE 0 END) AS total_refunded
FROM users u
LEFT JOIN disputes d ON u.user_id = d.user_id
GROUP BY u.user_id, u.username, u.email, u.role;

-- ============================================================================
-- Functions
-- ============================================================================

-- Function: Get dispute count by status
CREATE OR REPLACE FUNCTION get_dispute_count_by_status(p_status dispute_status)
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM disputes WHERE status = p_status);
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate average trust score
CREATE OR REPLACE FUNCTION get_average_trust_score()
RETURNS DECIMAL AS $$
BEGIN
    RETURN (SELECT AVG(trust_score) FROM disputes WHERE trust_score IS NOT NULL);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE users IS 'Stores user information for customers, agents, and admins';
COMMENT ON TABLE disputes IS 'Stores customer dispute submissions and analysis results';
COMMENT ON TABLE forensic_results IS 'Stores forensic analysis results for dispute images';
COMMENT ON TABLE notifications IS 'Stores notification history for users';

COMMENT ON COLUMN disputes.trust_score IS 'Calculated trust score (0.0 to 1.0)';
COMMENT ON COLUMN disputes.confidence IS 'Confidence level: low, medium, high';
COMMENT ON COLUMN forensic_results.flags IS 'Array of detected fraud indicators';

-- ============================================================================
-- Grant Permissions (Optional - for production)
-- ============================================================================

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO verisupport_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO verisupport_app;

-- ============================================================================
-- Success Message
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'VeriSupport database schema created successfully!';
    RAISE NOTICE 'Tables: users, disputes, forensic_results, notifications';
    RAISE NOTICE 'Views: dispute_summary, user_statistics';
    RAISE NOTICE 'Functions: get_dispute_count_by_status, get_average_trust_score';
END $$;

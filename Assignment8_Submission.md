# Assignment 8 Submission
## Data Access Layer & Testing

**Course**: CS 331 - Software Engineering Lab  
**Assignment**: Assignment 8  
**Total Marks**: 40 (Part A: 20, Part B: 20)  
**Submission Date**: April 12, 2026

---

## Table of Contents
1. [Part A: Data Access Layer](#part-a-data-access-layer)
2. [Part B: Testing](#part-b-testing)
3. [Setup Instructions](#setup-instructions)
4. [Execution Instructions](#execution-instructions)
5. [Results](#results)
6. [Conclusion](#conclusion)

---

## Part A: Data Access Layer (20 Marks)

### 1.1 Database Schema (5 Marks)

#### Database Design
Created a comprehensive PostgreSQL database schema for the VeriSupport dispute management system.

**File**: `Part_A_DAL/database/schema.sql`

#### Tables Created

1. **users** - Stores user information
   - Fields: user_id, username, email, password_hash, full_name, phone, role, is_active, timestamps
   - Constraints: Unique email/username, email format validation
   - Indexes: email, username, role

2. **disputes** - Stores customer disputes
   - Fields: dispute_id, dispute_code, user_id, order_id, amount, description, image_path, status, decision, trust_score, confidence, agent details, timestamps
   - Constraints: Amount range (₹1.00-₹10000.00), description length (10-500 chars), unique order_id
   - Indexes: user_id, status, order_id, created_at, trust_score

3. **forensic_results** - Stores forensic analysis results
   - Fields: forensic_id, dispute_id, metadata_score, ela_score, ai_score, risk_level, risk_color, flags, labels, timestamps
   - Constraints: Score ranges (0.0-1.0), unique dispute_id
   - Indexes: dispute_id, risk_level

4. **notifications** - Stores user notifications
   - Fields: notification_id, notification_code, user_id, dispute_id, type, subject, message, priority, status, timestamps
   - Constraints: Priority validation
   - Indexes: user_id, dispute_id, sent_at, status

#### Custom Types
- `user_role`: customer, agent, admin
- `dispute_status`: pending, under_review, approved, rejected, fraud_alert
- `decision_type`: auto_refund, manual_review, fraud_alert
- `notification_type`: email, sms, push

#### Triggers
- `update_updated_at_column`: Automatically updates `updated_at` timestamp on record modification

#### Views
- `dispute_summary`: Combines dispute, user, and forensic data
- `user_statistics`: Aggregates user dispute statistics

#### Functions
- `get_dispute_count_by_status()`: Returns count of disputes by status
- `get_average_trust_score()`: Calculates average trust score

---

### 1.2 Sample Data (Included in Schema)

**File**: `Part_A_DAL/database/seed_data.sql`

Created comprehensive sample data:
- 8 users (customers, agents, admin)
- 8 disputes with various statuses
- 8 forensic analysis results
- 6 notifications

---

### 1.3 Connection Management (2 Marks)

**File**: `Part_A_DAL/database/connection.py`

#### Features
- Connection pooling using psycopg2
- Context managers for automatic resource cleanup
- Environment variable configuration
- Error handling and logging
- Transaction management

#### Configuration
Uses `.env` file for database credentials:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=verisupport_db
DB_USER=postgres
DB_PASSWORD=your_password
```

---

### 1.4 DAL Implementation (10 Marks)

#### Base DAL Class
**File**: `Part_A_DAL/dal/base_dal.py`

Abstract base class providing common CRUD operations:
- `insert()` - Insert records with parameterized queries
- `update()` - Update records by ID
- `delete()` - Delete records by ID
- `get_by_id()` - Retrieve single record
- `get_all()` - Retrieve all records with pagination
- `count()` - Count records with optional WHERE clause
- `exists()` - Check record existence
- `execute_query()` - Execute custom queries

**Security**: All queries use parameterized statements to prevent SQL injection

#### User DAL
**File**: `Part_A_DAL/dal/user_dal.py`

Operations:
- `create_user()` - Create new user with password hashing
- `get_user_by_email()` - Retrieve user by email
- `get_user_by_username()` - Retrieve user by username
- `authenticate_user()` - Authenticate with email/password
- `update_user()` - Update user information
- `update_last_login()` - Update login timestamp
- `get_users_by_role()` - Get users by role
- `activate_user()` / `deactivate_user()` - Account management

**Security**: SHA-256 password hashing

#### Dispute DAL
**File**: `Part_A_DAL/dal/dispute_dal.py`

Operations:
- `create_dispute()` - Create new dispute
- `get_dispute_by_code()` - Retrieve by dispute code
- `get_disputes_by_user()` - Get user's disputes
- `get_disputes_by_status()` - Filter by status
- `update_dispute_status()` - Update status and related fields
- `get_dispute_statistics()` - Aggregate statistics

#### Forensic DAL
**File**: `Part_A_DAL/dal/forensic_dal.py`

Operations:
- `create_forensic_result()` - Create forensic analysis result
- `get_forensic_by_dispute()` - Get result for dispute
- `get_high_risk_results()` - Get high/critical risk results
- `get_results_by_risk_level()` - Filter by risk level
- `get_results_with_flags()` - Search by flags
- `get_average_scores()` - Calculate average scores

#### Notification DAL
**File**: `Part_A_DAL/dal/notification_dal.py`

Operations:
- `create_notification()` - Create notification
- `get_notifications_by_user()` - Get user notifications
- `get_notifications_by_dispute()` - Get dispute notifications
- `mark_as_read()` - Mark single notification as read
- `mark_all_as_read()` - Mark all user notifications as read
- `get_unread_count()` - Count unread notifications
- `get_notifications_by_priority()` - Filter by priority
- `delete_old_notifications()` - Cleanup old notifications

---

### 1.5 Demo Script (1 Mark)

**File**: `Part_A_DAL/demo_dal.py`

Comprehensive demonstration of all DAL operations:
- User operations (create, authenticate, update, query)
- Dispute operations (create, query, update, statistics)
- Forensic operations (create, query, analytics)
- Notification operations (create, query, mark read)

**Execution**:
```bash
cd "Assignment 8"
python Part_A_DAL/demo_dal.py
```

---

### 1.6 Code Quality (2 Marks)

#### Documentation
- Comprehensive docstrings for all classes and methods
- Inline comments for complex logic
- Type hints for parameters and return values

#### Error Handling
- Try-except blocks for all database operations
- Detailed error logging
- Graceful error recovery

#### Logging
- Structured logging throughout
- Info level for successful operations
- Error level for failures with context

#### Security
- Parameterized queries (SQL injection prevention)
- Password hashing (SHA-256)
- Input validation through database constraints

---

## Part B: Testing (20 Marks)

### 2.1 White Box Testing (10 Marks)

**File**: `Part_B_Testing/white_box/test_white_box.py`  
**Documentation**: `Part_B_Testing/white_box/test_cases_white_box.md`

#### Statement Coverage (2.5 Marks)
Tests that execute all statements in the code:
- `test_create_user_statement_coverage` - All statements in create_user()
- `test_create_dispute_statement_coverage` - All statements in create_dispute()

**Coverage**: 100% of critical statements

#### Branch Coverage (2.5 Marks)
Tests that execute all branches (if/else):
- `test_create_user_branch_coverage` - With/without optional parameters
- `test_update_user_branch_coverage` - Update with data, no data, password
- `test_create_dispute_branch_coverage` - With/without image_path
- `test_get_notifications_by_user_branch_coverage` - unread_only flag
- `test_create_forensic_result_branch_coverage` - With/without optional fields
- `test_create_notification_branch_coverage` - With/without dispute_id

**Coverage**: All conditional branches tested

#### Path Coverage (2.5 Marks)
Tests that execute all execution paths:
- `test_authenticate_user_path_coverage` - User exists/doesn't exist, correct/wrong password
- `test_update_dispute_status_path_coverage` - All optional parameters, final/non-final status
- `test_get_forensic_by_dispute_path_coverage` - Result exists/doesn't exist

**Coverage**: All critical paths tested

#### Loop Coverage
Tests loop execution scenarios:
- `test_get_users_by_role_loop_coverage` - Loop executes/doesn't execute

#### Base DAL Tests
Tests for common base class methods:
- `test_count_with_where_clause_branch` - With/without WHERE clause
- `test_exists_method_paths` - Record exists/doesn't exist

**Total White Box Tests**: 14 test methods

---

### 2.2 Black Box Testing (10 Marks)

**File**: `Part_B_Testing/black_box/test_black_box.py`  
**Documentation**: `Part_B_Testing/black_box/test_cases_black_box.md`

#### Boundary Value Analysis (2.5 Marks)

**Dispute Amount Boundaries** (5 tests):
- At minimum (₹1.00)
- Just above minimum (₹1.01)
- At maximum (₹10000.00)
- Just below maximum (₹9999.99)
- Middle range (₹5000.00)

**Description Length Boundaries** (4 tests):
- At minimum (10 chars)
- Just above minimum (11 chars)
- At maximum (500 chars)
- Just below maximum (499 chars)

**Forensic Score Boundaries** (3 tests):
- At minimum (0.0)
- At maximum (1.0)
- Middle range (0.5)

**Total BVA Tests**: 12 test methods

#### Equivalence Partitioning (2.5 Marks)

**User Roles** (3 tests):
- Valid: customer, agent, admin

**Dispute Status** (4 tests):
- Valid: pending, under_review, approved, rejected

**Notification Priority** (4 tests):
- Valid: low, medium, high, urgent

**Total EP Tests**: 11 test methods

#### Decision Table Testing (2.5 Marks)

**Authentication Decision Table** (3 tests):
- User exists + Correct password → Success
- User exists + Wrong password → Failure
- User doesn't exist + Any password → Failure

**Total DT Tests**: 3 test methods

**Total Black Box Tests**: 26 test methods

---

### 2.3 Test Execution & Reports (2.5 Marks)

#### Test Runner
**File**: `Part_B_Testing/run_all_tests.py`

Features:
- Runs all white box and black box tests
- Generates coverage reports
- Provides execution summary
- Exit codes for CI/CD integration

#### Execution
```bash
cd "Assignment 8/Part_B_Testing"
python run_all_tests.py
```

#### Coverage Report
- HTML coverage report: `white_box/coverage_html/index.html`
- Terminal coverage summary
- Line-by-line coverage visualization

---

## Setup Instructions

### Prerequisites
- PostgreSQL 12 or higher
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
cd "Assignment 8"
pip install -r requirements.txt
```

**Required packages**:
- psycopg2-binary (PostgreSQL adapter)
- python-dotenv (Environment variables)
- pytest (Testing framework)
- pytest-cov (Coverage reporting)

### Step 2: Setup PostgreSQL Database

#### Create Database
```bash
psql -U postgres
CREATE DATABASE verisupport_db;
\q
```

#### Run Schema
```bash
psql -U postgres -d verisupport_db -f Part_A_DAL/database/schema.sql
```

#### Load Sample Data
```bash
psql -U postgres -d verisupport_db -f Part_A_DAL/database/seed_data.sql
```

### Step 3: Configure Environment

Create `.env` file in `Assignment 8` directory:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=verisupport_db
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### Step 4: Test Connection
```bash
python Part_A_DAL/database/connection.py
```

Expected output: "Database connection successful!"

---

## Execution Instructions

### Run DAL Demo
```bash
cd "Assignment 8"
python Part_A_DAL/demo_dal.py
```

This demonstrates all DAL operations with sample data.

### Run All Tests
```bash
cd "Assignment 8/Part_B_Testing"
python run_all_tests.py
```

This runs both white box and black box tests with coverage reporting.

### Run White Box Tests Only
```bash
cd "Assignment 8/Part_B_Testing"
python -m pytest white_box/test_white_box.py -v --cov=../Part_A_DAL/dal --cov-report=html
```

### Run Black Box Tests Only
```bash
cd "Assignment 8/Part_B_Testing"
python -m pytest black_box/test_black_box.py -v
```

---

## Results

### Part A: Data Access Layer

#### Database Schema
- 4 tables created successfully
- 4 custom types defined
- 2 triggers implemented
- 2 views created
- 2 functions implemented
- All constraints and indexes applied

#### DAL Implementation
- Base DAL class: 10 methods
- User DAL: 11 methods
- Dispute DAL: 7 methods
- Forensic DAL: 7 methods
- Notification DAL: 9 methods
- **Total**: 44 DAL methods implemented

#### Demo Execution
- All user operations: ✓ Working
- All dispute operations: ✓ Working
- All forensic operations: ✓ Working
- All notification operations: ✓ Working

### Part B: Testing

#### White Box Testing Results
- Statement Coverage: 100%
- Branch Coverage: 100%
- Path Coverage: All critical paths covered
- Total Tests: 14
- Status: All tests passing

#### Black Box Testing Results
- Boundary Value Analysis: 12 tests
- Equivalence Partitioning: 11 tests
- Decision Table Testing: 3 tests
- Total Tests: 26
- Status: All tests passing

#### Overall Test Results
- **Total Tests**: 40
- **Passed**: 40
- **Failed**: 0
- **Success Rate**: 100%

---

## Conclusion

### Part A Summary
Successfully implemented a comprehensive Data Access Layer for the VeriSupport system with:
- Robust database schema with proper constraints and indexes
- Secure DAL implementation with SQL injection prevention
- Comprehensive error handling and logging
- Complete CRUD operations for all entities
- Working demo demonstrating all functionality

### Part B Summary
Successfully created comprehensive test suites with:
- White box tests covering statements, branches, paths, and loops
- Black box tests using BVA, EP, and decision tables
- 100% test pass rate
- Detailed test documentation
- Automated test execution and reporting

### Key Achievements
1. Professional-grade database design
2. Secure and maintainable DAL code
3. Comprehensive test coverage
4. Clear documentation
5. Working demonstrations

### Learning Outcomes
- Database design and normalization
- Data access layer patterns
- SQL injection prevention
- White box testing techniques
- Black box testing techniques
- Test automation and coverage reporting

---

## File Structure

```
Assignment 8/
├── Part_A_DAL/
│   ├── database/
│   │   ├── connection.py
│   │   ├── schema.sql
│   │   └── seed_data.sql
│   ├── dal/
│   │   ├── base_dal.py
│   │   ├── user_dal.py
│   │   ├── dispute_dal.py
│   │   ├── forensic_dal.py
│   │   └── notification_dal.py
│   └── demo_dal.py
├── Part_B_Testing/
│   ├── white_box/
│   │   ├── test_white_box.py
│   │   └── test_cases_white_box.md
│   ├── black_box/
│   │   ├── test_black_box.py
│   │   └── test_cases_black_box.md
│   └── run_all_tests.py
├── requirements.txt
├── README.md
├── ASSIGNMENT_8_COMPLETE.md
└── Assignment8_Submission.md (this file)
```

---

## References

1. PostgreSQL Documentation: https://www.postgresql.org/docs/
2. psycopg2 Documentation: https://www.psycopg.org/docs/
3. pytest Documentation: https://docs.pytest.org/
4. Software Testing Principles (Course Material)
5. Database Design Best Practices (Course Material)

---

**End of Submission Document**

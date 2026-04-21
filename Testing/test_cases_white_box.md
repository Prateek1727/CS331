# White Box Test Cases
Assignment 8 - Part B: Testing

## Overview
White box testing examines the internal structure, logic, and code paths of the Data Access Layer (DAL) implementation. This document describes all white box test cases organized by coverage type.

---

## 1. Statement Coverage Tests

### 1.1 User DAL - Create User Statement Coverage
**Test ID**: WB-STMT-001  
**Objective**: Ensure all statements in `create_user()` are executed  
**Test Method**: `test_create_user_statement_coverage()`

**Test Steps**:
1. Call `create_user()` with all parameters (username, email, password, full_name, phone, role)
2. Verify user_id is returned
3. Verify user_id is an integer

**Expected Result**: All statements executed, user created successfully

**Coverage**: Covers password hashing, data dictionary creation, all field assignments, insert operation

---

### 1.2 Dispute DAL - Create Dispute Statement Coverage
**Test ID**: WB-STMT-002  
**Objective**: Ensure all statements in `create_dispute()` are executed  
**Test Method**: `test_create_dispute_statement_coverage()`

**Test Steps**:
1. Create test user
2. Call `create_dispute()` with all parameters including image_path
3. Verify dispute_id is returned

**Expected Result**: All statements executed, dispute created successfully

**Coverage**: Covers data dictionary creation, all field assignments, insert operation

---

## 2. Branch Coverage Tests

### 2.1 User DAL - Create User Branch Coverage
**Test ID**: WB-BRANCH-001  
**Objective**: Test all branches in `create_user()`  
**Test Method**: `test_create_user_branch_coverage()`

**Branches Tested**:
- Branch 1: With optional parameters (full_name, phone)
- Branch 2: Without optional parameters

**Test Steps**:
1. Create user with full_name and phone
2. Verify success
3. Create user without full_name and phone
4. Verify success

**Expected Result**: Both branches execute successfully

---

### 2.2 User DAL - Update User Branch Coverage
**Test ID**: WB-BRANCH-002  
**Objective**: Test all branches in `update_user()`  
**Test Method**: `test_update_user_branch_coverage()`

**Branches Tested**:
- Branch 1: Update with data provided
- Branch 2: Update with no data (all None)
- Branch 3: Update with password (triggers password hashing)

**Test Steps**:
1. Create test user
2. Update with full_name → Expect True
3. Update with no parameters → Expect False
4. Update with password → Expect True (password hashing branch)

**Expected Result**: All branches execute with correct return values

---

### 2.3 Dispute DAL - Create Dispute Branch Coverage
**Test ID**: WB-BRANCH-003  
**Objective**: Test branches in `create_dispute()`  
**Test Method**: `test_create_dispute_branch_coverage()`

**Branches Tested**:
- Branch 1: With image_path
- Branch 2: Without image_path

**Test Steps**:
1. Create dispute with image_path
2. Verify success
3. Create dispute without image_path
4. Verify success

**Expected Result**: Both branches execute successfully

---

### 2.4 Notification DAL - Get Notifications Branch Coverage
**Test ID**: WB-BRANCH-004  
**Objective**: Test branches in `get_notifications_by_user()`  
**Test Method**: `test_get_notifications_by_user_branch_coverage()`

**Branches Tested**:
- Branch 1: unread_only=True (filters by read_at IS NULL)
- Branch 2: unread_only=False (returns all notifications)

**Test Steps**:
1. Create test notification
2. Get notifications with unread_only=True
3. Get notifications with unread_only=False
4. Verify unread count ≤ all count

**Expected Result**: Both query branches execute correctly

---

## 3. Path Coverage Tests

### 3.1 User DAL - Authentication Path Coverage
**Test ID**: WB-PATH-001  
**Objective**: Test all execution paths in `authenticate_user()`  
**Test Method**: `test_authenticate_user_path_coverage()`

**Paths Tested**:
- Path 1: User exists + Correct password → Success
- Path 2: User exists + Wrong password → Failure
- Path 3: User doesn't exist → Failure

**Test Steps**:
1. Create test user with known credentials
2. Authenticate with correct password → Expect user dict
3. Authenticate with wrong password → Expect None
4. Authenticate with non-existent email → Expect None

**Expected Result**: All three paths execute with correct outcomes

---

### 3.2 Dispute DAL - Update Status Path Coverage
**Test ID**: WB-PATH-002  
**Objective**: Test all paths in `update_dispute_status()`  
**Test Method**: `test_update_dispute_status_path_coverage()`

**Paths Tested**:
- Path 1: Update with all optional parameters
- Path 2: Update with only status (final status: approved/rejected/fraud_alert)
- Path 3: Update with non-final status

**Test Steps**:
1. Create test dispute
2. Update with all parameters (status, decision, trust_score, confidence, agent_id, agent_notes)
3. Update to final status (approved) → Sets resolution_timestamp
4. Create another dispute
5. Update to non-final status (pending) → No resolution_timestamp

**Expected Result**: All paths execute, resolution_timestamp set only for final statuses

---

### 3.3 Forensic DAL - Get by Dispute Path Coverage
**Test ID**: WB-PATH-003  
**Objective**: Test paths in `get_forensic_by_dispute()`  
**Test Method**: `test_get_forensic_by_dispute_path_coverage()`

**Paths Tested**:
- Path 1: Forensic result exists → Return dict
- Path 2: Forensic result doesn't exist → Return None

**Test Steps**:
1. Query existing dispute_id
2. Query non-existent dispute_id (999999)
3. Verify None returned for non-existent

**Expected Result**: Correct return values for both paths

---

## 4. Loop Coverage Tests

### 4.1 User DAL - Get Users by Role Loop Coverage
**Test ID**: WB-LOOP-001  
**Objective**: Test loop execution in `get_users_by_role()`  
**Test Method**: `test_get_users_by_role_loop_coverage()`

**Loop Scenarios**:
- Loop executes: Results exist (limit=5)
- Loop doesn't execute: No results (invalid role)

**Test Steps**:
1. Get users with role="customer" and limit=5
2. Verify list returned (loop processes results)
3. Get users with role="nonexistent_role"
4. Verify empty list (loop doesn't execute)

**Expected Result**: Loop handles both scenarios correctly

---

## 5. Base DAL Tests

### 5.1 Count Method Branch Coverage
**Test ID**: WB-BASE-001  
**Objective**: Test `count()` method with and without WHERE clause  
**Test Method**: `test_count_with_where_clause_branch()`

**Branches Tested**:
- Branch 1: Without WHERE clause
- Branch 2: With WHERE clause

**Test Steps**:
1. Call count() without parameters
2. Verify count ≥ 0
3. Call count() with WHERE clause and params
4. Verify count ≥ 0

**Expected Result**: Both branches execute successfully

---

### 5.2 Exists Method Path Coverage
**Test ID**: WB-BASE-002  
**Objective**: Test `exists()` method paths  
**Test Method**: `test_exists_method_paths()`

**Paths Tested**:
- Path 1: Record exists
- Path 2: Record doesn't exist

**Test Steps**:
1. Check if user_id=1 exists
2. Check if user_id=999999 exists
3. Verify False for non-existent

**Expected Result**: Correct boolean values returned

---

## 6. Forensic DAL Branch Coverage

### 6.1 Create Forensic Result Branch Coverage
**Test ID**: WB-BRANCH-005  
**Objective**: Test branches in `create_forensic_result()`  
**Test Method**: `test_create_forensic_result_branch_coverage()`

**Branches Tested**:
- Branch 1: With all optional parameters (flags, labels, processing_time)
- Branch 2: Without optional parameters

**Test Steps**:
1. Create forensic result with all optional fields
2. Verify success
3. Create forensic result with only required fields
4. Verify success

**Expected Result**: Both branches execute successfully

---

## 7. Notification DAL Branch Coverage

### 7.1 Create Notification Branch Coverage
**Test ID**: WB-BRANCH-006  
**Objective**: Test branches in `create_notification()`  
**Test Method**: `test_create_notification_branch_coverage()`

**Branches Tested**:
- Branch 1: With dispute_id
- Branch 2: Without dispute_id

**Test Steps**:
1. Create notification with dispute_id
2. Verify success
3. Create notification without dispute_id
4. Verify success

**Expected Result**: Both branches execute successfully

---

## Coverage Metrics

### Target Coverage
- **Statement Coverage**: 100%
- **Branch Coverage**: 100%
- **Path Coverage**: All critical paths
- **Loop Coverage**: All loop scenarios

### How to Run
```bash
cd "Assignment 8/Part_B_Testing"
python run_all_tests.py
```

### Coverage Report
HTML coverage report generated at: `white_box/coverage_html/index.html`

---

## Test Execution Summary

| Test Category | Test Count | Coverage Type |
|--------------|------------|---------------|
| Statement Coverage | 2 | All statements executed |
| Branch Coverage | 6 | All branches tested |
| Path Coverage | 3 | All critical paths |
| Loop Coverage | 1 | Loop execution scenarios |
| Base DAL Tests | 2 | Common operations |
| **Total** | **14** | **Comprehensive** |

---

## Notes
- All tests use unique identifiers (random hex) to avoid conflicts
- Tests are independent and can run in any order
- Database state is not reset between tests (tests are additive)
- Coverage report shows line-by-line execution

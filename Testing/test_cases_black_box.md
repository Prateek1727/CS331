# Black Box Test Cases
Assignment 8 - Part B: Testing

## Overview
Black box testing examines functionality without knowledge of internal implementation. This document describes all black box test cases organized by testing technique.

---

## 1. Boundary Value Analysis (BVA)

### 1.1 Dispute Amount Boundaries
**Test Class**: `TestDisputeAmountBoundaryValues`  
**Objective**: Test dispute amount at boundary values

#### Test Cases

| Test ID | Test Method | Input Amount | Expected Result |
|---------|-------------|--------------|-----------------|
| BB-BVA-001 | `test_amount_at_minimum_boundary` | ₹1.00 | Success |
| BB-BVA-002 | `test_amount_just_above_minimum` | ₹1.01 | Success |
| BB-BVA-003 | `test_amount_at_maximum_boundary` | ₹10000.00 | Success |
| BB-BVA-004 | `test_amount_just_below_maximum` | ₹9999.99 | Success |
| BB-BVA-005 | `test_amount_in_middle_range` | ₹5000.00 | Success |

**Boundary Analysis**:
- Minimum valid: ₹1.00
- Maximum valid: ₹10000.00
- Below minimum: ₹0.99 (invalid - not tested, would violate constraint)
- Above maximum: ₹10000.01 (invalid - not tested, would violate constraint)

---

### 1.2 Dispute Description Length Boundaries
**Test Class**: `TestDisputeDescriptionBoundaryValues`  
**Objective**: Test description length at boundary values

#### Test Cases

| Test ID | Test Method | Description Length | Expected Result |
|---------|-------------|-------------------|-----------------|
| BB-BVA-006 | `test_description_at_minimum_length` | 10 characters | Success |
| BB-BVA-007 | `test_description_just_above_minimum` | 11 characters | Success |
| BB-BVA-008 | `test_description_at_maximum_length` | 500 characters | Success |
| BB-BVA-009 | `test_description_just_below_maximum` | 499 characters | Success |

**Boundary Analysis**:
- Minimum valid: 10 characters
- Maximum valid: 500 characters
- Below minimum: 9 characters (invalid - not tested)
- Above maximum: 501 characters (invalid - not tested)

---

### 1.3 Forensic Score Boundaries
**Test Class**: `TestForensicScoreBoundaryValues`  
**Objective**: Test forensic scores at boundary values

#### Test Cases

| Test ID | Test Method | Score Values | Expected Result |
|---------|-------------|--------------|-----------------|
| BB-BVA-010 | `test_scores_at_minimum_boundary` | 0.0, 0.0, 0.0 | Success |
| BB-BVA-011 | `test_scores_at_maximum_boundary` | 1.0, 1.0, 1.0 | Success |
| BB-BVA-012 | `test_scores_in_middle_range` | 0.5, 0.5, 0.5 | Success |

**Boundary Analysis**:
- Minimum valid: 0.0 (all scores)
- Maximum valid: 1.0 (all scores)
- Applies to: metadata_score, ela_score, ai_score

---

## 2. Equivalence Partitioning

### 2.1 User Role Equivalence Classes
**Test Class**: `TestUserRoleEquivalencePartitioning`  
**Objective**: Test all valid user role values

#### Equivalence Classes

| Class ID | Class Type | Values | Test ID | Test Method |
|----------|-----------|--------|---------|-------------|
| EC-ROLE-01 | Valid | customer | BB-EP-001 | `test_valid_role_customer` |
| EC-ROLE-02 | Valid | agent | BB-EP-002 | `test_valid_role_agent` |
| EC-ROLE-03 | Valid | admin | BB-EP-003 | `test_valid_role_admin` |
| EC-ROLE-04 | Invalid | other | N/A | Not tested (DB constraint) |

**Test Steps** (for each valid role):
1. Create user with specified role
2. Verify user_id returned
3. Retrieve user from database
4. Verify role matches input

---

### 2.2 Dispute Status Equivalence Classes
**Test Class**: `TestDisputeStatusEquivalencePartitioning`  
**Objective**: Test all valid dispute status values

#### Equivalence Classes

| Class ID | Class Type | Values | Test ID | Test Method |
|----------|-----------|--------|---------|-------------|
| EC-STATUS-01 | Valid | pending | BB-EP-004 | `test_status_pending` |
| EC-STATUS-02 | Valid | under_review | BB-EP-005 | `test_status_under_review` |
| EC-STATUS-03 | Valid | approved | BB-EP-006 | `test_status_approved` |
| EC-STATUS-04 | Valid | rejected | BB-EP-007 | `test_status_rejected` |
| EC-STATUS-05 | Valid | fraud_alert | N/A | Not explicitly tested |
| EC-STATUS-06 | Invalid | other | N/A | Not tested (DB constraint) |

**Test Steps**:
1. Create dispute (default status: pending)
2. Update to target status
3. Retrieve dispute from database
4. Verify status matches

---

### 2.3 Notification Priority Equivalence Classes
**Test Class**: `TestNotificationPriorityEquivalencePartitioning`  
**Objective**: Test all valid notification priority values

#### Equivalence Classes

| Class ID | Class Type | Values | Test ID | Test Method |
|----------|-----------|--------|---------|-------------|
| EC-PRIORITY-01 | Valid | low | BB-EP-008 | `test_priority_low` |
| EC-PRIORITY-02 | Valid | medium | BB-EP-009 | `test_priority_medium` |
| EC-PRIORITY-03 | Valid | high | BB-EP-010 | `test_priority_high` |
| EC-PRIORITY-04 | Valid | urgent | BB-EP-011 | `test_priority_urgent` |
| EC-PRIORITY-05 | Invalid | other | N/A | Not tested (DB constraint) |

**Test Steps**:
1. Create notification with specified priority
2. Verify notification_id returned
3. Verify creation successful

---

## 3. Decision Table Testing

### 3.1 User Authentication Decision Table
**Test Class**: `TestAuthenticationDecisionTable`  
**Objective**: Test authentication logic with all input combinations

#### Decision Table

| Test ID | User Exists | Password Correct | Expected Result | Test Method |
|---------|------------|------------------|-----------------|-------------|
| BB-DT-001 | Yes | Yes | Success (User dict) | `test_user_exists_correct_password` |
| BB-DT-002 | Yes | No | Failure (None) | `test_user_exists_wrong_password` |
| BB-DT-003 | No | Any | Failure (None) | `test_user_not_exists_any_password` |

**Test Steps**:

**BB-DT-001**: User exists + Correct password
1. Create test user with known credentials
2. Authenticate with correct email and password
3. Verify user dict returned
4. Verify email matches

**BB-DT-002**: User exists + Wrong password
1. Use existing test user
2. Authenticate with correct email but wrong password
3. Verify None returned

**BB-DT-003**: User doesn't exist
1. Authenticate with non-existent email
2. Verify None returned

---

## 4. State Transition Testing

### 4.1 Dispute Status Transitions
**Objective**: Test valid state transitions for disputes

#### Valid Transitions

```
pending → under_review → approved
                      → rejected
                      → fraud_alert

pending → approved (direct)
pending → rejected (direct)
```

#### Test Coverage

| From Status | To Status | Valid? | Tested In |
|------------|-----------|--------|-----------|
| pending | under_review | Yes | BB-EP-005 |
| pending | approved | Yes | BB-EP-006 |
| pending | rejected | Yes | BB-EP-007 |
| under_review | approved | Yes | Implicit |
| under_review | rejected | Yes | Implicit |
| approved | rejected | No | Not tested |
| rejected | approved | No | Not tested |

---

## 5. Test Data Specifications

### 5.1 Valid Input Ranges

| Field | Data Type | Min | Max | Valid Examples |
|-------|-----------|-----|-----|----------------|
| amount | DECIMAL(10,2) | 1.00 | 10000.00 | 1.00, 5000.00, 10000.00 |
| description | TEXT | 10 chars | 500 chars | "Test dispute", "A"*500 |
| metadata_score | DECIMAL(5,4) | 0.0000 | 1.0000 | 0.0, 0.5, 1.0 |
| ela_score | DECIMAL(5,4) | 0.0000 | 1.0000 | 0.0, 0.5, 1.0 |
| ai_score | DECIMAL(5,4) | 0.0000 | 1.0000 | 0.0, 0.5, 1.0 |

### 5.2 Enumerated Values

| Field | Valid Values |
|-------|-------------|
| user_role | customer, agent, admin |
| dispute_status | pending, under_review, approved, rejected, fraud_alert |
| decision_type | auto_refund, manual_review, fraud_alert |
| notification_type | email, sms, push |
| notification_priority | low, medium, high, urgent |

---

## 6. Test Execution

### How to Run
```bash
cd "Assignment 8/Part_B_Testing"
python run_all_tests.py
```

Or run black box tests only:
```bash
cd "Assignment 8/Part_B_Testing"
python -m pytest black_box/test_black_box.py -v
```

---

## 7. Test Summary

| Testing Technique | Test Classes | Test Cases | Coverage |
|------------------|--------------|------------|----------|
| Boundary Value Analysis | 3 | 12 | Amount, Description, Scores |
| Equivalence Partitioning | 3 | 11 | Roles, Status, Priority |
| Decision Table Testing | 1 | 3 | Authentication |
| **Total** | **7** | **26** | **Comprehensive** |

---

## 8. Expected Results Summary

### All Tests Should Pass
- All boundary values within valid ranges should succeed
- All valid equivalence class values should succeed
- Authentication decision table should match expected outcomes
- No database constraint violations

### Test Independence
- Each test uses unique identifiers (random hex)
- Tests can run in any order
- No test depends on another test's data

---

## 9. Notes

### Test Data Generation
- Uses `os.urandom(4).hex()` for unique identifiers
- Prevents conflicts between test runs
- Ensures test isolation

### Database Constraints
- Invalid values (below min, above max) are not tested
- Database constraints prevent invalid data
- Focus is on valid input ranges and equivalence classes

### Coverage Focus
- Tests cover all valid input ranges
- Tests cover all valid enumerated values
- Tests cover critical decision logic
- Tests verify expected behavior without internal knowledge

import os
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import database

def test_database_and_cooldown():
    test_db = "test_attendance.db"
    if os.path.exists(test_db):
        os.remove(test_db)
        
    print("1. Testing database initialization...")
    database.init_db(test_db)
    assert os.path.exists(test_db), "Database file not created"
    
    print("2. Testing worker insertion and serialization...")
    # 128-dimensional mock face vector
    mock_vector_1 = np.random.randn(128).astype(np.float64)
    mock_vector_2 = np.random.randn(128).astype(np.float64)
    
    w1_id = database.insert_worker("Alice Smith", mock_vector_1, db_path=test_db)
    w2_id = database.insert_worker("Bob Jones", mock_vector_2, db_path=test_db)
    
    assert w1_id == 1, f"Expected w1_id=1, got {w1_id}"
    assert w2_id == 2, f"Expected w2_id=2, got {w2_id}"
    
    workers = database.get_all_workers(db_path=test_db)
    assert len(workers) == 2, f"Expected 2 workers, found {len(workers)}"
    np.testing.assert_allclose(workers[0]["encoding"], mock_vector_1 if workers[0]["name"] == "Alice Smith" else mock_vector_2)
    
    print("3. Testing attendance logging & cooldown...")
    # First check-in should succeed
    logged_1, msg_1 = database.log_attendance(w1_id, cooldown_minutes=5, db_path=test_db)
    assert logged_1 is True, f"First log should succeed, got {logged_1} ({msg_1})"
    
    # Immediate second check-in for w1 should be blocked by cooldown
    logged_2, msg_2 = database.log_attendance(w1_id, cooldown_minutes=5, db_path=test_db)
    assert logged_2 is False, f"Immediate second log should be blocked, got {logged_2} ({msg_2})"
    assert "Cooldown active" in msg_2
    
    # Check-in for different worker w2 should succeed
    logged_3, msg_3 = database.log_attendance(w2_id, cooldown_minutes=5, db_path=test_db)
    assert logged_3 is True, f"w2 check-in should succeed, got {logged_3} ({msg_3})"
    
    print("4. Testing attendance query...")
    today_str = datetime.now().strftime("%Y-%m-%d")
    logs = database.get_attendance_logs(date_str=today_str, db_path=test_db)
    assert len(logs) == 2, f"Expected 2 logs, found {len(logs)}"
    assert logs[0]["Worker Name"] in ["Alice Smith", "Bob Jones"]
    
    # Cleanup test db
    if os.path.exists(test_db):
        os.remove(test_db)
        
    print(" All database and logic tests PASSED!")

if __name__ == "__main__":
    test_database_and_cooldown()

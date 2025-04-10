import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        symptoms TEXT,
        diseases TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_user(name, email, password, symptoms=None, diseases=None):
    conn = get_db_connection()
    hashed_pw = generate_password_hash(password)
    try:
        conn.execute("""INSERT INTO users
                      (name, email, password, symptoms, diseases)
                      VALUES (?, ?, ?, ?, ?)""",
                    (name, email, hashed_pw, symptoms, diseases))
        print("database created")
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user["password"], password):
        return dict(user)  # Convert Row object to dictionary
    return None

def update_user_health(user_id, symptoms=None, diseases=None):
    """Update user's health information"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET symptoms = ?, diseases = ?
            WHERE id = ?
        """, (symptoms, diseases, user_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating user health: {e}")
        return False
    finally:
        conn.close()

def get_user_health(user_id):
    """Get user's health information"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT symptoms, diseases FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        return dict(user) if user else None
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()

import sqlite3

def save_audit(transcript, result):

    conn = sqlite3.connect("audit_results.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript TEXT,
        result TEXT
    )
    """)

    cursor.execute(
        "INSERT INTO audits (transcript, result) VALUES (?, ?)",
        (transcript, result)
    )

    conn.commit()
    conn.close()
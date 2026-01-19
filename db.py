# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:18:43 2026

@author: Oreoluwa
"""

# db.py
import sqlite3

DB_FILE = "app_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    # History table
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            url TEXT,
            title TEXT NOT NULL,
            verdict TEXT NOT NULL,
            satire_prob REAL,
            fake_prob REAL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def validate_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]  # user_id
    return None

def get_user_history(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT url, title, verdict, satire_prob, fake_prob, timestamp 
        FROM history 
        WHERE user_id=? ORDER BY id DESC
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def add_history(user_id, url, title, verdict, satire_prob, fake_prob):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO history (user_id, url, title, verdict, satire_prob, fake_prob, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (user_id, url, title, verdict, satire_prob, fake_prob))
    conn.commit()
    conn.close()

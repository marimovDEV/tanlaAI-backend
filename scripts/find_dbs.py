import os
import sqlite3
from pathlib import Path

def count_products_in_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shop_product';")
        if cursor.fetchone():
            cursor.execute("SELECT count(*) FROM shop_product;")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        conn.close()
        return "No Shop Table"
    except Exception as e:
        return f"Error: {e}"

def find_all_dbs():
    root = Path('C:/Users/user/Downloads/stitch')
    print(f"🔎 Deep Search in: {root}")
    
    sqlite_files = list(root.rglob('*.sqlite3'))
    
    if not sqlite_files:
        print("❌ No .sqlite3 files found in the entire root folder.")
        return

    print(f"📊 Found {len(sqlite_files)} database files:")
    for db_p in sqlite_files:
        size = os.path.getsize(db_p)
        count = count_products_in_db(db_p)
        print(f"📍 {db_p}")
        print(f"   Size: {size / 1024:.2f} KB | Products: {count}")

if __name__ == "__main__":
    find_all_dbs()

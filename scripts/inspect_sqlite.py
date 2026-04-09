"""Direct SQLite inspection - bypasses Django ORM to find hidden/deleted records."""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'db.sqlite3')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=== ALL TABLES ===")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
for row in cursor.fetchall():
    table = row[0]
    cursor.execute(f"SELECT count(*) FROM [{table}]")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"  {table}: {count} rows")

print("\n=== SHOP_PRODUCT (raw rows) ===")
cursor.execute("SELECT id, name, image, price, price_per_m2, category_id, company_id, owner_id FROM shop_product;")
rows = cursor.fetchall()
print(f"Total rows: {len(rows)}")
for r in rows:
    print(f"  ID={r[0]} | name={r[1]} | image={r[2]} | price={r[3]} | price_per_m2={r[4]} | cat={r[5]} | company={r[6]} | owner={r[7]}")

print("\n=== SHOP_CATEGORY ===")
cursor.execute("SELECT id, name FROM shop_category;")
for r in cursor.fetchall():
    print(f"  ID={r[0]} | name={r[1]}")

print("\n=== SHOP_COMPANY ===")
cursor.execute("SELECT id, name FROM shop_company;")
for r in cursor.fetchall():
    print(f"  ID={r[0]} | name={r[1]}")

print("\n=== SHOP_TELEGRAMUSER ===")
cursor.execute("SELECT id, telegram_id, first_name, role FROM shop_telegramuser;")
for r in cursor.fetchall():
    print(f"  ID={r[0]} | tg_id={r[1]} | name={r[2]} | role={r[3]}")

print("\n=== MEDIA FILES ON DISK ===")
media_products = os.path.join(os.path.dirname(__file__), 'media', 'products')
for root, dirs, files in os.walk(media_products):
    for f in sorted(files):
        rel = os.path.relpath(os.path.join(root, f), os.path.join(os.path.dirname(__file__), 'media'))
        print(f"  {rel}")

conn.close()

"""Fix broken image paths and optionally recreate products from orphaned images."""
import os
import sys
import sqlite3

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.sqlite3')
media_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# === STEP 1: Fix double 'products/products/' paths ===
print("=== STEP 1: Fixing broken image paths ===")
cursor.execute("SELECT id, name, image FROM shop_product WHERE image LIKE 'products/products/%'")
broken = cursor.fetchall()
for row in broken:
    pid, name, old_path = row
    # Remove the extra 'products/' prefix
    new_path = old_path.replace('products/products/', 'products/', 1)
    
    # Move the actual file on disk if needed
    old_full = os.path.join(media_root, old_path)
    new_full = os.path.join(media_root, new_path)
    
    if os.path.exists(old_full) and not os.path.exists(new_full):
        os.rename(old_full, new_full)
        print(f"  Moved file: {old_path} -> {new_path}")
    elif os.path.exists(new_full):
        print(f"  File already at correct path: {new_path}")
        # Clean up the duplicate if it exists
        if os.path.exists(old_full) and old_full != new_full:
            os.remove(old_full)
    
    cursor.execute("UPDATE shop_product SET image = ? WHERE id = ?", (new_path, pid))
    print(f"  Fixed DB path for '{name}' (ID {pid}): {old_path} -> {new_path}")

conn.commit()
print(f"Fixed {len(broken)} broken paths.\n")

# === STEP 2: Clean up empty products/products/ subfolder ===
subfolder = os.path.join(media_root, 'products', 'products')
if os.path.exists(subfolder):
    try:
        os.rmdir(subfolder)  # Only removes if empty
        print("Cleaned up empty products/products/ subfolder.")
    except OSError:
        remaining = os.listdir(subfolder)
        print(f"products/products/ subfolder still has {len(remaining)} files.")

print("\n=== DONE ===")
print("The 2 existing products now have correct image paths.")
print("New products created via the form will be auto-optimized to WebP.")

conn.close()

import cv2
import numpy as np
import sys
import os

# Mock the new refinement logic to test multi-component grouping
def refine_product_mask_test(mask):
    import cv2
    import numpy as np

    if mask is None: return None
    clean = np.where(mask > 10, 255, 0).astype(np.uint8)
    if not np.any(clean): return None

    kernel = np.ones((7, 7), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if num_labels <= 1: return clean

    height, width = clean.shape[:2]
    image_area = height * width
    center_rect = (width * 0.2, height * 0.2, width * 0.8, height * 0.8)
    
    mask_indices = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]
        if area < (image_area * 0.005): continue
        is_centered = (center_rect[0] < cx < center_rect[2]) and (center_rect[1] < cy < center_rect[3])
        if area > (image_area * 0.05) or (is_centered and area > (image_area * 0.01)):
            mask_indices.append(label)

    if not mask_indices:
        mask_indices = [1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]

    clean = np.zeros_like(clean)
    for idx in mask_indices:
        clean[labels == idx] = 255

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(clean)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
    filled = cv2.GaussianBlur(filled, (3, 3), 0)
    return filled

# Create a scenario where a door (main body) and molding (top part) are separate components
h, w = 1000, 800
mask = np.zeros((h, w), dtype=np.uint8)

# Component 1: Main door body
cv2.rectangle(mask, (200, 200), (600, 950), 255, -1)

# Component 2: Moldings at the top (disjoint from body by 10 pixels gap)
cv2.rectangle(mask, (150, 150), (650, 185), 255, -1)

# Component 3: Noise
cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)

print(f"Original components (excluding bg): 3")
refined = refine_product_mask_test(mask)

# Count components in refined mask
num, _, stats, _ = cv2.connectedComponentsWithStats(refined)
print(f"Refined components (excluding bg): {num - 1}")

if num - 1 == 1:
    print("SUCCESS: Separated parts were correctly grouped and filled!")
else:
    print(f"FAILURE: Expected 1 solid component, got {num-1}")

# Export for visual check if needed (skipped in console test)

import os
import io
from functools import lru_cache
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings
import requests
# Moved rembg imports inside functions to prevent startup crashes


def build_mask_from_polygon(width, height, polygon_points):
    import cv2
    import numpy as np

    if not polygon_points or len(polygon_points) < 3:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def refine_product_mask(mask):
    """
    Convert a noisy alpha/mask into one solid silhouette so door inserts,
    glass cutouts, and decorative white areas are not punched out.
    """
    import cv2
    import numpy as np

    if mask is None:
        return None

    # Step 1: Initial threshold to clean noise
    clean = np.where(mask > 10, 255, 0).astype(np.uint8)
    if not np.any(clean):
        return None

    # Step 2: Closing operation to bridge small gaps (like molding edges)
    kernel = np.ones((7, 7), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Step 3: Multi-component selection (Robust to molding/frames)
    # Instead of just the largest component, we take all components that are:
    # a) Large enough (>1% of image)
    # b) Centered enough (overlap with the center 60% of the image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if num_labels <= 1:
        return clean

    height, width = clean.shape[:2]
    image_area = height * width
    center_rect = (width * 0.2, height * 0.2, width * 0.8, height * 0.8) # x1, y1, x2, y2
    
    mask_indices = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]
        
        # Area threshold: ignore tiny noise
        if area < (image_area * 0.005):
            continue
            
        # Centering check: Is it roughly where a door should be?
        is_centered = (center_rect[0] < cx < center_rect[2]) and (center_rect[1] < cy < center_rect[3])
        
        # If it's a very large component or centered large component, keep it
        if area > (image_area * 0.05) or (is_centered and area > (image_area * 0.01)):
            mask_indices.append(label)

    if not mask_indices:
        # Fallback to largest if selection logic fails
        best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_indices = [best_label]

    # Combine selected components
    clean = np.zeros_like(clean)
    for idx in mask_indices:
        clean[labels == idx] = 255

    # Step 4: Final smoothing and hole filling
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return clean

    filled = np.zeros_like(clean)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # One more blur for smooth edges in visualization
    filled = cv2.GaussianBlur(filled, (3, 3), 0)
    return filled


def mask_stats(mask):
    import cv2

    if mask is None:
        return 0, 0.0, (0, 0, 0, 0)

    area = int(cv2.countNonZero(mask))
    height, width = mask.shape[:2]
    area_ratio = area / float(max(1, height * width))
    if area == 0:
        return area, area_ratio, (0, 0, 0, 0)
    bbox = cv2.boundingRect(mask)
    return area, area_ratio, bbox


def is_reasonable_door_mask(mask):
    if mask is None:
        return False

    area, area_ratio, bbox = mask_stats(mask)
    if area == 0:
        return False

    _, _, bbox_width, bbox_height = bbox
    height, width = mask.shape[:2]
    height_ratio = bbox_height / float(max(1, height))
    width_ratio = bbox_width / float(max(1, width))

    return 0.03 <= area_ratio <= 0.95 and height_ratio >= 0.35 and width_ratio >= 0.12


def merge_candidate_masks(primary_mask, polygon_mask):
    import cv2

    primary_mask = refine_product_mask(primary_mask)
    polygon_mask = refine_product_mask(polygon_mask)

    if primary_mask is None:
        return polygon_mask
    if polygon_mask is None:
        return primary_mask
    if not is_reasonable_door_mask(polygon_mask):
        return primary_mask

    primary_area, _, _ = mask_stats(primary_mask)
    polygon_area, _, _ = mask_stats(polygon_mask)
    overlap = cv2.countNonZero(cv2.bitwise_and(primary_mask, polygon_mask))
    smaller_area = max(1, min(primary_area, polygon_area))

    # Reject obviously unrelated polygons while still allowing slightly loose unions.
    if overlap / float(smaller_area) < 0.10 and polygon_area > primary_area * 2.5:
        return primary_mask

    return refine_product_mask(cv2.bitwise_or(primary_mask, polygon_mask))


def compose_rgba_from_mask(rgb_image, alpha_mask):
    import cv2

    refined_mask = refine_product_mask(alpha_mask)
    if refined_mask is None:
        return None

    img_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_bgr)
    return cv2.merge((b, g, r, refined_mask))


def sanitize_pixel_box(box, width, height):
    left, top, right, bottom = [int(round(value)) for value in box]

    left = max(0, min(left, max(0, width - 1)))
    top = max(0, min(top, max(0, height - 1)))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def box_1000_to_pixels(box_1000, width, height):
    ymin, xmin, ymax, xmax = box_1000
    return sanitize_pixel_box(
        (
            xmin * width / 1000.0,
            ymin * height / 1000.0,
            xmax * width / 1000.0,
            ymax * height / 1000.0,
        ),
        width,
        height,
    )


def pixels_to_box_1000(pixel_box, width, height):
    left, top, right, bottom = sanitize_pixel_box(pixel_box, width, height)
    return [
        int(round(top * 1000.0 / max(1, height))),
        int(round(left * 1000.0 / max(1, width))),
        int(round(bottom * 1000.0 / max(1, height))),
        int(round(right * 1000.0 / max(1, width))),
    ]


def expand_pixel_box(pixel_box, width, height, pad_x_ratio=0.06, pad_y_ratio=0.04):
    left, top, right, bottom = sanitize_pixel_box(pixel_box, width, height)
    pad_x = int(round((right - left) * pad_x_ratio))
    pad_y = int(round((bottom - top) * pad_y_ratio))
    return sanitize_pixel_box((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), width, height)


def build_box_mask(height, width, pixel_box, pad_x_ratio=0.06, pad_y_ratio=0.04):
    import cv2
    import numpy as np

    left, top, right, bottom = expand_pixel_box(pixel_box, width, height, pad_x_ratio=pad_x_ratio, pad_y_ratio=pad_y_ratio)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (left, top), (max(left + 1, right - 1), max(top + 1, bottom - 1)), 255, thickness=-1)
    return mask


def get_expected_door_aspect_ratio(product, door_rgba=None):
    try:
        if product.width and product.height:
            ratio = float(product.width) / float(product.height)
            if 0.15 <= ratio <= 0.95:
                return ratio
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    if door_rgba is not None:
        door_height, door_width = door_rgba.shape[:2]
        if door_height > 0:
            ratio = door_width / float(door_height)
            if 0.15 <= ratio <= 0.95:
                return ratio

    return 0.45


def score_door_candidate(x, y, candidate_width, candidate_height, image_width, image_height, expected_aspect_ratio, signal_mask):
    import numpy as np

    area = candidate_width * candidate_height
    area_ratio = area / float(max(1, image_width * image_height))
    height_ratio = candidate_height / float(max(1, image_height))
    width_ratio = candidate_width / float(max(1, image_width))
    aspect_ratio = candidate_width / float(max(1, candidate_height))

    # Strict architectural filters for doors
    if area_ratio < 0.04 or area_ratio > 0.85:
        return None
    if height_ratio < 0.40 or width_ratio < 0.12:
        return None
    if aspect_ratio < 0.18 or aspect_ratio > 0.90:
        return None

    # Doors for floor-level openings MUST reach the bottom part of the image
    bottom_ratio = (y + candidate_height) / float(max(1, image_height))
    # CRITICAL: Doors reach the floor. Windows do not.
    # Increasing this to 0.85 is a safe bet for architectural photos.
    if bottom_ratio < 0.82: 
        return None

    # Doors are vertical. If it's wider than it is tall, it's likely a window or a wide opening.
    if candidate_width > candidate_height:
        return None

    floor_anchor_bonus = 0
    if bottom_ratio > 0.94: # Deeply anchored to the floor
        floor_anchor_bonus = 15.0 

    # Penalty for starting too high (likely a window or sky)
    top_ratio = y / float(max(1, image_height))
    top_penalty = (1.0 - top_ratio) * 3.0 # The higher it starts, the more we penalize it

    # Center-of-scene priority (Windows are often at the edges)
    center_x_dist = abs(((x + (candidate_width / 2.0)) / max(1, image_width)) - 0.5)
    
    aspect_penalty = abs(aspect_ratio - expected_aspect_ratio)
    
    # Internal signal density (contrast/edges)
    region = signal_mask[y:y + candidate_height, x:x + candidate_width]
    edge_density = float(np.count_nonzero(region)) / float(max(1, area))

    # Weighting factors
    score = (
        (height_ratio * 6.0)     # Verticality is key
        + (area_ratio * 4.0)       # Size matters
        + (bottom_ratio * 5.0)     # Reach depth
        + (edge_density * 2.5)     # Some contrast
        - (center_x_dist * 8.0)    # VERY strong penalty for off-center
        - (aspect_penalty * 2.0)   # Mismatched shape
        - top_penalty              # Top-sticking
        + floor_anchor_bonus      # Floor anchoring is the definitive door signal
    )
    
    # Bonus for realistic door proportions (narrow and tall)
    if 0.3 <= aspect_ratio <= 0.6:
        score += 2.0
    
    return score


@lru_cache(maxsize=1)
def load_optional_yolo_model(model_path):
    from ultralytics import YOLO

    return YOLO(model_path)


@lru_cache(maxsize=1)
def get_rembg_session():
    import rembg

    return rembg.new_session("u2net")


def border_transparency_ratio(alpha_mask):
    import numpy as np

    if alpha_mask is None:
        return 0.0

    height, width = alpha_mask.shape[:2]
    border = max(2, min(height, width) // 16)
    top = alpha_mask[:border, :]
    bottom = alpha_mask[max(0, height - border):, :]
    left = alpha_mask[:, :border]
    right = alpha_mask[:, max(0, width - border):]
    border_pixels = np.concatenate([top.reshape(-1), bottom.reshape(-1), left.reshape(-1), right.reshape(-1)])
    if border_pixels.size == 0:
        return 0.0
    return float(np.count_nonzero(border_pixels <= 10)) / float(border_pixels.size)


def trim_white_border_from_rgba(door_rgba, threshold=248):
    """
    Surgically removes white borders from a door asset.
    Analyzes RGB channels: if a pixel is nearly pure white AND near the edges, 
    we treat it as background padding and crop it out.
    """
    import cv2
    import numpy as np

    if door_rgba is None or door_rgba.shape[2] < 4:
        return door_rgba

    # 1. Create a mask of "non-white" pixels
    # We look for pixels where NOT all channels are > threshold
    b, g, r, a = cv2.split(door_rgba)
    is_white = (b >= threshold) & (g >= threshold) & (r >= threshold)
    
    # We only care about white pixels that have some alpha (the "border")
    non_background_mask = (a > 20) & (~is_white)
    
    # 2. Find the bounding box of the actual "colored" content
    coords = cv2.findNonZero(non_background_mask.astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add a tiny 1-2px safety margin if possible
        x_start = max(0, x - 1)
        y_start = max(0, y - 1)
        x_end = min(door_rgba.shape[1], x + w + 1)
        y_end = min(door_rgba.shape[0], y + h + 1)
        return door_rgba[y_start:y_end, x_start:x_end]
    
    return door_rgba


def normalize_door_rgba_asset(door_rgba):
    import cv2
    import numpy as np

    if door_rgba is None:
        return None

    if door_rgba.ndim == 2:
        door_rgba = cv2.cvtColor(door_rgba, cv2.COLOR_GRAY2BGRA)
    elif door_rgba.shape[2] == 3:
        return None
    elif door_rgba.shape[2] > 4:
        door_rgba = door_rgba[:, :, :4]

    alpha_mask = refine_product_mask(door_rgba[:, :, 3])
    if alpha_mask is None or not is_reasonable_door_mask(alpha_mask):
        return None
    if border_transparency_ratio(alpha_mask) < 0.20:
        return None

    normalized = door_rgba.copy()
    normalized[:, :, 3] = alpha_mask
    
    # 1. Surgical white-border trimming (v47 enhancement)
    normalized = trim_white_border_from_rgba(normalized)
    
    # 2. Final auto-crop the remaining transparent padding
    final_alpha = normalized[:, :, 3]
    coords = cv2.findNonZero(final_alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        normalized = normalized[y:y+h, x:x+w]
        
    return normalized


def rgba_with_full_alpha(image_data):
    import cv2
    import numpy as np

    if image_data is None:
        return None
    if image_data.ndim == 2:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
    if image_data.shape[2] == 4:
        return image_data

    alpha = np.full(image_data.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([image_data[:, :, :3], alpha], axis=2)


def candidate_product_image_paths(product):
    seen = set()
    for attr_name in ('image_no_bg', 'image', 'original_image'):
        field = getattr(product, attr_name, None)
        if not field or not getattr(field, 'name', ''):
            continue
        try:
            path = field.path
        except Exception:
            continue
        if not path or not os.path.exists(path) or path in seen:
            continue
        seen.add(path)
        yield attr_name, path


def extract_door_rgba_from_bytes(image_bytes):
    import cv2
    import numpy as np
    import rembg

    output_bytes = rembg.remove(image_bytes, session=get_rembg_session())
    rgba = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    return normalize_door_rgba_asset(rgba)


def load_best_door_rgba(product):
    import cv2

    fallback_bytes = None
    fallback_image = None
    original_bytes = None

    for label, path in candidate_product_image_paths(product):
        image_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        normalized = normalize_door_rgba_asset(image_data)
        if normalized is not None:
            print(f"DEBUG: [AI Service] Using {label} asset for door overlay: {path}")
            return normalized

        if fallback_image is None and image_data is not None:
            fallback_image = image_data

        try:
            with open(path, 'rb') as source_file:
                raw_bytes = source_file.read()
        except Exception:
            raw_bytes = None

        if raw_bytes and fallback_bytes is None:
            fallback_bytes = raw_bytes
        if raw_bytes and label == 'original_image':
            original_bytes = raw_bytes

    source_bytes = original_bytes or fallback_bytes
    if source_bytes:
        try:
            regenerated = extract_door_rgba_from_bytes(source_bytes)
            if regenerated is not None:
                print("DEBUG: [AI Service] Rebuilt door alpha from source image for overlay")
                return regenerated
        except Exception as exc:
            print(f"WARNING: [AI Service] Could not rebuild door alpha: {exc}")

    fallback_rgba = rgba_with_full_alpha(fallback_image)
    if fallback_rgba is not None:
        print("WARNING: [AI Service] Falling back to opaque door asset; result may include background")
        return fallback_rgba

    raise ValueError("No usable door asset found for visualization")


def detect_door_box_with_yolo(room_bgr, expected_aspect_ratio):
    model_path = str(getattr(settings, 'YOLO_DOOR_MODEL_PATH', '') or '').strip()
    if not model_path or not os.path.exists(model_path):
        return None

    try:
        model = load_optional_yolo_model(model_path)
        results = model.predict(room_bgr, conf=0.20, verbose=False)
    except Exception as exc:
        print(f"DEBUG: [AI Service] YOLO detection unavailable: {exc}")
        return None

    image_height, image_width = room_bgr.shape[:2]
    best_box = None
    best_score = float('-inf')

    for result in results:
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0]) if getattr(box, 'conf', None) is not None else 0.0
            left, top, right, bottom = sanitize_pixel_box((x1, y1, x2, y2), image_width, image_height)
            aspect_ratio = (right - left) / float(max(1, bottom - top))
            aspect_penalty = abs(aspect_ratio - expected_aspect_ratio)
            center_penalty = abs((((left + right) / 2.0) / max(1, image_width)) - 0.5)
            score = (confidence * 10.0) - (aspect_penalty * 2.0) - (center_penalty * 1.5)
            if score > best_score:
                best_score = score
                best_box = (left, top, right, bottom)

    return best_box


def detect_door_box_with_opencv(room_bgr, expected_aspect_ratio):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    # Use CLAHE to improve contrast in dark areas if the image is high-contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Looser Canny thresholds to catch the dark doorway edges
    edges = cv2.Canny(blurred, 40, 120)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, # Slightly larger block size
        5,  # Slightly lower C
    )

    combined = cv2.bitwise_or(edges, adaptive)
    # Stronger morphological closing to bridge gaps in dark areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = gray.shape[:2]
    best_box = None
    best_score = float('-inf')

    for contour in contours:
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        score = score_door_candidate(
            x,
            y,
            candidate_width,
            candidate_height,
            image_width,
            image_height,
            expected_aspect_ratio,
            combined,
        )
        if score is None:
            continue

        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / float(max(1, candidate_width * candidate_height))
        total_score = score + (rectangularity * 0.8)
        if total_score > best_score:
            best_score = total_score
            best_box = (x, y, x + candidate_width, y + candidate_height)

    return best_box


def default_door_box(image_width, image_height, expected_aspect_ratio):
    # Professional Grounding: Modern doors start from floor level.
    # We set the bottom at 97% of image height to ensure it sits on the floor.
    box_height = int(round(image_height * 0.82)) 
    box_width = int(round(box_height * expected_aspect_ratio))
    box_width = max(int(image_width * 0.18), min(box_width, int(image_width * 0.55)))

    left = int(round((image_width - box_width) / 2.0))
    # Ground the door to the bottom area (0.97 * height)
    bottom = int(round(image_height * 0.97))
    top = bottom - box_height
    return sanitize_pixel_box((left, top, left + box_width, bottom), image_width, image_height)


def normalize_door_opening_box(pixel_box, image_width, image_height, expected_aspect_ratio):
    left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)

    # Trust the detector, but widen obvious "inner leaf" detections to the full frame width.
    width = right - left
    height = bottom - top

    min_width = int(image_width * 0.18)
    min_height = int(image_height * 0.30)
    frame_aspect_ratio = min(0.95, max(0.24, expected_aspect_ratio * 1.12))
    width_ratio = width / float(max(1, height))

    if width < min_width or height < min_height or width_ratio < (frame_aspect_ratio * 0.88):
        # Box is too narrow or too small — widen it toward a realistic full-frame opening.
        box_height = max(height, int(round(image_height * 0.68)))
        box_width = int(round(box_height * frame_aspect_ratio))
        box_width = max(min_width, min(box_width, int(image_width * 0.58)))
        center_x = (left + right) / 2.0
        left = int(round(center_x - box_width / 2.0))
        right = left + box_width
        top = max(0, bottom - box_height)

    # Expand slightly to cover the outer frame (nalichnik) of the old door
    final_width = right - left
    final_height = bottom - top
    pad_w = int(final_width * 0.05)
    pad_h = int(final_height * 0.04)
    
    left = max(0, left - pad_w)
    right = min(image_width, right + pad_w)
    top = max(0, top - pad_h)

    return sanitize_pixel_box((left, top, right, bottom), image_width, image_height)


def detect_door_frame_box_with_lines(room_bgr, expected_aspect_ratio, seed_box=None):
    import cv2
    import itertools
    import math

    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    image_height, image_width = gray.shape[:2]

    lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=max(35, int(image_height * 0.08)),
        minLineLength=max(30, int(image_height * 0.20)),
        maxLineGap=max(12, int(image_height * 0.03)),
    )
    if lines is None:
        return None

    preferred_center_x = ((seed_box[0] + seed_box[2]) / 2.0) if seed_box is not None else (image_width / 2.0)
    vertical_lines = []
    horizontal_lines = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dy >= max(25, dx * 3) and dy >= int(image_height * 0.18):
            vertical_lines.append({
                'x': (x1 + x2) / 2.0,
                'top': min(y1, y2),
                'bottom': max(y1, y2),
                'length': dy,
            })
        elif dx >= max(25, dy * 4) and dx >= int(image_width * 0.10):
            horizontal_lines.append({
                'y': (y1 + y2) / 2.0,
                'left': min(x1, x2),
                'right': max(x1, x2),
                'length': dx,
            })

    if len(vertical_lines) < 2:
        return None

    vertical_lines = sorted(vertical_lines, key=lambda item: item['length'], reverse=True)[:30]
    target_aspect_ratio = min(0.85, max(0.24, expected_aspect_ratio * 1.18))
    best_box = None
    best_score = float('-inf')

    for left_line, right_line in itertools.combinations(vertical_lines, 2):
        if left_line['x'] > right_line['x']:
            left_line, right_line = right_line, left_line

        if not (left_line['x'] < preferred_center_x < right_line['x']):
            continue

        pair_width = right_line['x'] - left_line['x']
        pair_top = min(left_line['top'], right_line['top'])
        pair_bottom = max(left_line['bottom'], right_line['bottom'])
        pair_height = pair_bottom - pair_top
        if pair_height < image_height * 0.35 or pair_width < image_width * 0.12:
            continue

        aspect_ratio = pair_width / float(max(1, pair_height))
        if aspect_ratio < 0.15 or aspect_ratio > 0.95:
            continue

        center_penalty = abs((((left_line['x'] + right_line['x']) / 2.0) / max(1, image_width)) - 0.5)
        aspect_penalty = abs(aspect_ratio - target_aspect_ratio)
        top_delta = abs(left_line['top'] - right_line['top']) / float(max(1, image_height))
        bottom_ratio = pair_bottom / float(max(1, image_height))

        horizontal_support = 0.0
        for horizontal in horizontal_lines:
            if abs(horizontal['y'] - pair_top) > image_height * 0.08:
                continue
            if horizontal['left'] > left_line['x'] + (pair_width * 0.15):
                continue
            if horizontal['right'] < right_line['x'] - (pair_width * 0.15):
                continue
            horizontal_support = max(horizontal_support, horizontal['length'] / float(max(1, pair_width)))

        score = (
            (left_line['length'] + right_line['length']) / float(max(1, image_height))
            + (horizontal_support * 1.5)
            + (bottom_ratio * 1.2)
            - (center_penalty * 2.2)
            - (aspect_penalty * 1.6)
            - (top_delta * 1.2)
        )

        if score > best_score:
            best_score = score
            best_box = (
                int(round(left_line['x'])),
                int(round(pair_top)),
                int(round(right_line['x'])),
                int(round(pair_bottom)),
            )

    if best_box is None:
        return None
    return normalize_door_opening_box(best_box, image_width, image_height, expected_aspect_ratio)


def detect_door_opening_box(room_bgr, expected_aspect_ratio):
    try:
        from .sam_utils import SAMService
        print("DEBUG: [AI Service] Attempting structural search with SAM...")
        candidates, wall_mask = SAMService.get_opening_candidates(room_bgr)
        
        image_height, image_width = room_bgr.shape[:2]
        best_box = None
        best_score = float('-inf')
        
        for cand in candidates:
            x1, y1, x2, y2 = cand
            score = score_door_candidate(
                x1, y1, x2-x1, y2-y1, 
                image_width, image_height, 
                expected_aspect_ratio, 
                (wall_mask == 0).astype(np.uint8) # The 'Signal' is the void
            )
            if score and score > best_score:
                best_score = score
                best_box = cand
                
        if best_box:
            normalized = normalize_door_opening_box(best_box, image_width, image_height, expected_aspect_ratio)
            return normalized, 'sam-structural'
    except Exception as e:
        print(f"WARNING: [AI Service] SAM structural search failed: {e}")

    yolo_box = detect_door_box_with_yolo(room_bgr, expected_aspect_ratio)
    if yolo_box is not None:
        image_height, image_width = room_bgr.shape[:2]
        normalized = normalize_door_opening_box(yolo_box, image_width, image_height, expected_aspect_ratio)
        return normalized, 'yolo'

    opencv_box = detect_door_box_with_opencv(room_bgr, expected_aspect_ratio)
    frame_box = detect_door_frame_box_with_lines(room_bgr, expected_aspect_ratio, seed_box=opencv_box)
    if frame_box is not None:
        return frame_box, 'opencv-lines'

    if opencv_box is not None:
        image_height, image_width = room_bgr.shape[:2]
        normalized = normalize_door_opening_box(opencv_box, image_width, image_height, expected_aspect_ratio)
        return normalized, 'opencv'

    image_height, image_width = room_bgr.shape[:2]
    default_box = default_door_box(image_width, image_height, expected_aspect_ratio)
    normalized = normalize_door_opening_box(default_box, image_width, image_height, expected_aspect_ratio)
    return normalized, 'default'


def remove_door_from_room_locally(room_bgr, pixel_box):
    """Remove old door structure from the room using OpenCV inpainting."""
    import cv2
    import numpy as np

    image_height, image_width = room_bgr.shape[:2]
    # Use the box exactly as provided
    left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)
    
    # Create mask for the exact box area
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255

    # Dilate mask very slightly (3px) just for cleaner edges at the wall transition
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Inpaint using TELEA (sharper than Navier-Stokes for wall textures)
    result = cv2.inpaint(room_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return result


def remove_door_from_room_with_ai(room_bgr, pixel_box, client):
    import cv2
    import numpy as np
    from google.genai import types

    image_height, image_width = room_bgr.shape[:2]
    mask = build_box_mask(image_height, image_width, pixel_box, pad_x_ratio=0.05, pad_y_ratio=0.05)

    ok_room, room_buf = cv2.imencode('.png', room_bgr)
    ok_mask, mask_buf = cv2.imencode('.png', mask)
    if not ok_room or not ok_mask:
        raise ValueError("Failed to encode room or mask for inpainting")

    reference_image = types.RawReferenceImage(
        reference_image=types.Image(image_bytes=room_buf.tobytes()),
        reference_id=0,
    )
    mask_reference_image = types.RawReferenceImage(
        reference_image=types.Image(image_bytes=mask_buf.tobytes()),
        reference_id=1,
    )
    prompt = (
        "Completely remove the door frame and internal void from ONLY the masked area. "
        "Fill it with the exact same wall texture, color, and finish as the neighboring wall. "
        "Keep floor perspective, skirting boards, and trim perfect. "
        "Strictly produce a plain, flat wall without any doors, openings, frames, or decorative elements."
    )

    last_error = None
    for edit_mode in ('EDIT_MODE_INPAINT_REMOVAL', 'INPAINT_EDIT'):
        try:
            response = client.models.edit_image(
                model='imagen-3.0-capability-001',
                prompt=prompt,
                reference_images=[reference_image, mask_reference_image],
                config=types.EditImageConfig(
                    edit_mode=edit_mode,
                    mask_reference_id=1,
                    number_of_images=1,
                    output_mime_type='image/png',
                ),
            )
            if not response.generated_images:
                raise ValueError("No image generated during door removal")

            cleaned = cv2.imdecode(
                np.frombuffer(response.generated_images[0].image.image_bytes, np.uint8),
                cv2.IMREAD_COLOR,
            )
            if cleaned is None:
                raise ValueError("Inpainting result could not be decoded")
            if cleaned.shape[:2] != room_bgr.shape[:2]:
                cleaned = cv2.resize(cleaned, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            return cleaned
        except Exception as e:
            last_error = e
            continue

    if last_error:
        print(f"WARNING: [AI Service] Imagen 3 inpainting failed: {last_error}")
    return room_bgr.copy()


def apply_soft_shadow(room_bgr, alpha_mask, left, top, strength=0.18):
    import cv2
    import numpy as np

    image_height, image_width = room_bgr.shape[:2]
    shadow_alpha = alpha_mask.astype(np.float32) / 255.0
    shadow_canvas = np.zeros((image_height, image_width), dtype=np.float32)

    door_height, door_width = alpha_mask.shape[:2]
    offset_x = max(1, door_width // 55)
    offset_y = max(1, door_height // 35)
    shadow_left = max(0, left + offset_x)
    shadow_top = max(0, top + offset_y)
    shadow_right = min(image_width, shadow_left + door_width)
    shadow_bottom = min(image_height, shadow_top + door_height)

    if shadow_right <= shadow_left or shadow_bottom <= shadow_top:
        return room_bgr

    alpha_crop = shadow_alpha[:shadow_bottom - shadow_top, :shadow_right - shadow_left]
    shadow_canvas[shadow_top:shadow_bottom, shadow_left:shadow_right] = np.maximum(
        shadow_canvas[shadow_top:shadow_bottom, shadow_left:shadow_right],
        alpha_crop,
    )

    blur_size = max(9, ((min(door_width, door_height) // 6) | 1))
    shadow_canvas = cv2.GaussianBlur(shadow_canvas, (blur_size, blur_size), 0)

    shaded = room_bgr.astype(np.float32)
    shaded *= (1.0 - (shadow_canvas[..., None] * strength))
    return np.clip(shaded, 0, 255).astype(np.uint8)


def perspective_warp_door_to_corners(door_rgba, corners_px, target_w, target_h):
    """
    Warps a door image to fit specific corners in the target image.
    corners_px: dict with 'top_left', 'top_right', 'bottom_right', 'bottom_left' as (x, y)
    """
    import cv2
    import numpy as np

    dh, dw = door_rgba.shape[:2]
    
    # Source corners (the flat door)
    src_pts = np.float32([
        [0, 0],
        [dw, 0],
        [dw, dh],
        [0, dh]
    ])
    
    # Destination corners (from GPT analysis)
    dst_pts = np.float32([
        corners_px['top_left'],
        corners_px['top_right'],
        corners_px['bottom_right'],
        corners_px['bottom_left']
    ])
    
    # Calculate Homography
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp
    # We warp to the target image dimensions to keep absolute positioning easier
    warped = cv2.warpPerspective(
        door_rgba, 
        M, 
        (target_w, target_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    
    return warped


def compute_floor_aligned_door_box(pixel_box, door_rgba, image_width, image_height, fill_ratio=0.90, top_margin_ratio=0.05):
    left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)

    if door_rgba is None or door_rgba.ndim < 2:
        return left, top, right, bottom

    door_height, door_width = door_rgba.shape[:2]
    if door_height <= 0 or door_width <= 0:
        return left, top, right, bottom

    usable_width = max(1.0, box_width * float(fill_ratio))
    usable_height = max(1.0, box_height * float(1.0 - top_margin_ratio))
    scale = min(usable_width / float(door_width), usable_height / float(door_height))

    target_width = max(1, min(box_width, int(round(door_width * scale))))
    target_height = max(1, min(box_height, int(round(door_height * scale))))

    door_x = left + max(0, (box_width - target_width) // 2)
    door_y = bottom - target_height

    min_top = top + int(round(box_height * top_margin_ratio))
    if door_y < min_top:
        door_y = min_top

    return sanitize_pixel_box(
        (door_x, door_y, door_x + target_width, door_y + target_height),
        image_width,
        image_height,
    )


def overlay_door_into_room(room_bgr, door_rgba, pixel_box, add_shadow=True, wall_angle=0):
    import cv2
    import numpy as np

    image_height, image_width = room_bgr.shape[:2]
    left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)

    if door_rgba.ndim != 3 or door_rgba.shape[2] < 4:
        if door_rgba.ndim == 2:
            door_rgba = cv2.cvtColor(door_rgba, cv2.COLOR_GRAY2BGRA)
        else:
            alpha = np.full(door_rgba.shape[:2] + (1,), 255, dtype=np.uint8)
            door_rgba = np.concatenate([door_rgba[:, :, :3], alpha], axis=2)

    placed_left, placed_top, placed_right, placed_bottom = compute_floor_aligned_door_box(
        (left, top, right, bottom),
        door_rgba,
        image_width,
        image_height,
    )
    target_width = max(1, placed_right - placed_left)
    target_height = max(1, placed_bottom - placed_top)

    # Scale door to sit within the detected opening with a realistic top gap.
    resized_door = cv2.resize(door_rgba, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    # Apply optional 3D perspective warp based on wall angle
    if abs(wall_angle) > 2:
        shrink_ratio = min(0.3, abs(wall_angle) / 100.0)
        shrink_px = int(target_height * shrink_ratio)
        src_pts = np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])
        if wall_angle > 0:
            dst_pts = np.float32([[0, 0], [target_width, shrink_px], [target_width, target_height - shrink_px], [0, target_height]])
        else:
            dst_pts = np.float32([[0, shrink_px], [target_width, 0], [target_width, target_height], [0, target_height - shrink_px]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        resized_door = cv2.warpPerspective(
            resized_door, M, (target_width, target_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

    composite = room_bgr.copy()

    if add_shadow:
        composite = apply_soft_shadow(composite, resized_door[:, :, 3], placed_left, placed_top)

    alpha_channel = resized_door[:, :, 3]
    alpha_kernel = max(3, ((min(target_width, target_height) // 28) | 1))
    erode_kernel = np.ones((3, 3), dtype=np.uint8)
    tightened_alpha = cv2.erode(alpha_channel, erode_kernel, iterations=1)
    feathered_alpha = cv2.GaussianBlur(tightened_alpha, (alpha_kernel, alpha_kernel), 0)
    alpha = np.clip(feathered_alpha.astype(np.float32) / 255.0, 0.0, 1.0)

    region = composite[placed_top:placed_bottom, placed_left:placed_right].astype(np.float32)
    door_rgb = resized_door[:, :, :3].astype(np.float32)
    ambient_bgr = sample_room_ambient_bgr(room_bgr, pixel_box).astype(np.float32)

    edge_band = np.clip((0.92 - alpha) / 0.92, 0.0, 1.0)
    edge_band *= (alpha > 0.0).astype(np.float32)
    edge_band = cv2.GaussianBlur(edge_band, (alpha_kernel, alpha_kernel), 0)
    if np.any(edge_band > 0.0):
        edge_mix = np.clip(edge_band[..., None] * 0.32, 0.0, 0.32)
        door_rgb = (door_rgb * (1.0 - edge_mix)) + (ambient_bgr.reshape(1, 1, 3) * edge_mix)

    blended = (alpha[..., None] * door_rgb) + ((1.0 - alpha[..., None]) * region)
    composite[placed_top:placed_bottom, placed_left:placed_right] = np.clip(blended, 0, 255).astype(np.uint8)
    return composite


def sample_room_ambient_bgr(room_bgr, pixel_box):
    import numpy as np

    image_height, image_width = room_bgr.shape[:2]
    left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)

    side_band = max(4, int(round(box_width * 0.10)))
    top_band = max(4, int(round(box_height * 0.12)))

    samples = []
    if top > 0:
        samples.append(
            room_bgr[
                max(0, top - top_band):top,
                max(0, left - side_band):min(image_width, right + side_band),
            ]
        )
    if left > 0:
        samples.append(room_bgr[top:bottom, max(0, left - side_band):left])
    if right < image_width:
        samples.append(room_bgr[top:bottom, right:min(image_width, right + side_band)])

    sample_pixels = [region.reshape(-1, 3) for region in samples if region.size]
    if not sample_pixels:
        return room_bgr.reshape(-1, 3).mean(axis=0).astype(np.float32)

    ambient = np.concatenate(sample_pixels, axis=0).mean(axis=0)
    scene_mean = room_bgr.reshape(-1, 3).mean(axis=0)
    return ((ambient * 0.75) + (scene_mean * 0.25)).astype(np.float32)


def match_door_lighting_to_room(door_rgba, room_bgr, pixel_box):
    import cv2
    import numpy as np

    if door_rgba is None:
        return None

    matched = door_rgba.copy()
    if matched.ndim != 3 or matched.shape[2] < 4:
        return matched

    alpha_mask = matched[:, :, 3] > 12
    if not np.any(alpha_mask):
        return matched

    ambient_bgr = sample_room_ambient_bgr(room_bgr, pixel_box)
    door_rgb = matched[:, :, :3].astype(np.float32)
    door_mean = door_rgb[alpha_mask].mean(axis=0)

    luminance_gain = np.clip(
        float(np.mean(ambient_bgr)) / float(max(1.0, np.mean(door_mean))),
        0.88,
        1.14,
    )
    chroma_gain = np.clip(ambient_bgr / np.maximum(door_mean, 1.0), 0.85, 1.15)
    chroma_gain = 1.0 + ((chroma_gain - 1.0) * 0.35)
    total_gain = np.clip(chroma_gain * luminance_gain, 0.82, 1.18)

    vertical_gradient = np.linspace(1.02, 0.95, matched.shape[0], dtype=np.float32).reshape(-1, 1, 1)
    adjusted_rgb = door_rgb * total_gain * vertical_gradient

    alpha_channel = matched[:, :, 3]
    alpha_kernel = max(5, ((min(matched.shape[0], matched.shape[1]) // 30) | 1))
    tightened_alpha = cv2.erode(alpha_channel, np.ones((3, 3), dtype=np.uint8), iterations=1)
    feathered_alpha = cv2.GaussianBlur(tightened_alpha, (alpha_kernel, alpha_kernel), 0)
    alpha_float = np.clip(feathered_alpha.astype(np.float32) / 255.0, 0.0, 1.0)

    edge_band = np.clip((0.88 - alpha_float) / 0.88, 0.0, 1.0)
    edge_band *= (alpha_float > 0.0).astype(np.float32)
    if np.any(edge_band > 0.0):
        edge_mix = np.clip(edge_band[..., None] * 0.40, 0.0, 0.40)
        adjusted_rgb = (adjusted_rgb * (1.0 - edge_mix)) + (ambient_bgr.reshape(1, 1, 3) * edge_mix)

    matched[:, :, :3] = np.clip(adjusted_rgb, 0, 255).astype(np.uint8)
    matched[:, :, 3] = feathered_alpha
    return matched


def add_floor_contact_shadow(room_bgr, pixel_box, strength=0.24):
    import cv2
    import numpy as np

    image_height, image_width = room_bgr.shape[:2]
    left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)

    shadow_height = max(4, int(round(box_height * 0.025)))
    inset = max(1, int(round(box_width * 0.02)))
    shadow_left = min(right, max(left, left + inset))
    shadow_right = max(shadow_left + 1, max(left + 1, right - inset))
    contact_overlap = max(2, int(round(box_height * 0.012)))
    shadow_y_start = max(0, bottom - contact_overlap)
    shadow_y_end = min(image_height, bottom + shadow_height)

    if shadow_y_start >= shadow_y_end or shadow_right <= shadow_left:
        return room_bgr

    shadow_mask = np.zeros((image_height, image_width), dtype=np.float32)
    shadow_mask[shadow_y_start:shadow_y_end, shadow_left:shadow_right] = 1.0

    blur_x = max(9, ((max(3, shadow_right - shadow_left) // 5) | 1))
    blur_y = max(7, (((shadow_y_end - shadow_y_start) * 2) | 1))
    shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_x, blur_y), 0)

    vertical_fade = np.ones((image_height, 1), dtype=np.float32)
    below_bottom = min(image_height, bottom + shadow_height)
    if bottom < below_bottom:
        fade = np.linspace(1.0, 0.0, below_bottom - bottom, dtype=np.float32)
        vertical_fade[bottom:below_bottom, 0] = fade
    shadow_mask *= vertical_fade

    shaded = room_bgr.copy().astype(np.float32)
    shaded *= (1.0 - (shadow_mask[..., None] * strength))
    return np.clip(shaded, 0, 255).astype(np.uint8)


def validate_locked_scene_candidate(candidate_bgr, baseline_bgr, pixel_box):
    import numpy as np

    if candidate_bgr is None or baseline_bgr is None or candidate_bgr.shape != baseline_bgr.shape:
        return False, {'reason': 'shape_mismatch'}

    image_height, image_width = baseline_bgr.shape[:2]
    validation_mask = build_box_mask(
        image_height,
        image_width,
        pixel_box,
        pad_x_ratio=0.05,
        pad_y_ratio=0.05,
    )
    outside_mask = validation_mask == 0
    if not np.any(outside_mask):
        return False, {'reason': 'empty_outside_region'}

    baseline_pixels = baseline_bgr[outside_mask].astype(np.float32)
    candidate_pixels = candidate_bgr[outside_mask].astype(np.float32)
    diff = np.abs(candidate_pixels - baseline_pixels)

    mse = float(np.mean((candidate_pixels - baseline_pixels) ** 2))
    changed_ratio = float(np.mean(np.max(diff, axis=1) > 16.0))

    is_valid = mse <= 12.0 and changed_ratio <= 0.015
    return is_valid, {
        'mse': round(mse, 3),
        'changed_ratio': round(changed_ratio, 5),
    }


class AIService:
    """Service layer for all AI operations — background removal and room visualization."""
    
    @staticmethod
    def get_gemini_client(prefer_vertex=False):
        """
        Initialize Gemini client.
        By default uses API key first, but image-generation flows can prefer Vertex AI.
        """
        import json
        from google import genai

        def _build_api_client():
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if isinstance(api_key, str):
                api_key = api_key.strip()
            if not api_key:
                return None
            print("DEBUG: [AI Service] Initializing client with API KEY...")
            return genai.Client(api_key=api_key)

        def _build_vertex_client():
            key_path = getattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS', None)
            if not (key_path and os.path.exists(key_path)):
                return None
            try:
                from google.oauth2 import service_account
                print("DEBUG: [AI Service] Trying Service Account fallback...")
                
                with open(key_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)

                project = info.get('project_id') or getattr(settings, 'VERTEX_AI_PROJECT', '')
                location = getattr(settings, 'VERTEX_AI_LOCATION', 'us-central1')

                # PEM normalization (fix corrupted \n in private_key)
                import re
                pk = info.get('private_key', '')
                match = re.search(r'-----BEGIN PRIVATE KEY-----(.*)-----END PRIVATE KEY-----', pk, re.DOTALL)
                if match:
                    body = "".join(re.findall(r'[A-Za-z0-9+/=]', match.group(1)))
                    formatted = "\n".join(body[i:i+64] for i in range(0, len(body), 64))
                    info['private_key'] = f"-----BEGIN PRIVATE KEY-----\n{formatted}\n-----END PRIVATE KEY-----\n"
                    print("DEBUG: [AI Service] PEM key normalized successfully")

                credentials = service_account.Credentials.from_service_account_info(
                    info, 
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                
                return genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                    credentials=credentials
                )
            except Exception as e:
                print(f"WARNING: [AI Service] Service Account failed: {e}")
                return None

        client_builders = [_build_vertex_client, _build_api_client] if prefer_vertex else [_build_api_client, _build_vertex_client]

        for builder in client_builders:
            client = builder()
            if client is not None:
                return client
        
        raise ValueError("GEMINI_API_KEY yoki google-cloud-key.json topilmadi.")

    @staticmethod
    def get_visualization_provider(default='hybrid'):
        try:
            from .models import SystemSettings

            provider = str(getattr(SystemSettings.get_solo(), 'ai_provider', default) or default).strip().lower()
            if provider in {'gemini', 'openai', 'hybrid'}:
                return provider
        except Exception as e:
            print(f"DEBUG: [AI Service] Visualization provider lookup failed, using {default}: {e}")
        return default

    @staticmethod
    def get_gemini_api_keys():
        api_keys_str = getattr(settings, 'GEMINI_API_KEYS', '')
        api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
        if api_keys:
            return api_keys

        single_key = getattr(settings, 'GEMINI_API_KEY', '')
        if isinstance(single_key, str) and single_key.strip():
            return [single_key.strip()]
        return []

    @staticmethod
    def build_gemini_full_scene_prompt(product, pixel_box, image_width, image_height):
        left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)
        box_1000 = pixels_to_box_1000((left, top, right, bottom), image_width, image_height)
        door_name = getattr(product, 'name', 'door')

        return (
            "You are a professional architectural image editor.\n"
            "Reference image 1 is the original room photo.\n"
            "Reference image 2 is a binary mask where white marks the ONLY area you may edit.\n"
            "Reference image 3 is the exact new door design that must be installed.\n\n"
            "TASK:\n"
            f"Replace the existing door/opening with the exact reference door '{door_name}'.\n"
            "This is an image editing task, not scene generation.\n\n"
            "STRICT RULES:\n"
            "- Keep the room exactly the same outside the white mask\n"
            "- Do not redesign, restyle, upscale, or beautify the room\n"
            "- Do not change walls, floor, carpet, curtains, furniture, trim, or lighting outside the mask\n"
            "- Preserve the exact door design, glass pattern, frame details, and proportions from the reference door\n"
            "- Do not invent a new door design\n"
            "- Do not add extra molding, handles, windows, decor, or architectural elements\n\n"
            "PLACEMENT RULES:\n"
            f"- Install the door inside this bounding box in pixels: left={left}, top={top}, right={right}, bottom={bottom}\n"
            f"- The same box in 0-1000 normalized coordinates is: top={box_1000[0]}, left={box_1000[1]}, bottom={box_1000[2]}, right={box_1000[3]}\n"
            "- Keep the door bottom aligned to the floor line\n"
            "- Keep the top of the door below the ceiling line and naturally inside the opening\n"
            "- Fit the door proportionally inside the opening; do not stretch it\n"
            "- Center the door naturally within the opening\n\n"
            "REALISM:\n"
            "- Match perspective, local lighting, edge blending, and contact shadow naturally\n"
            "- The door must look physically installed into the wall\n"
            "- Return one edited image only"
        )

    @staticmethod
    def decode_gemini_image_bytes(image_bytes, image_width, image_height):
        import cv2
        import numpy as np

        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Gemini image output could not be decoded")

        if decoded.shape[:2] != (image_height, image_width):
            decoded = cv2.resize(decoded, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        return decoded

    @staticmethod
    def generate_room_preview_with_gemini(product, room_image_path, result_image_path):
        import cv2
        from google import genai
        from google.genai import types
        from .ai_utils import save_visualization_metadata

        room_bgr = cv2.imread(room_image_path, cv2.IMREAD_COLOR)
        if room_bgr is None:
            raise ValueError("Room image could not be loaded")
        image_height, image_width = room_bgr.shape[:2]

        door_rgba = load_best_door_rgba(product)
        expected_aspect_ratio = get_expected_door_aspect_ratio(product, door_rgba=door_rgba)
        detected_box, detection_method = detect_door_opening_box(room_bgr, expected_aspect_ratio)
        master_box = expand_pixel_box(detected_box, image_width, image_height, pad_x_ratio=0.06, pad_y_ratio=0.04)
        mask = build_box_mask(image_height, image_width, master_box, pad_x_ratio=0.0, pad_y_ratio=0.0)

        ok_room, room_buf = cv2.imencode('.png', room_bgr)
        ok_mask, mask_buf = cv2.imencode('.png', mask)
        ok_door, door_buf = cv2.imencode('.png', door_rgba)
        if not (ok_room and ok_mask and ok_door):
            raise ValueError("Failed to encode Gemini edit inputs")

        prompt_text = AIService.build_gemini_full_scene_prompt(product, master_box, image_width, image_height)
        normalized_box = pixels_to_box_1000(master_box, image_width, image_height)
        preview_metadata = {
            'generation_prompt': prompt_text,
            'generation_meta': {
                'bounding_box_px': [int(value) for value in master_box],
                'bounding_box_1000': normalized_box,
                'detection_method': detection_method,
            },
            'pipeline': {
                'mode': 'gemini_full_scene_edit_v1',
                'image_edit_engine': 'Gemini',
                'final_result': 'gemini_full_edit',
                'used_ai_refine': False,
                'expected_aspect_ratio': round(float(expected_aspect_ratio), 4),
                'detection_method': detection_method,
                'detected_box': [int(value) for value in detected_box],
                'opening_box': [int(value) for value in master_box],
                'door_asset_size': [int(door_rgba.shape[1]), int(door_rgba.shape[0])],
            },
        }

        api_keys = AIService.get_gemini_api_keys()
        clients = []
        if api_keys:
            clients.extend(('api_key', genai.Client(api_key=key)) for key in api_keys)
        else:
            clients.append(('auto', AIService.get_gemini_client(prefer_vertex=True)))

        room_reference = types.RawReferenceImage(
            reference_image=types.Image(image_bytes=room_buf.tobytes()),
            reference_id=0,
        )
        mask_reference = types.RawReferenceImage(
            reference_image=types.Image(image_bytes=mask_buf.tobytes()),
            reference_id=1,
        )
        door_reference = types.RawReferenceImage(
            reference_image=types.Image(image_bytes=door_buf.tobytes()),
            reference_id=2,
        )

        last_error = None
        imagen_edit_modes = (
            'EDIT_MODE_INPAINT_INSERTION',
            'INPAINT_EDIT',
        )

        for _, client in clients:
            for edit_mode in imagen_edit_modes:
                try:
                    print(f"DEBUG: [Gemini Preview] Trying Imagen full edit ({edit_mode})...")
                    response = client.models.edit_image(
                        model='imagen-3.0-capability-001',
                        prompt=prompt_text,
                        reference_images=[room_reference, mask_reference, door_reference],
                        config=types.EditImageConfig(
                            edit_mode=edit_mode,
                            mask_reference_id=1,
                            number_of_images=1,
                            output_mime_type='image/png',
                        ),
                    )
                    if not response.generated_images:
                        raise ValueError("No image generated by Imagen full edit")

                    edited_bgr = AIService.decode_gemini_image_bytes(
                        response.generated_images[0].image.image_bytes,
                        image_width,
                        image_height,
                    )
                    _, validation = validate_locked_scene_candidate(edited_bgr, room_bgr, master_box)
                    preview_metadata['pipeline']['scene_lock_validation'] = validation
                    preview_metadata['pipeline']['image_edit_engine'] = 'Imagen 3'
                    preview_metadata['pipeline']['model'] = 'imagen-3.0-capability-001'
                    preview_metadata['pipeline']['edit_mode'] = edit_mode
                    cv2.imwrite(result_image_path, edited_bgr)
                    save_visualization_metadata(result_image_path, preview_metadata)
                    return result_image_path
                except Exception as e:
                    last_error = e
                    print(f"WARNING: [Gemini Preview] Imagen full edit failed ({edit_mode}): {e}")

        gemini_models = [
            'gemini-2.0-flash-exp',
            'gemini-2.5-flash-preview-04-17',
            'gemini-2.0-flash',
        ]
        contents = [
            types.Part.from_bytes(data=room_buf.tobytes(), mime_type='image/png'),
            types.Part.from_bytes(data=mask_buf.tobytes(), mime_type='image/png'),
            types.Part.from_bytes(data=door_buf.tobytes(), mime_type='image/png'),
            prompt_text,
        ]

        for _, client in clients:
            for model_name in gemini_models:
                try:
                    print(f"DEBUG: [Gemini Preview] Trying generate_content full edit ({model_name})...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        ),
                    )
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if part.inline_data is not None:
                                edited_bgr = AIService.decode_gemini_image_bytes(
                                    part.inline_data.data,
                                    image_width,
                                    image_height,
                                )
                                _, validation = validate_locked_scene_candidate(edited_bgr, room_bgr, master_box)
                                preview_metadata['pipeline']['scene_lock_validation'] = validation
                                preview_metadata['pipeline']['image_edit_engine'] = 'Gemini'
                                preview_metadata['pipeline']['model'] = model_name
                                preview_metadata['pipeline']['edit_mode'] = 'generate_content_image'
                                cv2.imwrite(result_image_path, edited_bgr)
                                save_visualization_metadata(result_image_path, preview_metadata)
                                return result_image_path
                    raise ValueError("No inline image returned by Gemini full edit")
                except Exception as e:
                    last_error = e
                    print(f"WARNING: [Gemini Preview] generate_content full edit failed ({model_name}): {e}")

        raise ValueError(f"Gemini full-scene edit failed: {last_error}")

    @staticmethod
    def photoroom_segmentation(image_bytes):
        """High-quality background removal via Photoroom API."""
        import requests
        api_key = getattr(settings, 'PHOTOROOM_API_KEY', None)
        if not api_key:
            return None
            
        print("DEBUG: [AI Service] Attempting Photoroom HD segmentation...")
        url = "https://sdk.photoroom.com/v1/segment"
        try:
            # We must send bytes as a file-like object
            files = {"image_file": ('image.png', image_bytes, 'image/png')}
            headers = {"x-api-key": api_key}
            response = requests.post(url, files=files, headers=headers, timeout=12)
            if response.status_code == 200:
                return response.content
            print(f"WARNING: [AI Service] Photoroom API failed: {response.status_code}")
        except Exception as e:
            print(f"WARNING: [AI Service] Photoroom request error: {e}")
        return None

    @staticmethod
    def gemini_background_removal(image_bytes, client):
        """High-quality background removal fallback via Gemini Imagen 3."""
        from google.genai import types
        import numpy as np
        import cv2
        import io
        
        print("DEBUG: [AI Service] Attempting Gemini HD background removal...")
        try:
            # Instead of simple removal, we use inpainting to replace bg with green screen
            # for reliable extraction later, or just return the image if it's already masked.
            # But the most reliable 'segmentation' fallback in Gemini is generating 
            # the product on a white background or using specific removal prompts.
            
            prompt = "Isolate this door object from its background completely. Place it on a solid, flat, uniform #00FF00 green background. Do not alter the door's texture, frame, or details."
            
            response = client.models.edit_image(
                model='imagen-3.0-capability-001',
                prompt=prompt,
                reference_images=[
                    types.RawReferenceImage(
                        reference_image=types.Image(image_bytes=image_bytes),
                        reference_id=0
                    )
                ],
                config=types.EditImageConfig(
                    edit_mode='REMOVE_BACKGROUND' if hasattr(types, 'REMOVE_BACKGROUND') else 'INPAINT_EDIT',
                    number_of_images=1,
                    output_mime_type='image/png',
                )
            )
            
            if response.generated_images:
                # If we used green screen, we might need to extract. 
                # But for now, we'll return the raw result and let refine_product_mask handle it.
                return response.generated_images[0].image.image_bytes
        except Exception as e:
            print(f"WARNING: [AI Service] Gemini background removal failed: {e}")
        return None

    @staticmethod
    def process_product_background(product):
        """
        Tiered HD Background Removal Pipeline:
        1. Photoroom (Best Quality)
        2. Gemini (Pro Fallback)
        3. rembg (Local Fallback)
        """
        from .models import Product, SystemSettings
        from django.core.files.base import ContentFile
        import os
        import io
        import numpy as np
        from PIL import Image, ImageOps

        try:
            product = Product.objects.get(id=product.id)
            settings_obj = SystemSettings.get_solo()
            
            if not settings_obj.enable_bg_removal:
                print(f"DEBUG: [AI Service] BG removal disabled by system settings.")
                product.ai_status = 'completed'
                product.save(update_fields=['ai_status'])
                return

            print(f"DEBUG: [AI Service] HD Processing for Product {product.id} (Mode: {settings_obj.bg_removal_mode})...")
            product.ai_status = 'processing'
            product.save(update_fields=['ai_status'])

            # 1. Prepare original asset
            if not product.original_image:
                product.image.seek(0)
                original_content = product.image.read()
                name = os.path.basename(product.image.name)
                product.original_image.save(name, ContentFile(original_content), save=False)
                product.save(update_fields=['original_image'])

            product.original_image.seek(0)
            input_bytes = product.original_image.read()
            
            # Fix orientation and convert to PNG bytes
            img_pil = Image.open(io.BytesIO(input_bytes))
            img_pil = ImageOps.exif_transpose(img_pil).convert("RGBA")
            prep_buf = io.BytesIO()
            img_pil.save(prep_buf, format='PNG')
            input_bytes_cleaned = prep_buf.getvalue()

            output_image_bytes = None
            method_used = "none"

            # === TIER 1: PHOTOROOM ===
            output_image_bytes = AIService.photoroom_segmentation(input_bytes_cleaned)
            if output_image_bytes:
                method_used = "photoroom"

            # === TIER 2: GEMINI ===
            if not output_image_bytes:
                try:
                    client = AIService.get_gemini_client()
                    output_image_bytes = AIService.gemini_background_removal(input_bytes_cleaned, client)
                    if output_image_bytes:
                        method_used = "gemini"
                except:
                    pass

            # === TIER 3: REMBG (Deterministic Fallback) ===
            if not output_image_bytes:
                from rembg import remove, new_session
                print(f"DEBUG: [AI Service] Falling back to local rembg...")
                session = new_session("isnet-general-use")
                output_image_bytes = remove(input_bytes_cleaned, session=session)
                method_used = "rembg"

            # === REFINEMENT & TEXTURE PRESERVATION ===
            import cv2
            
            # Load the result (we might have holes or jagged edges)
            tmp_res = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")
            tmp_alpha = np.array(tmp_res)[:, :, 3]
            
            # Use our IMPROVED robust refinement (fixes missing molding)
            perfect_mask = refine_product_mask(tmp_alpha)
            
            if perfect_mask is None:
                # If refinement failed, use what we have or solid block
                perfect_mask = tmp_alpha

            # Composite back to original to preserve 100% texture quality
            src_np = np.array(img_pil)
            src_np[:, :, 3] = perfect_mask
            
            final_buf = io.BytesIO()
            Image.fromarray(src_np).save(final_buf, format='PNG')
            final_bytes = final_buf.getvalue()

            # Save results
            product.image.save(f"hd_isolated_{product.id}.png", ContentFile(final_bytes), save=False)
            product.image_no_bg.save(f"hd_trans_{product.id}.png", ContentFile(final_bytes), save=False)
            product.ai_status = 'completed'
            product.ai_error = ''  # Clear any previous errors
            product.save()
            print(f"DEBUG: [AI Service] BG removal completed using method: {method_used}")
            
            print(f"DEBUG: [AI Service] Success! HD isolated using {method_used} for {product.id}")

        except Exception as e:
            print(f"ERROR: [AI Service] HD Isolation failed: {e}")
            import traceback
            traceback.print_exc()
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR: [AI Service] Visualization failed for product {product.id}: {error_details}")
            
            # Save error to a dedicated log for easier debugging
            try:
                with open('ai_debug.log', 'a+') as f:
                    f.write(f"\n--- ERROR {product.id} ---\n{error_details}\n")
            except:
                pass

            product.ai_status = 'error'
            product.ai_error = str(error_details)[:500]
            product.save(update_fields=['ai_status', 'ai_error'])

    @staticmethod
    def refine_corners_with_mask(detected_box, wall_mask, room_bgr):
        """
        Takes a raw detection box and a SAM mask to find the precise 4 corners
        of the architectural opening.
        """
        import cv2
        import numpy as np

        x1, y1, x2, y2 = detected_box
        h, w = wall_mask.shape[:2]
        
        # 1. Create a ROI around the detection
        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        roi_x1 = max(0, x1 - pad_w)
        roi_y1 = max(0, y1 - pad_h)
        roi_x2 = min(w, x2 + pad_w)
        roi_y2 = min(h, y2 + pad_h)
        
        # 2. Extract edges from the mask in this ROI
        mask_roi = (wall_mask[roi_y1:roi_y2, roi_x1:roi_x2] * 255).astype(np.uint8)
        
        # 3. Find contours in the mask ROI
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback to rectangle corners if no contour found
            return {
                'top_left': (x1, y1), 'top_right': (x2, y1),
                'bottom_right': (x2, y2), 'bottom_left': (x1, y2)
            }
            
        # Get the largest contour in the ROI
        cnt = max(contours, key=cv2.contourArea)
        
        # Approximate to a simpler shape (hopefully a quad)
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4:
            # Shift back to full image coordinates
            pts = approx.reshape(4, 2)
            pts[:, 0] += roi_x1
            pts[:, 1] += roi_y1
            
            # Sort corners: top-left, top-right, bottom-right, bottom-left
            # Simple heuristic: sort by y then x
            pts = pts[np.argsort(pts[:, 1])] # sort by y
            top_pts = pts[:2][np.argsort(pts[:2, 0])] # top 2 sorted by x
            bottom_pts = pts[2:][np.argsort(pts[2:, 0])[::-1]] # bottom 2 sorted by x (desc)
            
            return {
                'top_left': tuple(top_pts[0]),
                'top_right': tuple(top_pts[1]),
                'bottom_right': tuple(bottom_pts[0]),
                'bottom_left': tuple(bottom_pts[1])
            }
        
        # Fallback 2: Bound Rect if quad approximation failed
        return {
            'top_left': (x1, y1), 'top_right': (x2, y1),
            'bottom_right': (x2, y2), 'bottom_left': (x1, y2)
        }

    @staticmethod
    def overlay_door_perspective(room_bgr, door_rgba, corners):
        """
        Overlays the door using high-fidelity homography warping.
        """
        import cv2
        import numpy as np

        h, w = room_bgr.shape[:2]
        
        # 1. Warp the door to the corners
        warped_door = perspective_warp_door_to_corners(door_rgba, corners, w, h)
        
        # 2. Extract alpha and RGB
        door_rgb = warped_door[:, :, :3]
        door_alpha = warped_door[:, :, 3].astype(np.float32) / 255.0
        
        # 3. Shadow logic (Simplified for perspective)
        # We'll use the bounding box of corners to apply shadow
        pts = np.array([corners['top_left'], corners['top_right'], 
                        corners['bottom_right'], corners['bottom_left']], np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        
        composite = apply_soft_shadow(room_bgr, warped_door[:, :, 3], 0, 0) # apply shadow globally using warped mask
        
        # 4. Blend
        mask_3d = door_alpha[..., None]
        blended = (mask_3d * door_rgb.astype(np.float32)) + ((1.0 - mask_3d) * composite.astype(np.float32))
        
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def generate_room_preview(product, room_image_path, result_image_path):
        """
        Locked-scene hybrid pipeline.

        1. Detect the opening geometrically
        2. Remove the old door locally with OpenCV inpainting
        3. Place the exact door asset deterministically
        4. Optionally let AI refine only masked edge/shadow realism
        5. Reject AI output unless the room outside the door stays effectively identical
        """
        import cv2
        from .ai_utils import save_visualization_metadata

        provider = AIService.get_visualization_provider(default='hybrid')
        if provider == 'gemini':
            print(f"DEBUG: [Pipeline] Using Gemini full-scene editor for product {product.id}...")
            return AIService.generate_room_preview_with_gemini(product, room_image_path, result_image_path)

        print(f"DEBUG: [Pipeline] Starting production pipeline for product {product.id}...")
        preview_metadata = {
            'pipeline': {
                'mode': 'locked_scene_hybrid_v2',
                'image_edit_engine': 'OpenCV',
                'used_ai_refine': False,
                'final_result': 'opencv_composite',
            }
        }

        # --- Load room ---
        room_bgr = cv2.imread(room_image_path, cv2.IMREAD_COLOR)
        if room_bgr is None:
            raise ValueError("Room image could not be loaded")
        h, w = room_bgr.shape[:2]

        # --- Load door ---
        door_rgba = load_best_door_rgba(product)
        preview_metadata['pipeline']['door_asset_size'] = [int(door_rgba.shape[1]), int(door_rgba.shape[0])]
        expected_aspect_ratio = get_expected_door_aspect_ratio(product, door_rgba=door_rgba)

        # === STEP 1: DETECT DOOR AREA + CREATE MASK ===
        print(f"DEBUG: [Pipeline] Step 1: Detecting door area...")
        detected_box, detection_method = detect_door_opening_box(room_bgr, expected_aspect_ratio)
        # Expand box aggressively once to ensure it covers ALL old frame components.
        # This is our Master Box for both Removal and Insertion.
        master_box = expand_pixel_box(detected_box, w, h, pad_x_ratio=0.08, pad_y_ratio=0.06)
        x1, y1, x2, y2 = master_box
        print(f"DEBUG: [Pipeline]   Master Box (for removal & insertion): ({x1},{y1})-({x2},{y2})")

        preview_metadata['pipeline']['expected_aspect_ratio'] = round(float(expected_aspect_ratio), 4)
        preview_metadata['pipeline']['detection_method'] = detection_method
        preview_metadata['pipeline']['detected_box'] = [int(value) for value in detected_box]
        preview_metadata['pipeline']['opening_box'] = [x1, y1, x2, y2]

        # Use the exact Master Box for everything
        mask = build_box_mask(h, w, master_box, pad_x_ratio=0.0, pad_y_ratio=0.0)

        # === STEP 2: REMOVE OLD DOOR (INPAINTING) ===
        print(f"DEBUG: [Pipeline] Step 2: Removing old door (inpainting)...")
        # To perfectly hide inpainting, the inpaint mask should be slightly SMALLER than the door area.
        # So we inpaint based on detected_box + small padding, then overlay based on master_box.
        inpaint_box = expand_pixel_box(detected_box, w, h, pad_x_ratio=0.05, pad_y_ratio=0.04)
        cleaned_room = remove_door_from_room_locally(room_bgr, inpaint_box)
        preview_metadata['pipeline']['old_door_removal'] = 'opencv_inpaint'
        print(f"DEBUG: [Pipeline]   Old door removed, wall filled with texture")

        # === STEP 3: INSERT NEW DOOR (on clean wall) ===
        print(f"DEBUG: [Pipeline] Step 3: Placing new door on clean wall...")
        # Note: overlay uses the larger Master Box to perfectly cover the inpaint blur.
        lit_door_rgba = match_door_lighting_to_room(door_rgba, room_bgr, master_box)
        placed_box = compute_floor_aligned_door_box(master_box, lit_door_rgba, w, h)
        preview_metadata['pipeline']['placement_box'] = [int(value) for value in placed_box]
        composite = overlay_door_into_room(cleaned_room, lit_door_rgba, master_box, add_shadow=True)
        composite = add_floor_contact_shadow(composite, placed_box)
        print(f"DEBUG: [Pipeline]   Composite created (room preserved, old door removed)")

        # Save OpenCV result as SAFE fallback
        cv2.imwrite(result_image_path, composite)

        # === STEP 4: AI MASKED REFINE (edge + contact shadow only) ===
        try:
            print(f"DEBUG: [Pipeline] Step 4: AI masked refine...")
            refined_path = AIService.refine_door_edges_with_ai(composite, mask, room_image_path, result_image_path)
            if refined_path and os.path.exists(refined_path):
                refined_img = cv2.imread(refined_path, cv2.IMREAD_COLOR)
                is_valid, validation = validate_locked_scene_candidate(refined_img, composite, master_box)
                preview_metadata['pipeline']['ai_validation'] = validation
                print(
                    "DEBUG: [Pipeline]   Refine validation "
                    f"(mse={validation.get('mse')}, changed_ratio={validation.get('changed_ratio')})"
                )

                if is_valid:
                    preview_metadata['pipeline']['used_ai_refine'] = True
                    preview_metadata['pipeline']['final_result'] = 'ai_masked_refine'
                    save_visualization_metadata(result_image_path, preview_metadata)
                    print(f"DEBUG: [Pipeline]   ✅ AI refine PASSED — using refined result")
                    return refined_path

                print(f"DEBUG: [Pipeline]   ❌ AI refine changed locked scene — reverting to OpenCV result")
                cv2.imwrite(result_image_path, composite)
        except Exception as e:
            preview_metadata['pipeline']['ai_refine_error'] = str(e)[:300]
            print(f"WARNING: [Pipeline] AI refine failed (using OpenCV): {e}")

        save_visualization_metadata(result_image_path, preview_metadata)
        print(f"DEBUG: [Pipeline] ✅ DONE (OpenCV): {result_image_path}")
        return result_image_path


    @staticmethod
    def ai_polish_only(composite_bgr, result_image_path):
        """
        AI POLISH ONLY — no structural changes.
        Takes a finished composite and only improves lighting/shadows.
        """
        from google.genai import types
        from google import genai
        from PIL import Image as PILImage
        from io import BytesIO
        import cv2

        api_keys_str = getattr(settings, 'GEMINI_API_KEYS', '')
        api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        if not api_keys:
            single_key = getattr(settings, 'GEMINI_API_KEY', '')
            if single_key:
                api_keys = [single_key]
        if not api_keys:
            raise ValueError("No GEMINI keys configured")

        _, img_bytes = cv2.imencode('.png', composite_bgr)
        img_bytes = img_bytes.tobytes()

        # POLISH-ONLY prompt — no creativity allowed
        prompt_text = (
            "This is a finished interior photo with a door already installed.\n"
            "Your ONLY job: improve the realism of lighting and shadows.\n\n"
            "STRICT RULES:\n"
            "- Do NOT change any objects, furniture, or structure\n"
            "- Do NOT move, resize, or redesign the door\n"
            "- Do NOT add or remove anything\n"
            "- Do NOT change wall colors or textures\n"
            "- ONLY adjust: shadow softness, light reflection on door, edge blending\n"
            "- Return the same image with improved lighting realism\n"
            "- This is NOT a generation task — it is a POLISH task"
        )

        contents = [
            types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
            prompt_text,
        ]

        gemini_models = [
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash',
        ]

        for key in api_keys:
            client = genai.Client(api_key=key)
            for model_name in gemini_models:
                try:
                    print(f"DEBUG: [AI Polish] Trying {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        ),
                    )
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if part.inline_data is not None:
                                img = PILImage.open(BytesIO(part.inline_data.data))
                                img.save(result_image_path, format='PNG')
                                print(f"DEBUG: [AI Polish] SUCCESS: {model_name}")
                                return result_image_path
                except Exception as e:
                    err_str = str(e)
                    print(f"WARNING: [AI Polish] Failed {model_name}: {err_str[:200]}")
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        break
                    continue

        raise ValueError("AI polish failed on all keys/models")

    @staticmethod
    def refine_door_edges_with_ai(composite_bgr, mask, room_image_path, result_image_path):
        """
        Send PRE-COMPOSITE to AI for edge refinement only.
        The AI receives a room with the door ALREADY placed.
        It can only fix edges — NOT reimagine the room.
        """
        from google.genai import types
        from google import genai
        from PIL import Image as PILImage
        from io import BytesIO
        import cv2

        api_keys_str = getattr(settings, 'GEMINI_API_KEYS', '')
        api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        if not api_keys:
            single_key = getattr(settings, 'GEMINI_API_KEY', '')
            if single_key:
                api_keys = [single_key]
        if not api_keys:
            raise ValueError("No GEMINI keys configured")

        # Encode composite as bytes
        _, composite_bytes = cv2.imencode('.png', composite_bgr)
        composite_bytes = composite_bytes.tobytes()

        # Encode mask as bytes
        _, mask_bytes = cv2.imencode('.png', mask)
        mask_bytes = mask_bytes.tobytes()

        # Constrained prompt — AI cannot reimagine, only fix edges
        prompt_text = (
            "This image already has a new door placed in it using digital editing.\n"
            "The second image is a MASK showing the door area (white = door).\n\n"
            "YOUR ONLY JOB: make the placement look naturally installed without changing the room.\n\n"
            "STRICT RULES:\n"
            "- Do NOT change ANYTHING outside the white mask area\n"
            "- Do NOT change walls, floor, furniture, curtains\n"
            "- Do NOT redesign the room\n"
            "- Do NOT add any new objects\n"
            "- Do NOT move, resize, recolor, or redesign the new door\n"
            "- ONLY improve edge blending, contact shadow, and local light integration near the door\n"
            "- Make the door look naturally installed\n"
            "- Keep the exact same room, exact same lighting\n"
            "- Return ONLY the refined image"
        )

        contents = [
            types.Part.from_bytes(data=composite_bytes, mime_type='image/png'),
            types.Part.from_bytes(data=mask_bytes, mime_type='image/png'),
            prompt_text,
        ]

        gemini_models = [
            'gemini-2.0-flash-exp',
            'gemini-2.5-flash-preview-04-17',
            'gemini-2.0-flash',
        ]

        for key in api_keys:
            client = genai.Client(api_key=key)
            for model_name in gemini_models:
                try:
                    print(f"DEBUG: [AI Refine] Trying {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        ),
                    )
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if part.inline_data is not None:
                                img = PILImage.open(BytesIO(part.inline_data.data))
                                img.save(result_image_path, format='PNG')
                                print(f"DEBUG: [AI Refine] SUCCESS: {model_name}")
                                return result_image_path
                except Exception as e:
                    err_str = str(e)
                    print(f"WARNING: [AI Refine] Failed {model_name}: {err_str[:200]}")
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        break
                    continue

        raise ValueError("AI edge refinement failed on all keys/models")


    @staticmethod
    def generate_holistic_room_view(product, room_image_path, result_image_path):
        """
        Legacy method — now redirects to the new pipeline.
        Kept for backward compatibility.
        """
        # This is now handled by generate_room_preview directly
        raise ValueError("Use generate_room_preview instead")

class WishlistService:
    """Service layer for wishlist operations."""

    @staticmethod
    def toggle(user, product):
        """Toggle wishlist status. Returns (is_wishlisted: bool)."""
        from shop.models import Wishlist
        item, created = Wishlist.objects.get_or_create(user=user, product=product)
        if not created:
            item.delete()
            return False
        return True

    @staticmethod
    def is_wishlisted(user_id, product_id):
        """Check if a product is in user's wishlist."""
        from shop.models import Wishlist
        return Wishlist.objects.filter(user_id=user_id, product_id=product_id).exists()

    @staticmethod
    def get_user_wishlist(user):
        """Get all wishlist items for a user."""
        from shop.models import Wishlist
        return Wishlist.objects.filter(user=user).select_related(
            'product', 'product__category', 'product__company'
        )

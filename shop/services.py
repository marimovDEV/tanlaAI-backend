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

    # Trust the detector. Just make sure the box is at least a minimum viable size.
    width = right - left
    height = bottom - top

    min_width = int(image_width * 0.15)
    min_height = int(image_height * 0.30)

    if width < min_width or height < min_height:
        # Box too small — fall back to center-based default
        box_height = int(round(image_height * 0.65))
        box_width = int(round(box_height * expected_aspect_ratio))
        box_width = max(min_width, min(box_width, int(image_width * 0.55)))
        center_x = (left + right) / 2.0
        left = int(round(center_x - box_width / 2.0))
        right = left + box_width
        top = max(0, bottom - box_height)

    # Expand slightly to cover the outer frame (nalichnik) of the old door
    final_width = right - left
    final_height = bottom - top
    pad_w = int(final_width * 0.04)
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
    """Remove old door using OpenCV inpainting — fills the area with wall texture."""
    import cv2
    import numpy as np

    result = room_bgr.copy()
    image_height, image_width = result.shape[:2]
    x1, y1, x2, y2 = sanitize_pixel_box(pixel_box, image_width, image_height)

    # Shrink inpaint area slightly to preserve wall edges/frame
    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.03)
    rx1 = max(0, x1 + pad_x)
    ry1 = max(0, y1 + pad_y)
    rx2 = min(image_width, x2 - pad_x)
    ry2 = min(image_height, y2 - pad_y)

    # Create mask for the door area
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask[ry1:ry2, rx1:rx2] = 255

    # Inpaint using TELEA algorithm (fills with surrounding texture)
    result = cv2.inpaint(result, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

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


def overlay_door_into_room(room_bgr, door_rgba, pixel_box, add_shadow=True, wall_angle=0):
    import cv2
    import numpy as np

    image_height, image_width = room_bgr.shape[:2]
    left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)
    target_width = max(1, right - left)
    target_height = max(1, bottom - top)

    if door_rgba.ndim != 3 or door_rgba.shape[2] < 4:
        if door_rgba.ndim == 2:
            door_rgba = cv2.cvtColor(door_rgba, cv2.COLOR_GRAY2BGRA)
        else:
            alpha = np.full(door_rgba.shape[:2] + (1,), 255, dtype=np.uint8)
            door_rgba = np.concatenate([door_rgba[:, :, :3], alpha], axis=2)

    # Scale door to FILL the detected box exactly — no floating
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

    alpha = resized_door[:, :, 3].astype(np.float32) / 255.0
    composite = room_bgr.copy()

    if add_shadow:
        composite = apply_soft_shadow(composite, resized_door[:, :, 3], left, top)

    region = composite[top:bottom, left:right].astype(np.float32)
    door_rgb = resized_door[:, :, :3].astype(np.float32)
    blended = (alpha[..., None] * door_rgb) + ((1.0 - alpha[..., None]) * region)
    composite[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)
    return composite


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
        Main pipeline for room visualization.
        
        TIER 1: Gemini AI holistic generation (best quality, like gemini.google.com)
        TIER 2: Surgical OpenCV overlay (fallback, keeps room 1=1)
        """
         # === TIER 1: SURGICAL OVERLAY (Primary: Keeps room 1:1) ===
        try:
            print(f"DEBUG: [AI Service] TIER 1: Attempting 1:1 Surgical Overlay for product {product.id}...")
            import cv2
            import numpy as np

            room_bgr = cv2.imread(room_image_path, cv2.IMREAD_COLOR)
            if room_bgr is None:
                raise ValueError("Room image could not be loaded")

            # Safely get door image path
            door_path = None
            if hasattr(product, 'image_no_bg') and product.image_no_bg and product.image_no_bg.name and os.path.exists(product.image_no_bg.path):
                door_path = product.image_no_bg.path
            elif hasattr(product, 'image') and product.image and product.image.name and os.path.exists(product.image.path):
                door_path = product.image.path
            elif hasattr(product, 'original_image') and product.original_image and product.original_image.name and os.path.exists(product.original_image.path):
                door_path = product.original_image.path

            if not door_path:
                raise ValueError("No valid door image found for product")

            door_rgba = cv2.imread(door_path, cv2.IMREAD_UNCHANGED)
            if door_rgba is None:
                raise ValueError("Door image could not be loaded via cv2")
            if door_rgba.ndim == 2:
                door_rgba = cv2.cvtColor(door_rgba, cv2.COLOR_GRAY2BGRA)
            elif door_rgba.shape[2] == 3:
                alpha = np.full(door_rgba.shape[:2] + (1,), 255, dtype=np.uint8)
                door_rgba = np.concatenate([door_rgba, alpha], axis=2)

            expected_aspect_ratio = get_expected_door_aspect_ratio(product, door_rgba=door_rgba)
            detected_box, detection_method = detect_door_opening_box(room_bgr, expected_aspect_ratio)
            
            # Refine scene understanding with SAM
            try:
                from .sam_utils import SAMService
                print(f"DEBUG: [AI Service] Refining scene understanding with SAM for {product.id}...")
                wall_mask = SAMService.get_wall_mask(room_bgr, hint_box=detected_box)
                corners = AIService.refine_corners_with_mask(detected_box, wall_mask, room_bgr)
                
                area = 0.5 * abs(
                    (corners['top_left'][0] * (corners['top_right'][1] - corners['bottom_left'][1]) +
                     corners['top_right'][0] * (corners['bottom_left'][1] - corners['top_left'][1]) +
                     corners['bottom_left'][0] * (corners['top_left'][1] - corners['top_right'][1]))
                )
                if area < 500:
                    raise ValueError("Refined corners produce degenerate area")
                    
                print(f"DEBUG: [AI Service] Perspective corners identified: {corners}")
                use_perspective = True
            except Exception as sam_err:
                print(f"WARNING: [AI Service] SAM refinement failed, falling back to box: {sam_err}")
                use_perspective = False

            # Remove old door
            cleaned_room = remove_door_from_room_locally(room_bgr, detected_box)

            # Overlay new door
            if use_perspective:
                final_room = AIService.overlay_door_perspective(cleaned_room, door_rgba, corners)
            else:
                final_room = overlay_door_into_room(cleaned_room, door_rgba, detected_box, add_shadow=True)

            if not cv2.imwrite(result_image_path, final_room):
                raise ValueError("Failed to save final room visualization")

            print(f"DEBUG: [AI Service] TIER 1 SUCCESS: 1:1 Surgical overlay ready: {result_image_path}")
            return result_image_path

        except Exception as surgical_err:
            print(f"WARNING: [AI Service] TIER 1 failed (Surgical): {surgical_err}")

        # === TIER 2: AI HOLISTIC GENERATION (Fallback only) ===
        try:
            print(f"DEBUG: [AI Service] TIER 2: Attempting Gemini holistic generation for product {product.id}...")
            result = AIService.generate_holistic_room_view(product, room_image_path, result_image_path)
            if result and os.path.exists(result):
                print(f"DEBUG: [AI Service] TIER 2 SUCCESS: Gemini holistic generation complete")
                return result
        except Exception as holistic_err:
            print(f"WARNING: [AI Service] TIER 2 failed (Gemini holistic): {holistic_err}")

        # === TIER 3: DALL-E 3 (Last resort) ===
        try:
            from .ai_utils import visualize_door_in_room
            print(f"DEBUG: [AI Service] TIER 3: Attempting DALL-E 3 fallback for product {product.id}...")
            result = visualize_door_in_room(product, room_image_path, result_image_path)
            if result and os.path.exists(result) and os.path.getsize(result) > 1000:
                print(f"DEBUG: [AI Service] TIER 3 SUCCESS: DALL-E 3 complete")
                return result
        except Exception as dalle_err:
            print(f"WARNING: [AI Service] TIER 3 failed (DALL-E 3): {dalle_err}")
        except Exception as error:
            print(f"ERROR: [AI Service] Room preview generation failed: {error}")
            raise


    @staticmethod
    def generate_holistic_room_view(product, room_image_path, result_image_path):
        """
        Holistic AI Reconstruction using Gemini generate_content.
        Sends room + door images to Gemini and gets back a photorealistic composite.
        """

        # === GEMINI generate_content (like gemini.google.com) ===
        try:
            from google.genai import types
            from PIL import Image as PILImage
            from io import BytesIO

            print(f"DEBUG: [AI Service] TIER 1: Gemini generate_content for product {product.id}...")
            
            # Fetch multiple keys for fallback
            api_keys_str = getattr(settings, 'GEMINI_API_KEYS', '')
            api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
            if not api_keys:
                single_key = getattr(settings, 'GEMINI_API_KEY', '')
                if single_key:
                    api_keys = [single_key]
            
            if not api_keys:
                raise ValueError("No GEMINI keys configured")

            with open(room_image_path, 'rb') as f:
                room_bytes = f.read()

            door_path = product.image.path
            if product.image_no_bg and product.image_no_bg.name and os.path.exists(product.image_no_bg.path):
                door_path = product.image_no_bg.path
            elif product.original_image and product.original_image.name and os.path.exists(product.original_image.path):
                door_path = product.original_image.path

            with open(door_path, 'rb') as f:
                door_bytes = f.read()

            room_mime = 'image/png' if room_image_path.lower().endswith('.png') else 'image/jpeg'
            door_mime = 'image/png' if door_path.lower().endswith('.png') else 'image/jpeg'

            prompt_text = (
                "Birinchi rasmda xona ko'rsatilgan, ikkinchi rasmda eshik ko'rsatilgan. "
                "Shu eshikni xonadagi mavjud eshik o'rniga qo'yib ber. "
                "Muhim qoidalar: "
                "1. Xonaning devori, poli, gilamlari, yorug'ligi, ranglari AYNAN o'zgarishsiz qolsin. "
                "2. Eshik xonadagi eski eshik joylashgan AYNAN shu joyga o'rnatilsin. "
                "3. Eshikning o'lchami xonadagi eshik o'lchamiga mos bo'lsin. "
                "4. Eshik perspektivasi (burchagi) xona perspektivasiga mos bo'lsin. "
                "5. Natija fotorealistik bo'lsin, sun'iy ko'rinmasin. "
                "6. Faqat bitta rasm qaytar — xona + yangi eshik."
            )

            contents = [
                types.Part.from_bytes(data=room_bytes, mime_type=room_mime),
                types.Part.from_bytes(data=door_bytes, mime_type=door_mime),
                prompt_text,
            ]

            gemini_models = [
                'gemini-3.1-flash-image-preview',# Nano Banana 2 (High Speed, used in Gemini Web App)
                'gemini-3-pro-image-preview',    # Nano Banana Pro (Highest Quality)
                'gemini-2.5-flash-image',        # Standard
            ]
            
            from google import genai
            
            for key in api_keys:
                print(f"DEBUG: [AI Service] Using Nano Banana Engine with key {key[:10]}...")
                client = genai.Client(api_key=key)
                
                # Real-World 'Kirish Eshigi' Methodology (User's v4 - Anti-Fantasy)
                prompt_text = (
                    "STRICT REAL-WORLD VISUALIZATION INSTRUCTIONS:\n"
                    "1. DESIGN FIDELITY: Maintain the EXACT 'kirish eshigi' design and solid patterns of the CHOSEN DOOR IMAGE. DO NOT create a full glass or balcony door.\n"
                    "2. CONTEXT: Create a realistic entry door that opens into a dark hallway or inner space. AVOID large balconies or strong backlighting.\n"
                    "3. LIGHTING: Use very SUBTLE lighting from the hallway behind. Maintain the original dark glass and gold-etched details with grounded realism.\n"
                    "4. ARCHITECTURAL BLEND: Seamlessly blend the door frame with the existing ornate cornice and room structure, ensuring it is flush with the floor.\n"
                    "5. PRESERVATION: Keep all foreground elements (carpet, curtains, radiator, pillows, furniture) from the original room 100% UNCHANGED.\n"
                    "6. ANTI-FANTASY: Strictly avoid any fantastical elements, glows, or unrealistic lighting effects. The result must be 100% realistic."
                )
                
                contents = [
                    types.Part.from_bytes(data=room_bytes, mime_type=room_mime),
                    types.Part.from_bytes(data=door_bytes, mime_type=door_mime),
                    prompt_text,
                ]

                for model_name in gemini_models:
                    try:
                        print(f"DEBUG: [AI Service]   Requesting Professional Generation ({model_name})...")
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
                                    print(f"DEBUG: [AI Service] Professional Success: {model_name} on key {key[:10]}...")
                                    return result_image_path
                                elif part.text:
                                    print(f"DEBUG: [AI Service]   Model returned text instead of image: {part.text[:100]}...")
                    except Exception as e:
                        err_str = str(e)
                        print(f"WARNING: [AI Service]   Generation failed for {model_name}: {err_str[:200]}")
                        
                        # Stop iterating models if quota exhausted, move onto next key
                        if "429 RESOURCE_EXHAUSTED" in err_str or "exceeded your current quota" in err_str:
                            print(f"DEBUG: [AI Service]   Quota exhausted for key {key[:10]}, skipping to next key...")
                            break # Move to next key immediately
                        continue # Try next model if it wasn't a quota error

        except Exception as gemini_err:
            print(f"WARNING: [AI Service] Gemini tier failed completely: {gemini_err}")

        raise ValueError("All holistic reconstruction methods failed (DALL-E 3 + Gemini)")

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

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

    clean = np.where(mask > 10, 255, 0).astype(np.uint8)
    if not np.any(clean):
        return None

    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if num_labels <= 1:
        return clean

    height, width = clean.shape[:2]
    image_center = (width / 2.0, height / 2.0)
    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area <= 0:
            continue

        component_center = (x + (w / 2.0), y + (h / 2.0))
        distance = ((component_center[0] - image_center[0]) ** 2 + (component_center[1] - image_center[1]) ** 2) ** 0.5
        score = float(area) - (distance * 2.0)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return clean

    clean = np.where(labels == best_label, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return clean

    filled = np.zeros_like(clean)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
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

    if area_ratio < 0.03 or area_ratio > 0.80:
        return None
    if height_ratio < 0.35 or width_ratio < 0.10:
        return None
    if aspect_ratio < 0.15 or aspect_ratio > 0.95:
        return None

    center_penalty = abs(((x + (candidate_width / 2.0)) / max(1, image_width)) - 0.5)
    bottom_ratio = (y + candidate_height) / float(max(1, image_height))
    if bottom_ratio < 0.55:
        return None

    aspect_penalty = abs(aspect_ratio - expected_aspect_ratio)
    region = signal_mask[y:y + candidate_height, x:x + candidate_width]
    edge_density = float(np.count_nonzero(region)) / float(max(1, area))

    return (
        (height_ratio * 4.0)
        + (area_ratio * 3.0)
        + (bottom_ratio * 1.5)
        + (edge_density * 3.0)
        - (center_penalty * 2.5)
        - (aspect_penalty * 1.5)
    )


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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    combined = cv2.bitwise_or(edges, adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.dilate(combined, kernel, iterations=1)

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
    box_height = int(round(image_height * 0.72))
    box_width = int(round(box_height * expected_aspect_ratio))
    box_width = max(int(image_width * 0.18), min(box_width, int(image_width * 0.55)))

    left = int(round((image_width - box_width) / 2.0))
    top = int(round(image_height * 0.14))
    return sanitize_pixel_box((left, top, left + box_width, top + box_height), image_width, image_height)


def normalize_door_opening_box(pixel_box, image_width, image_height, expected_aspect_ratio):
    left, top, right, bottom = sanitize_pixel_box(pixel_box, image_width, image_height)
    width = right - left
    height = bottom - top

    target_aspect_ratio = min(0.85, max(0.24, expected_aspect_ratio * 1.18))
    min_height = int(round(image_height * 0.48))
    desired_height = max(height, min_height)
    desired_width = max(width, int(round(desired_height * target_aspect_ratio)))
    desired_width = min(desired_width, int(round(image_width * 0.72)))

    center_x = (left + right) / 2.0
    bottom = min(image_height, bottom + max(2, int(round(desired_height * 0.03))))
    top = max(0, int(round(bottom - desired_height)))
    left = int(round(center_x - (desired_width / 2.0)))
    right = int(round(center_x + (desired_width / 2.0)))
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
    import cv2

    image_height, image_width = room_bgr.shape[:2]
    mask = build_box_mask(image_height, image_width, pixel_box, pad_x_ratio=0.10, pad_y_ratio=0.06)
    return cv2.inpaint(room_bgr, mask, 7, cv2.INPAINT_TELEA)


def remove_door_from_room_with_ai(room_bgr, pixel_box, client):
    import cv2
    import numpy as np
    from google.genai import types

    image_height, image_width = room_bgr.shape[:2]
    mask = build_box_mask(image_height, image_width, pixel_box, pad_x_ratio=0.12, pad_y_ratio=0.08)

    ok_room, room_buf = cv2.imencode('.png', room_bgr)
    ok_mask, mask_buf = cv2.imencode('.png', mask)
    if not ok_room or not ok_mask:
        raise ValueError("Failed to encode room or mask for inpainting")

    reference_image = types.RawReferenceImage(
        reference_image=types.Image(image_bytes=room_buf.tobytes()),
        reference_id=0,
    )
    mask_image = types.Image(image_bytes=mask_buf.tobytes())
    prompt = (
        "Remove the door and frame from the masked area and reconstruct the wall naturally. "
        "Keep wall texture, molding, trim, skirting, and floor perspective consistent. "
        "Do not add another door, doorway opening, furniture, or artifacts."
    )

    last_error = None
    for edit_mode in ('EDIT_MODE_INPAINT_REMOVAL', 'INPAINT_EDIT'):
        try:
            response = client.models.edit_image(
                model='imagen-3.0-capability-001',
                prompt=prompt,
                reference_images=[reference_image],
                config=types.EditImageConfig(
                    edit_mode=edit_mode,
                    number_of_images=1,
                    output_mime_type='image/png',
                    mask=mask_image,
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
        except Exception as exc:
            last_error = exc

    raise last_error or ValueError("AI door removal failed")


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


def overlay_door_into_room(room_bgr, door_rgba, pixel_box, add_shadow=True):
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

    door_height, door_width = door_rgba.shape[:2]
    scale = min(target_width / float(max(1, door_width)), target_height / float(max(1, door_height)))
    scale = max(scale, 1e-6)
    resized_width = max(1, int(round(door_width * scale)))
    resized_height = max(1, int(round(door_height * scale)))

    resized_door = cv2.resize(door_rgba, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    place_left = left + max(0, (target_width - resized_width) // 2)
    place_top = bottom - resized_height
    place_left, place_top, place_right, place_bottom = sanitize_pixel_box(
        (place_left, place_top, place_left + resized_width, place_top + resized_height),
        image_width,
        image_height,
    )

    actual_width = place_right - place_left
    actual_height = place_bottom - place_top
    resized_door = resized_door[:actual_height, :actual_width]
    alpha = resized_door[:, :, 3].astype(np.float32) / 255.0

    composite = room_bgr.copy()
    if add_shadow:
        composite = apply_soft_shadow(composite, resized_door[:, :, 3], place_left, place_top)

    region = composite[place_top:place_bottom, place_left:place_right].astype(np.float32)
    door_rgb = resized_door[:, :, :3].astype(np.float32)
    blended = (alpha[..., None] * door_rgb) + ((1.0 - alpha[..., None]) * region)
    composite[place_top:place_bottom, place_left:place_right] = np.clip(blended, 0, 255).astype(np.uint8)
    return composite


class AIService:
    """Service layer for all AI operations — background removal and room visualization."""
    
    @staticmethod
    def get_gemini_client():
        """
        Initialize Gemini client.
        API Key FIRST (ishlab turgan usul), Service Account fallback.
        """
        import json
        from google import genai

        # 1. API Key FIRST (user confirmed this was working)
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if isinstance(api_key, str):
            api_key = api_key.strip()
        
        if api_key:
            print("DEBUG: [AI Service] Initializing client with API KEY...")
            return genai.Client(api_key=api_key)

        # 2. Fallback: Service Account (Vertex AI)
        key_path = getattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS', None)
        if key_path and os.path.exists(key_path):
            try:
                from google.oauth2 import service_account
                print("DEBUG: [AI Service] Trying Service Account fallback...")
                project = getattr(settings, 'VERTEX_AI_PROJECT', '')
                location = getattr(settings, 'VERTEX_AI_LOCATION', 'us-central1')
                
                with open(key_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)

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
        
        raise ValueError("GEMINI_API_KEY yoki google-cloud-key.json topilmadi.")

    @staticmethod
    def process_product_background(product):
        """
        Remove the background using rembg (U2Net) and save the transparent version.
        """
        from .models import Product
        from django.core.files.base import ContentFile
        import os
        import io

        try:
            # Refresh instance
            product = Product.objects.get(id=product.id)
            if product.ai_status == 'completed':
                return

            print(f"DEBUG: [AI Service] Processing Background for Product {product.id} (rembg)...")
            product.ai_status = 'processing'
            product.save(update_fields=['ai_status'])

            # 1. Ensure we have original image saved
            if not product.original_image:
                print(f"DEBUG: [AI Service] Initializing original_image from main image for Product {product.id}")
                product.image.seek(0)
                original_content = product.image.read()
                name = os.path.basename(product.image.name)
                product.original_image.save(name, ContentFile(original_content), save=False)
                product.save(update_fields=['original_image'])

            # Prepare image
            product.original_image.seek(0)
            input_image_bytes = product.original_image.read()
            from PIL import Image, ImageOps
            import io

            # 2. Fix Orientation (EXIF) To prevent 90-degree rotations
            img_pil = Image.open(io.BytesIO(input_image_bytes))
            img_pil = ImageOps.exif_transpose(img_pil).convert("RGBA")
            
            # Re-convert to bytes for rembg
            prep_buf = io.BytesIO()
            img_pil.save(prep_buf, format='PNG')
            input_bytes_cleaned = prep_buf.getvalue()

            # 3. Direct rembg processing (Deterministic & Robust)
            from rembg import remove, new_session
            print(f"DEBUG: [AI Service] Running rembg isnet-general-use for {product.id}")
            
            # Using a session for better performance if needed, or simple remove
            session = new_session("isnet-general-use")
            output_image_bytes = remove(input_bytes_cleaned, session=session)
            
            # Save the new transparent file
            product.image.save(f"isolated_{product.id}.png", ContentFile(output_image_bytes), save=False)
            product.image_no_bg.save(f"trans_{product.id}.png", ContentFile(output_image_bytes), save=False)
            product.ai_status = 'completed'
            product.save()
            print(f"DEBUG: [AI Service] Final rembg result saved for {product.id}")

        except Exception as e:
            print(f"ERROR: [AI Service] UNRECOVERABLE AI Failure for {product.id}: {e}")
            import traceback
            traceback.print_exc()
            product.ai_status = 'error'
            product.save(update_fields=['ai_status'])

    @staticmethod
    def generate_room_preview(product, room_image_path, result_image_path):
        """
        Deterministic room visualization pipeline:
        1. Detect the existing door opening with YOLO (if configured) or OpenCV.
        2. Remove the old door from the detected area with AI inpainting.
        3. Overlay the selected product door into the exact detected box.
        """
        try:
            import cv2
            import numpy as np

            room_bgr = cv2.imread(room_image_path, cv2.IMREAD_COLOR)
            if room_bgr is None:
                raise ValueError("Room image could not be loaded")

            door_path = product.image.path
            if product.image_no_bg and product.image_no_bg.name and os.path.exists(product.image_no_bg.path):
                door_path = product.image_no_bg.path
            elif product.original_image and product.original_image.name and os.path.exists(product.original_image.path):
                door_path = product.original_image.path

            door_rgba = cv2.imread(door_path, cv2.IMREAD_UNCHANGED)
            if door_rgba is None:
                raise ValueError("Door image could not be loaded")
            if door_rgba.ndim == 2:
                door_rgba = cv2.cvtColor(door_rgba, cv2.COLOR_GRAY2BGRA)
            elif door_rgba.shape[2] == 3:
                alpha = np.full(door_rgba.shape[:2] + (1,), 255, dtype=np.uint8)
                door_rgba = np.concatenate([door_rgba, alpha], axis=2)

            expected_aspect_ratio = get_expected_door_aspect_ratio(product, door_rgba=door_rgba)
            detected_box, detection_method = detect_door_opening_box(room_bgr, expected_aspect_ratio)
            box_1000 = pixels_to_box_1000(detected_box, room_bgr.shape[1], room_bgr.shape[0])
            print(f"DEBUG: [AI Service] Door opening detected via {detection_method}: {box_1000}")

            try:
                client = AIService.get_gemini_client()
                cleaned_room = remove_door_from_room_with_ai(room_bgr, detected_box, client)
                print(f"DEBUG: [AI Service] Old door removed with AI for product {product.id}")
            except Exception as removal_error:
                print(f"WARNING: [AI Service] AI door removal failed, using OpenCV inpaint: {removal_error}")
                cleaned_room = remove_door_from_room_locally(room_bgr, detected_box)

            final_room = overlay_door_into_room(cleaned_room, door_rgba, detected_box, add_shadow=True)

            if not cv2.imwrite(result_image_path, final_room):
                raise ValueError("Failed to save final room visualization")

            print(f"DEBUG: [AI Service] Deterministic room preview ready: {result_image_path}")
            return result_image_path

        except Exception as error:
            print(f"ERROR: [AI Service] Room preview generation failed: {error}")
            raise


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

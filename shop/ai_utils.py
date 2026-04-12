"""
TanlaAI Production Door Visualization Pipeline v40
===================================================
3-stage pipeline: Room Understanding → Door Transformation → Realistic Compositing

Pipeline modes:
  PREMIUM: GPT-4o image editing (best quality, ~$0.04-0.08/request)
  FAST:    OpenCV perspective + lighting + composite (free, deterministic)
  GEMINI:  Imagen 3 reconstruction/inpainting (fallback)
"""

import os
import io
import time
import base64
import json
import traceback

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from django.core.files.base import ContentFile
from django.conf import settings


# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════

def _log(start_t, step, msg):
    elapsed = time.time() - start_t
    print(f"[v40 {elapsed:6.2f}s] {step}: {msg}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: ROOM UNDERSTANDING (GPT-4o Vision)
# ═══════════════════════════════════════════════════════════════════════

_ROOM_ANALYSIS_PROMPT = """You are an expert interior designer and computer vision specialist.
Analyze this room photo and find the MAIN DOOR in the image.

Return a JSON object with EXACTLY this structure:
{
  "door_found": true,
  "door_corners": {
    "top_left": [x, y],
    "top_right": [x, y],
    "bottom_right": [x, y],
    "bottom_left": [x, y]
  },
  "door_box": {
    "ymin": 0.0,
    "xmin": 0.0,
    "ymax": 1.0,
    "xmax": 1.0
  },
  "wall_angle": 0,
  "lighting": {
    "direction": "left",
    "warmth": "warm",
    "intensity": 0.7
  },
  "design_dna": "Modern Uzbek living room with light walls"
}

RULES:
- All coordinates are NORMALIZED (0.0 to 1.0), where (0,0) is top-left corner of image.
- "door_corners" are the 4 corner points of the door INCLUDING its frame, following the wall's perspective.
  BE VERY PRECISE about the bottom corners — ensure they reach the ABSOLUTE floor level.
- "door_box" is a simple bounding box around the door area. Ensure the bottom (ymax) touches the floor.
- "wall_angle" is the estimated angle of the wall plane relative to camera:
  0 = wall faces camera directly
  negative = wall angles to the left
  positive = wall angles to the right
  Range: -30 to +30 degrees.
- "lighting.direction": where the main light comes from ("left", "right", "top", "ambient")
- "lighting.warmth": color temperature ("warm", "neutral", "cool")
- "lighting.intensity": brightness level 0.0 (dark) to 1.0 (very bright)
- "design_dna": brief style description of the room
- If NO door is found, set "door_found": false and use default values.

Be PRECISE with corner coordinates. They are critical for perspective-correct door placement."""


def analyze_room_advanced(room_img_pil, log_fn=None):
    """
    Analyze room image using GPT-4o Vision.
    Returns detailed room analysis including door corners for perspective transform.

    Args:
        room_img_pil: PIL Image of the room (RGB)
        log_fn: optional logging callback

    Returns:
        dict with door_corners, door_box, wall_angle, lighting, design_dna
    """
    log = log_fn or (lambda step, msg: None)

    default_result = {
        "door_found": False,
        "door_corners": {
            "top_left": [0.38, 0.15],
            "top_right": [0.62, 0.15],
            "bottom_right": [0.62, 0.85],
            "bottom_left": [0.38, 0.85],
        },
        "door_box": {"ymin": 0.15, "xmin": 0.38, "ymax": 0.85, "xmax": 0.62},
        "wall_angle": 0,
        "lighting": {"direction": "ambient", "warmth": "neutral", "intensity": 0.6},
        "design_dna": "Modern interior room",
    }

    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        log("1", "No OpenAI key, using defaults")
        return default_result

    try:
        import requests as http_requests

        # Encode image
        buf = io.BytesIO()
        room_img_pil.save(buf, format='JPEG', quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _ROOM_ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 500,
        }

        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20,
        )

        if resp.status_code != 200:
            log("1", f"GPT-4o API error: {resp.status_code}")
            return default_result

        result = json.loads(resp.json()['choices'][0]['message']['content'])
        log("1", f"Room analyzed: door_found={result.get('door_found')}, angle={result.get('wall_angle', 0)}")

        # Validate and sanitize the result
        result = _sanitize_room_analysis(result, default_result)
        return result

    except Exception as e:
        log("1", f"Room analysis error: {e}")
        return default_result


def _sanitize_room_analysis(result, default):
    """Validate and fix GPT output to ensure all fields are present and rational."""
    sanitized = dict(default)  # start with defaults

    if isinstance(result, dict):
        sanitized["door_found"] = bool(result.get("door_found", False))

        # Door corners
        corners = result.get("door_corners", {})
        if isinstance(corners, dict):
            for key in ["top_left", "top_right", "bottom_right", "bottom_left"]:
                val = corners.get(key)
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    x = max(0.0, min(1.0, float(val[0])))
                    y = max(0.0, min(1.0, float(val[1])))
                    sanitized["door_corners"][key] = [x, y]

        # Door box
        box = result.get("door_box", {})
        if isinstance(box, dict):
            for key in ["ymin", "xmin", "ymax", "xmax"]:
                val = box.get(key)
                if val is not None:
                    sanitized["door_box"][key] = max(0.0, min(1.0, float(val)))

        # Wall angle
        angle = result.get("wall_angle", 0)
        try:
            sanitized["wall_angle"] = max(-30, min(30, float(angle)))
        except (TypeError, ValueError):
            sanitized["wall_angle"] = 0

        # Lighting
        light = result.get("lighting", {})
        if isinstance(light, dict):
            direction = light.get("direction", "ambient")
            if direction in ("left", "right", "top", "ambient"):
                sanitized["lighting"]["direction"] = direction
            warmth = light.get("warmth", "neutral")
            if warmth in ("warm", "neutral", "cool"):
                sanitized["lighting"]["warmth"] = warmth
            intensity = light.get("intensity", 0.6)
            try:
                sanitized["lighting"]["intensity"] = max(0.0, min(1.0, float(intensity)))
            except (TypeError, ValueError):
                pass

        # Design DNA
        dna = result.get("design_dna", "")
        if isinstance(dna, str) and len(dna) > 3:
            sanitized["design_dna"] = dna[:200]

    return sanitized


def _normalized_box_from_pixels(pixel_box, width, height):
    left, top, right, bottom = pixel_box
    return {
        "xmin": max(0.0, min(1.0, left / float(max(1, width)))),
        "ymin": max(0.0, min(1.0, top / float(max(1, height)))),
        "xmax": max(0.0, min(1.0, right / float(max(1, width)))),
        "ymax": max(0.0, min(1.0, bottom / float(max(1, height)))),
    }


def _corners_from_pixel_box(pixel_box, width, height):
    left, top, right, bottom = pixel_box
    return {
        "top_left": [
            max(0.0, min(1.0, left / float(max(1, width)))),
            max(0.0, min(1.0, top / float(max(1, height)))),
        ],
        "top_right": [
            max(0.0, min(1.0, right / float(max(1, width)))),
            max(0.0, min(1.0, top / float(max(1, height)))),
        ],
        "bottom_right": [
            max(0.0, min(1.0, right / float(max(1, width)))),
            max(0.0, min(1.0, bottom / float(max(1, height)))),
        ],
        "bottom_left": [
            max(0.0, min(1.0, left / float(max(1, width)))),
            max(0.0, min(1.0, bottom / float(max(1, height)))),
        ],
    }


def merge_room_analysis_with_detection(room_analysis, detected_box, room_size, detection_method):
    """Preserve GPT lighting/style analysis while anchoring geometry to deterministic detection."""
    merged = _sanitize_room_analysis(room_analysis or {}, {
        "door_found": True,
        "door_corners": {
            "top_left": [0.38, 0.15],
            "top_right": [0.62, 0.15],
            "bottom_right": [0.62, 0.85],
            "bottom_left": [0.38, 0.85],
        },
        "door_box": {"ymin": 0.15, "xmin": 0.38, "ymax": 0.85, "xmax": 0.62},
        "wall_angle": 0,
        "lighting": {"direction": "ambient", "warmth": "neutral", "intensity": 0.6},
        "design_dna": "Modern interior room",
    })

    width, height = room_size
    merged["door_found"] = True
    merged["door_box"] = _normalized_box_from_pixels(detected_box, width, height)
    merged["door_corners"] = _corners_from_pixel_box(detected_box, width, height)
    merged["geometry_source"] = f"gpt+{detection_method}"
    merged["detection_method"] = detection_method
    return merged


def analyze_product_details(product_img_pil):
    """Describe the door product briefly using GPT-4o."""
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        return "a modern architectural door"

    try:
        import requests as http_requests
        buf = io.BytesIO()
        product_img_pil.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this door product briefly (max 15 words)."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
        }
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=12,
        )
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception:
        return "a modern architectural door"


def summarize_room_analysis(room_analysis, detected_box, detection_method, room_size, roi_box=None):
    width, height = room_size
    normalized_box = _normalized_box_from_pixels(detected_box, width, height)
    lighting = room_analysis.get("lighting", {})
    summary = {
        "door_found": bool(room_analysis.get("door_found", True)),
        "geometry_source": room_analysis.get("geometry_source", detection_method),
        "detection_method": detection_method,
        "wall_angle": round(float(room_analysis.get("wall_angle", 0.0)), 2),
        "design_dna": room_analysis.get("design_dna", "Modern interior room"),
        "door_box": normalized_box,
        "lighting": {
            "direction": lighting.get("direction", "ambient"),
            "warmth": lighting.get("warmth", "neutral"),
            "intensity": round(float(lighting.get("intensity", 0.6)), 2),
        },
        "preserve_elements": [
            "carpet",
            "walls",
            "curtains",
            "furniture",
            "TV",
            "floor perspective",
        ],
    }

    if roi_box is not None:
        left, top, right, bottom = roi_box
        summary["roi_box"] = {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
        }

    return summary


def build_nano_banana_prompt(room_analysis, product_desc):
    lighting = room_analysis.get("lighting", {})
    light_dir = lighting.get("direction", "ambient")
    light_warmth = lighting.get("warmth", "neutral")
    light_intensity = float(lighting.get("intensity", 0.6))
    wall_angle = float(room_analysis.get("wall_angle", 0.0))
    design_dna = room_analysis.get("design_dna", "modern Uzbek living room")

    return (
        "Image 1 is the real room crop. Image 2 is the replacement door product. "
        "Image 3 is a black-and-white mask. Edit ONLY the white masked region from Image 3.\n\n"
        "TASK:\n"
        f"Replace the existing door in Image 1 with the door from Image 2 ({product_desc}).\n\n"
        "NON-NEGOTIABLE RULES:\n"
        "- Keep the exact room geometry, camera position, carpet, curtains, TV, walls, and furniture unchanged.\n"
        "- The new door must look physically installed in the existing opening, not pasted on top.\n"
        f"- Match the room perspective and wall angle ({wall_angle:.1f} degrees).\n"
        f"- Match the existing lighting: direction {light_dir}, tone {light_warmth}, intensity {light_intensity:.2f}.\n"
        "- Add realistic contact shadow on the floor, frame edges, and wall junction.\n"
        "- Preserve wall texture, molding, skirting, and floor lines around the door.\n"
        "- Keep the door closed and proportionally correct for the opening.\n"
        f"- Preserve the design identity of the space: {design_dna}.\n"
        "- Return a photorealistic result that looks like a real phone photo."
    )


def _extract_image_and_text_from_gemini_response(response):
    parts = getattr(response, "parts", None)
    if parts is None:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None)
    parts = parts or []

    texts = []
    image = None

    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text.strip())
            continue

        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue

        try:
            image = part.as_image()
        except Exception:
            data = getattr(inline, "data", None)
            if isinstance(data, str):
                data = base64.b64decode(data)
            if data:
                image = Image.open(io.BytesIO(data)).convert("RGB")

        if image is not None:
            break

    return image, "\n".join(chunk for chunk in texts if chunk).strip()


def nano_banana_edit(room_img_pil, door_img_pil, mask_pil, room_analysis, log_fn=None):
    """Use Gemini's native image model for multi-image editing."""
    log = log_fn or (lambda step, msg: None)

    try:
        from google.genai import types
        from shop.services import AIService

        client = AIService.get_gemini_client(prefer_vertex=True)
        if not client:
            log("NB", "No Gemini client available for Nano Banana")
            return None, None

        primary_model = getattr(settings, "GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
        fallback_model = getattr(settings, "GEMINI_IMAGE_FALLBACK_MODEL", "gemini-2.5-flash-image")
        is_vertex_client = bool(getattr(client, "vertexai", False))
        model_candidates = [fallback_model, primary_model] if is_vertex_client else [primary_model, fallback_model]
        product_desc = analyze_product_details(door_img_pil)
        prompt = build_nano_banana_prompt(room_analysis, product_desc)

        contents = [
            room_img_pil.convert("RGB"),
            door_img_pil.convert("RGB"),
            mask_pil.convert("RGB"),
            prompt,
        ]
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
        )

        for model_name in model_candidates:
            if not model_name:
                continue
            try:
                log("NB", f"Trying Nano Banana model {model_name}...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                image, response_text = _extract_image_and_text_from_gemini_response(response)
                if image is not None:
                    log("NB", f"Nano Banana success via {model_name}")
                    return image.convert("RGB"), {
                        "engine": "nano-banana",
                        "model": model_name,
                        "prompt": prompt,
                        "response_text": response_text,
                        "product_description": product_desc,
                    }
            except Exception as model_error:
                log("NB", f"{model_name} failed: {model_error}")

        return None, {
            "engine": "nano-banana",
            "model": primary_model,
            "prompt": prompt,
            "response_text": "",
            "product_description": product_desc,
        }

    except Exception as exc:
        log("NB", f"Nano Banana pipeline error: {exc}")
        return None, None


def save_visualization_metadata(result_image_path, metadata):
    if not metadata:
        return

    metadata_path = os.path.splitext(result_image_path)[0] + ".json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def load_visualization_metadata(result_image_path):
    metadata_path = os.path.splitext(result_image_path)[0] + ".json"
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: PERSPECTIVE TRANSFORM (OpenCV)
# ═══════════════════════════════════════════════════════════════════════

def perspective_transform_door(door_bgra, dst_corners_px, room_width, room_height):
    """
    Apply perspective warp to door image to match room wall angle.

    Args:
        door_bgra: numpy array BGRA of the door image
        dst_corners_px: 4 destination corners in pixel coords [(x,y),...] 
                        order: [top_left, top_right, bottom_right, bottom_left]
        room_width, room_height: dimensions of room image

    Returns:
        warped_bgra: numpy array BGRA warped to room perspective
    """
    h, w = door_bgra.shape[:2]

    # Source corners (door image corners, rectangular)
    src = np.float32([
        [0, 0],       # top-left
        [w, 0],       # top-right
        [w, h],       # bottom-right
        [0, h],       # bottom-left
    ])

    dst = np.float32(dst_corners_px)

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the door image
    warped = cv2.warpPerspective(
        door_bgra,
        M,
        (room_width, room_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return warped


def corners_from_analysis(analysis, room_width, room_height):
    """
    Convert normalized corner coordinates from room analysis to pixel coordinates.

    Returns:
        list of 4 [x, y] pixel coordinates: [TL, TR, BR, BL]
    """
    corners = analysis.get("door_corners", {})
    tl = corners.get("top_left", [0.38, 0.15])
    tr = corners.get("top_right", [0.62, 0.15])
    br = corners.get("bottom_right", [0.62, 0.85])
    bl = corners.get("bottom_left", [0.38, 0.85])

    result = []
    for pt in [tl, tr, br, bl]:
        px = int(round(pt[0] * room_width))
        py = int(round(pt[1] * room_height))
        px = max(0, min(room_width - 1, px))
        py = max(0, min(room_height - 1, py))
        result.append([px, py])

    return result


def box_from_corners(corners_px, room_width, room_height):
    """Get a bounding box (left, top, right, bottom) from corner points."""
    xs = [c[0] for c in corners_px]
    ys = [c[1] for c in corners_px]
    left = max(0, min(xs))
    top = max(0, min(ys))
    right = min(room_width, max(xs))
    bottom = min(room_height, max(ys))
    return (left, top, right, bottom)


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: LIGHTING & COLOR MATCHING
# ═══════════════════════════════════════════════════════════════════════

def match_lighting(door_bgra, room_bgr, door_box, lighting_info):
    """
    Match door colors and lighting to the room environment.

    Uses LAB color space for perceptual color matching:
    1. Sample room colors around the door region
    2. Match door brightness (L channel) to room brightness
    3. Apply warmth/coolness shift based on lighting analysis

    Args:
        door_bgra: BGRA door image (warped or original)
        room_bgr: BGR room image
        door_box: (left, top, right, bottom) of door placement area
        lighting_info: dict with direction, warmth, intensity

    Returns:
        adjusted door_bgra with matched lighting
    """
    left, top, right, bottom = door_box
    rh, rw = room_bgr.shape[:2]

    # Sample the room region around the door (wall area)
    # Expand the sampling area slightly beyond the door box
    pad_x = max(10, int((right - left) * 0.15))
    pad_y = max(10, int((bottom - top) * 0.08))

    sample_left = max(0, left - pad_x)
    sample_right = min(rw, right + pad_x)
    sample_top = max(0, top - pad_y)
    sample_bottom = min(rh, bottom + pad_y)

    # Get room sample in LAB space
    room_region = room_bgr[sample_top:sample_bottom, sample_left:sample_right]
    if room_region.size == 0:
        return door_bgra

    room_lab = cv2.cvtColor(room_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    room_l_mean = np.mean(room_lab[:, :, 0])
    room_a_mean = np.mean(room_lab[:, :, 1])
    room_b_mean = np.mean(room_lab[:, :, 2])

    # Get door colors in LAB space
    alpha = door_bgra[:, :, 3]
    door_mask = alpha > 30  # only visible pixels
    if not np.any(door_mask):
        return door_bgra

    door_bgr = door_bgra[:, :, :3].copy()
    door_lab = cv2.cvtColor(door_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Get mean door L, a, b (only for visible pixels)
    door_l_mean = np.mean(door_lab[:, :, 0][door_mask])
    door_a_mean = np.mean(door_lab[:, :, 1][door_mask])
    door_b_mean = np.mean(door_lab[:, :, 2][door_mask])

    # === BRIGHTNESS MATCHING ===
    # Shift door L channel to be closer to room L
    # Use a gentle blending factor to avoid over-correction
    blend_factor = 0.35  # how much to match (0=no change, 1=full match)
    l_shift = (room_l_mean - door_l_mean) * blend_factor

    # Apply intensity adjustment from lighting analysis
    intensity = lighting_info.get("intensity", 0.6)
    # If room is very bright, lighten; if dark, darken
    intensity_adjustment = (intensity - 0.5) * 15.0  # ±7.5 L units

    door_lab[:, :, 0] = np.clip(
        door_lab[:, :, 0] + l_shift + intensity_adjustment, 0, 255
    )

    # === COLOR TEMPERATURE MATCHING ===
    warmth = lighting_info.get("warmth", "neutral")
    if warmth == "warm":
        # Warm = shift b channel towards yellow, a towards red slightly
        door_lab[:, :, 2] = np.clip(door_lab[:, :, 2] + 3.0, 0, 255)  # yellow
        door_lab[:, :, 1] = np.clip(door_lab[:, :, 1] + 1.5, 0, 255)  # reddish
    elif warmth == "cool":
        # Cool = shift b channel towards blue
        door_lab[:, :, 2] = np.clip(door_lab[:, :, 2] - 3.0, 0, 255)  # blue
        door_lab[:, :, 1] = np.clip(door_lab[:, :, 1] - 1.0, 0, 255)

    # Subtle room hue matching
    a_shift = (room_a_mean - door_a_mean) * 0.15
    b_shift = (room_b_mean - door_b_mean) * 0.15
    door_lab[:, :, 1] = np.clip(door_lab[:, :, 1] + a_shift, 0, 255)
    door_lab[:, :, 2] = np.clip(door_lab[:, :, 2] + b_shift, 0, 255)

    # Convert back to BGR
    adjusted_bgr = cv2.cvtColor(door_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Reconstruct BGRA
    result = door_bgra.copy()
    result[:, :, :3] = adjusted_bgr
    return result


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: SHADOW & EDGE BLENDING
# ═══════════════════════════════════════════════════════════════════════

def apply_directional_shadow(room_bgr, alpha_mask_full, lighting_info, strength=0.20):
    """
    Apply realistic shadow based on lighting direction.

    Args:
        room_bgr: the room image (BGR)
        alpha_mask_full: full-size alpha mask of the door (same size as room)
        lighting_info: dict with direction, warmth, intensity
        strength: shadow darkness (0-1)

    Returns:
        room_bgr with shadow applied
    """
    rh, rw = room_bgr.shape[:2]
    direction = lighting_info.get("direction", "ambient")

    # Shadow offset based on light direction
    shadow_offsets = {
        "left": (max(2, rw // 80), max(1, rh // 120)),     # shadow to right
        "right": (-max(2, rw // 80), max(1, rh // 120)),   # shadow to left
        "top": (0, max(2, rh // 60)),                       # shadow below
        "ambient": (max(1, rw // 120), max(1, rh // 80)),  # subtle right-down
    }
    offset_x, offset_y = shadow_offsets.get(direction, (1, 1))

    # Create shadow canvas
    shadow_canvas = np.zeros((rh, rw), dtype=np.float32)

    # Shift the alpha mask to create shadow
    M_shadow = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    shifted_alpha = cv2.warpAffine(
        alpha_mask_full.astype(np.float32) / 255.0,
        M_shadow,
        (rw, rh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Shadow only where alpha is NOT (shadow visible outside the door)
    door_presence = alpha_mask_full.astype(np.float32) / 255.0
    shadow_canvas = np.maximum(shadow_canvas, shifted_alpha * (1.0 - door_presence))

    # Blur the shadow for softness
    blur_size = max(9, (min(rw, rh) // 40) | 1)
    shadow_canvas = cv2.GaussianBlur(shadow_canvas, (blur_size, blur_size), 0)

    # Apply shadow to room
    shaded = room_bgr.astype(np.float32)
    shaded *= (1.0 - (shadow_canvas[..., None] * strength))
    return np.clip(shaded, 0, 255).astype(np.uint8)


def feather_alpha_edges(alpha_mask, feather_px=3):
    """
    Soften the edges of an alpha mask for natural blending.

    Args:
        alpha_mask: uint8 alpha mask
        feather_px: number of pixels to feather

    Returns:
        feathered alpha mask
    """
    if feather_px <= 0:
        return alpha_mask

    # Erode slightly then blur for soft edges
    kernel_size = max(1, feather_px)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(alpha_mask, kernel, iterations=1)
    blur_size = max(3, (feather_px * 2 + 1) | 1)
    feathered = cv2.GaussianBlur(eroded, (blur_size, blur_size), 0)
    return feathered


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: COMPOSITING
# ═══════════════════════════════════════════════════════════════════════

def composite_door_into_room(room_bgr, warped_door_bgra, lighting_info):
    """
    Final compositing: blend the perspective-corrected, lighting-matched door
    into the room with shadows and edge feathering.

    Args:
        room_bgr: BGR room image
        warped_door_bgra: BGRA door image (same size as room, warped)
        lighting_info: lighting analysis dict

    Returns:
        final BGR composite
    """
    rh, rw = room_bgr.shape[:2]

    # Extract alpha
    raw_alpha = warped_door_bgra[:, :, 3]

    # Feather edges for natural blending
    alpha = feather_alpha_edges(raw_alpha, feather_px=2)
    alpha_f = alpha.astype(np.float32) / 255.0

    # Apply shadow first
    room_with_shadow = apply_directional_shadow(room_bgr, raw_alpha, lighting_info)

    # Blend door into room
    door_rgb = warped_door_bgra[:, :, :3].astype(np.float32)
    room_f = room_with_shadow.astype(np.float32)

    composite = (alpha_f[..., None] * door_rgb) + ((1.0 - alpha_f[..., None]) * room_f)
    return np.clip(composite, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# PREMIUM PIPELINE: GPT-4o IMAGE EDITING
# ═══════════════════════════════════════════════════════════════════════

def premium_gpt4o_edit(room_img_pil, door_img_pil, room_analysis, log_fn=None):
    """
    Use OpenAI GPT-4o image generation to replace the door.
    This sends both images and a detailed prompt for highest quality.

    Args:
        room_img_pil: PIL Image of room (RGB)
        door_img_pil: PIL Image of door (RGB or RGBA)
        room_analysis: dict from analyze_room_advanced()
        log_fn: logging callback

    Returns:
        PIL Image result, or None if failed
    """
    log = log_fn or (lambda step, msg: None)
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        log("P", "No OpenAI key for premium pipeline")
        return None

    try:
        import requests as http_requests

        # Prepare room image
        room_buf = io.BytesIO()
        room_img_pil.save(room_buf, format='PNG')
        room_b64 = base64.b64encode(room_buf.getvalue()).decode('utf-8')

        # Prepare door image
        door_buf = io.BytesIO()
        door_for_send = door_img_pil.convert('RGBA') if door_img_pil.mode != 'RGBA' else door_img_pil
        door_for_send.save(door_buf, format='PNG')
        door_b64 = base64.b64encode(door_buf.getvalue()).decode('utf-8')

        design_dna = room_analysis.get("design_dna", "modern interior")
        wall_angle = room_analysis.get("wall_angle", 0)
        lighting = room_analysis.get("lighting", {})
        light_dir = lighting.get("direction", "ambient")
        light_warmth = lighting.get("warmth", "neutral")

        prompt = f"""Look at these two images:
Image 1 (first): A room photo. Style: {design_dna}.
Image 2 (second): A door product on transparent/plain background.

TASK: Replace the existing door in the room (Image 1) with the door from Image 2.

CRITICAL REQUIREMENTS:
1. Keep the EXACT room layout unchanged — walls, floor, carpet, curtains, furniture, TV, everything stays.
2. Match the door's PERSPECTIVE to the wall angle ({wall_angle:.0f} degrees from camera).
3. Match LIGHTING: light comes from {light_dir}, {light_warmth} tone.
4. Add realistic SHADOWS on the floor and wall edges matching the {light_dir} lighting.
5. The door must look PHYSICALLY INSTALLED — not floating or overlaid.
6. Keep the door frame/trim consistent with room's architectural style.
7. Result must look like a REAL PHOTOGRAPH, not AI-generated.
8. The door should be CLOSED.
9. Wall texture around the door should blend seamlessly.

Output: A single photorealistic image of the room with the new door installed."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{room_b64}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{door_b64}"},
                        },
                    ],
                }
            ],
        }

        log("P", "Sending to GPT-4o image edit...")
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if resp.status_code != 200:
            log("P", f"GPT-4o API error: {resp.status_code} - {resp.text[:200]}")
            return None

        response_data = resp.json()
        content = response_data['choices'][0]['message']['content']

        # GPT-4o text response doesn't directly return images in chat completions
        # We need to use the images API instead
        log("P", "GPT-4o chat completed, trying DALL-E 3 edit...")

        # Try OpenAI Images API (DALL-E) for actual image editing
        return _try_dalle_edit(room_img_pil, door_img_pil, room_analysis, api_key, log)

    except Exception as e:
        log("P", f"Premium pipeline error: {e}")
        return None


def _try_dalle_edit(room_img_pil, door_img_pil, room_analysis, api_key, log):
    """
    Try using OpenAI's image generation to create the composite.
    Uses the gpt-image-1 model which supports image inputs.
    """
    try:
        import requests as http_requests

        design_dna = room_analysis.get("design_dna", "modern interior")
        wall_angle = room_analysis.get("wall_angle", 0)
        lighting = room_analysis.get("lighting", {})
        light_dir = lighting.get("direction", "ambient")
        light_warmth = lighting.get("warmth", "neutral")

        # Prepare room image as PNG bytes
        room_buf = io.BytesIO()
        room_img_pil.convert('RGBA').save(room_buf, format='PNG')
        room_buf.seek(0)

        # Prepare door image as PNG bytes
        door_buf = io.BytesIO()
        door_for_send = door_img_pil.convert('RGBA') if door_img_pil.mode != 'RGBA' else door_img_pil
        door_for_send.save(door_buf, format='PNG')
        door_buf.seek(0)

        prompt = (
            f"Replace the existing door in this {design_dna} room with the door from the second image. "
            f"Keep ALL room elements exactly unchanged (walls, floor, carpet, curtains, furniture). "
            f"Match perspective (wall angle {wall_angle:.0f}°), {light_warmth} {light_dir} lighting. "
            f"Add realistic shadows. The door must look physically installed, not overlaid. "
            f"Photorealistic result, closed door."
        )

        headers = {"Authorization": f"Bearer {api_key}"}

        # Use the images/edits endpoint with multiple images
        files = [
            ('image[]', ('room.png', room_buf, 'image/png')),
            ('image[]', ('door.png', door_buf, 'image/png')),
        ]
        data = {
            'model': 'gpt-image-1',
            'prompt': prompt,
            'size': '1024x1024',
            'quality': 'high',
        }

        log("P", "Trying gpt-image-1 API...")
        resp = http_requests.post(
            "https://api.openai.com/v1/images/edits",
            headers=headers,
            files=files,
            data=data,
            timeout=90,
        )

        if resp.status_code == 200:
            result_data = resp.json()
            # Could be b64_json or url
            image_data = result_data.get('data', [{}])[0]

            if 'b64_json' in image_data:
                img_bytes = base64.b64decode(image_data['b64_json'])
                result_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                log("P", "gpt-image-1 SUCCESS!")
                return result_img
            elif 'url' in image_data:
                img_resp = http_requests.get(image_data['url'], timeout=30)
                if img_resp.status_code == 200:
                    result_img = Image.open(io.BytesIO(img_resp.content)).convert('RGB')
                    log("P", "gpt-image-1 SUCCESS (via URL)!")
                    return result_img

        log("P", f"gpt-image-1 failed: {resp.status_code} - {resp.text[:300]}")
        return None

    except Exception as e:
        log("P", f"DALL-E edit error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# GEMINI PIPELINE (FALLBACK)
# ═══════════════════════════════════════════════════════════════════════

def gemini_reconstruct(room_img_pil, door_img_pil, room_analysis, log_fn=None):
    """
    Gemini Imagen 3 — Replace the door in the ACTUAL room photo.
    
    Strategy: INPAINTING FIRST (preserves room identity 100%).
    The original room photo is kept intact — only the door area is replaced.
    This ensures carpet, curtains, TV, walls, etc. remain exactly the same.
    
    Fallback: Full reconstruction with style reference (room may look different).
    """
    log = log_fn or (lambda step, msg: None)

    try:
        from shop.services import AIService
        from google.genai import types

        client = AIService.get_gemini_client()
        if not client:
            log("G", "No Gemini client available")
            return None

        design_dna = room_analysis.get("design_dna", "modern interior room")
        prod_desc = analyze_product_details(door_img_pil)
        lighting = room_analysis.get("lighting", {})
        light_dir = lighting.get("direction", "ambient")
        light_warmth = lighting.get("warmth", "neutral")

        # Prepare room image bytes
        r_buf = io.BytesIO()
        room_img_pil.save(r_buf, format='PNG')
        
        # Prepare door image bytes (RGB for better Gemini compatibility)
        d_buf = io.BytesIO()
        door_img_pil.convert('RGB').save(d_buf, format='PNG')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ATTEMPT 1: INPAINTING (PRIMARY — preserves room identity 100%)
        # Uses edit_image with mask to replace ONLY the door area.
        # Everything else in the room stays EXACTLY the same.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        try:
            log("G1", "🎯 ATTEMPT 1: Inpainting (room identity preserved)...")
            bx = room_analysis['door_box']
            rw, rh = room_img_pil.size
            l = int(bx['xmin'] * rw)
            t = int(bx['ymin'] * rh)
            r = int(bx['xmax'] * rw)
            b = int(bx['ymax'] * rh)

            # Create mask covering the door area with generous padding
            pad_x = max(20, int((r - l) * 0.12))
            pad_y = max(15, int((b - t) * 0.06))
            mask = Image.new("L", (rw, rh), 0)
            ImageDraw.Draw(mask).rectangle(
                [l - pad_x, t - pad_y, r + pad_x, b + pad_y], fill=255
            )
            m_buf = io.BytesIO()
            mask.filter(ImageFilter.GaussianBlur(10)).save(m_buf, format='PNG')

            prompt_inpaint = (
                f"Replace the door in the masked area with this door product [1]. "
                f"The new door ({prod_desc}) must be CLOSED and properly installed "
                f"in the existing door frame. Match the wall texture where the door "
                f"meets the wall. The door must fit the perspective and scale of "
                f"the room. Lighting: {light_warmth} {light_dir}. "
                f"Keep everything outside the mask EXACTLY unchanged. "
                f"The result must look like a real photograph of this room "
                f"with the new door professionally installed."
            )
            res1 = client.models.edit_image(
                model='imagen-3.0-capability-001',
                prompt=prompt_inpaint,
                reference_images=[
                    types.RawReferenceImage(
                        reference_id=0,
                        reference_image=types.Image(image_bytes=r_buf.getvalue()),
                    ),
                    types.SubjectReferenceImage(
                        reference_id=1,
                        image=types.Image(image_bytes=d_buf.getvalue()),
                        config=types.SubjectReferenceConfig(
                            subject_type='SUBJECT_REFERENCE_TYPE_PRODUCT'
                        ),
                    ),
                    types.MaskReferenceImage(
                        reference_id=2,
                        reference_image=types.Image(image_bytes=m_buf.getvalue()),
                    ),
                ],
                config=types.EditImageConfig(
                    edit_mode='EDIT_MODE_INPAINT_INSERTION',
                    number_of_images=1,
                    output_mime_type='image/png',
                ),
            )
            if res1 and res1.generated_images:
                result = Image.open(
                    io.BytesIO(res1.generated_images[0].image.image_bytes)
                ).convert("RGB")
                log("G1", "✅ INPAINTING SUCCESS — room identity preserved!")
                return result
        except Exception as e1:
            log("G1", f"Attempt 1 (inpainting) failed: {e1}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ATTEMPT 2: INPAINTING with different edit mode / simpler prompt
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        try:
            log("G2", "ATTEMPT 2: Inpainting (alt mode)...")
            # Reset buffers
            r_buf.seek(0)
            d_buf.seek(0)
            m_buf.seek(0)

            prompt_alt = (
                f"Install this {prod_desc} door [1] into the room [0] in the masked area. "
                f"Closed door, properly fitted, matching room perspective and lighting. "
                f"Photorealistic result."
            )

            # Try INPAINT_EDIT mode as alternative
            for edit_mode in ['EDIT_MODE_INPAINT_INSERTION', 'INPAINT_EDIT']:
                try:
                    res2 = client.models.edit_image(
                        model='imagen-3.0-capability-001',
                        prompt=prompt_alt,
                        reference_images=[
                            types.RawReferenceImage(
                                reference_id=0,
                                reference_image=types.Image(image_bytes=r_buf.getvalue()),
                            ),
                            types.SubjectReferenceImage(
                                reference_id=1,
                                image=types.Image(image_bytes=d_buf.getvalue()),
                                config=types.SubjectReferenceConfig(
                                    subject_type='SUBJECT_REFERENCE_TYPE_PRODUCT'
                                ),
                            ),
                            types.MaskReferenceImage(
                                reference_id=2,
                                reference_image=types.Image(image_bytes=m_buf.getvalue()),
                            ),
                        ],
                        config=types.EditImageConfig(
                            edit_mode=edit_mode,
                            number_of_images=1,
                            output_mime_type='image/png',
                        ),
                    )
                    if res2 and res2.generated_images:
                        result = Image.open(
                            io.BytesIO(res2.generated_images[0].image.image_bytes)
                        ).convert("RGB")
                        log("G2", f"✅ Inpainting ({edit_mode}) SUCCESS!")
                        return result
                except Exception as e_mode:
                    log("G2", f"Mode {edit_mode} failed: {e_mode}")
                    r_buf.seek(0)
                    d_buf.seek(0)
                    m_buf.seek(0)
                    continue

        except Exception as e2:
            log("G2", f"Attempt 2 failed: {e2}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ATTEMPT 3: Full reconstruction (LAST RESORT — room may differ)
        # Uses generate_images with style_reference — the room will be 
        # re-generated, so carpet/curtains/etc may look different.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        try:
            log("G3", "ATTEMPT 3: Full reconstruction (fallback)...")
            r_buf.seek(0)
            d_buf.seek(0)

            prompt_reconstruct = (
                f"Photorealistic interior photo of THIS EXACT ROOM from reference [0]. "
                f"The room has: {design_dna}. "
                f"REPLACE the existing door with the door product [1] ({prod_desc}). "
                f"The new door must be CLOSED and properly installed. "
                f"KEEP EVERYTHING ELSE IDENTICAL: same walls, same floor, same carpet, "
                f"same curtains, same furniture, same lighting ({light_warmth} {light_dir}). "
                f"Only the door changes. Professional architectural photography, 8K."
            )
            res3 = client.models.generate_images(
                model='imagen-3.0-capability-001',
                prompt=prompt_reconstruct,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type='image/png',
                    style_reference_config=types.StyleReferenceConfig(
                        style_reference_images=[
                            types.StyleReferenceImage(
                                reference_id=0,
                                image=types.Image(image_bytes=r_buf.getvalue()),
                            )
                        ]
                    ),
                    subject_reference_config=types.SubjectReferenceConfig(
                        subject_reference_images=[
                            types.SubjectReferenceImage(
                                reference_id=1,
                                image=types.Image(image_bytes=d_buf.getvalue()),
                                subject_type='SUBJECT_REFERENCE_TYPE_PRODUCT',
                            )
                        ]
                    ),
                ),
            )
            if res3 and res3.generated_images:
                result = Image.open(
                    io.BytesIO(res3.generated_images[0].image.image_bytes)
                ).convert("RGB")
                log("G3", "Attempt 3 SUCCESS — full reconstruction (room may differ)")
                return result
        except Exception as e3:
            log("G3", f"Attempt 3 failed: {e3}")

        return None

    except Exception as e:
        log("G", f"Gemini pipeline error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# FAST PIPELINE: OpenCV Deterministic
# ═══════════════════════════════════════════════════════════════════════

def fast_opencv_pipeline(room_bgr, door_bgra, room_analysis, log_fn=None):
    """
    Deterministic OpenCV pipeline:
    1. Perspective transform door to match room wall
    2. Match lighting/colors
    3. Composite with feathered edges and shadows

    Args:
        room_bgr: numpy BGR room image
        door_bgra: numpy BGRA door image (background removed)
        room_analysis: dict from analyze_room_advanced()
        log_fn: logging callback

    Returns:
        numpy BGR final composite
    """
    log = log_fn or (lambda step, msg: None)
    rh, rw = room_bgr.shape[:2]

    # Get perspective corners
    corners_px = corners_from_analysis(room_analysis, rw, rh)
    log("F1", f"Door corners (px): {corners_px}")

    # Perspective transform
    warped = perspective_transform_door(door_bgra, corners_px, rw, rh)
    log("F2", "Perspective transform done")

    # Get bounding box for lighting sampling
    door_box = box_from_corners(corners_px, rw, rh)

    # Match lighting
    lighting_info = room_analysis.get("lighting", {})
    warped_lit = match_lighting(warped, room_bgr, door_box, lighting_info)
    log("F3", "Lighting matched")

    # Remove existing door from room (inpaint the area) before compositing
    left, top, right, bottom = door_box
    inpaint_mask = np.zeros((rh, rw), dtype=np.uint8)
    pad = max(5, int((right - left) * 0.03))
    cv2.rectangle(inpaint_mask, (left - pad, top - pad), (right + pad, bottom + pad), 255, -1)
    room_cleaned = cv2.inpaint(room_bgr, inpaint_mask, 7, cv2.INPAINT_TELEA)
    log("F4", "Room door area inpainted")

    # Final composite
    result = composite_door_into_room(room_cleaned, warped_lit, lighting_info)
    log("F5", "Compositing done")

    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

def visualize_door_in_room(product, room_image_path, result_image_path, box_1000=None, override_prompt=None):
    """
    v45 - The Ultimate bulletproof ROI Pipeline using OpenCV strict bounds
    1. Uses `detect_door_opening_box` to find the mathematically perfect bounding box.
    2. Clips an ROI around this box to save Gemini compute and isolate the edit.
    3. Tries Gemini AI inpainting (insertion).
    4. Safe flawless OpenCV overlay fallback if Gemini API fails!
    """
    start_t = time.time()
    log = lambda step, msg: _log(start_t, step, msg)
    room_raw = None
    metadata = {}

    try:
        from shop.services import (
            AIService, load_best_door_rgba, 
            detect_door_opening_box, get_expected_door_aspect_ratio,
            overlay_door_into_room
        )
        from google.genai import types
        import cv2

        log("0", "Loading resources...")
        room_raw = ImageOps.exif_transpose(Image.open(room_image_path)).convert("RGB")
        room_raw.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        rw, rh = room_raw.size
        room_bgr = cv2.cvtColor(np.array(room_raw), cv2.COLOR_RGB2BGR)

        # Load product door
        door_bgra_cv = load_best_door_rgba(product)
        if door_bgra_cv is None:
            raise ValueError("Door asset missing")
            
        door_rgb_np = cv2.cvtColor(door_bgra_cv[:, :, :3], cv2.COLOR_BGR2RGB)
        door_rgba_np = np.dstack([door_rgb_np, door_bgra_cv[:, :, 3]])
        door_pil = Image.fromarray(door_rgba_np, 'RGBA')

        # 1. Precise GPT-4 Vision Box Detection (Bulletproof)
        log("1a", "Running GPT room analysis for lighting/style and precise bounding box...")
        raw_room_analysis = analyze_room_advanced(room_raw, log_fn=log)
        
        expected_aspect = get_expected_door_aspect_ratio(product, door_rgba=door_bgra_cv)
        door_ref = raw_room_analysis.get("door_box", {"ymin": 0.2, "xmin": 0.35, "ymax": 0.85, "xmax": 0.65})
        raw_gpt_box = (door_ref["xmin"] * rw, door_ref["ymin"] * rh, door_ref["xmax"] * rw, door_ref["ymax"] * rh)
        
        from shop.services import normalize_door_opening_box
        detected_box = normalize_door_opening_box(raw_gpt_box, rw, rh, expected_aspect)
        detection_method = "gpt-4o-vision"
        
        log("1", f"Door opening detected via [{detection_method}]: {detected_box}")
        room_analysis = merge_room_analysis_with_detection(raw_room_analysis, detected_box, (rw, rh), detection_method)
        
        xmin, ymin, xmax, ymax = detected_box
        
        # 2. ROI Crop
        pad_x = (xmax - xmin) * 0.25
        pad_y = (ymax - ymin) * 0.15
        
        c_left = max(0, int(xmin - pad_x))
        c_top = max(0, int(ymin - pad_y))
        c_right = min(rw, int(xmax + pad_x))
        c_bottom = min(rh, int(ymax + pad_y))
        
        roi_img = room_raw.crop((c_left, c_top, c_right, c_bottom))
        roi_w, roi_h = roi_img.size
        
        # 3. Exact mask inside ROI
        mask_pil = Image.new("L", (roi_w, roi_h), 0)
        roi_box = [xmin - c_left, ymin - c_top, xmax - c_left, ymax - c_top]
        ImageDraw.Draw(mask_pil).rectangle(roi_box, fill=255)
        # Soften border
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(3))

        analysis_summary = summarize_room_analysis(
            room_analysis,
            detected_box,
            detection_method,
            (rw, rh),
            roi_box=(c_left, c_top, c_right, c_bottom),
        )
        metadata = {
            "pipeline": {
                "version": "v47.1",
                "room_analysis_engine": "gpt-4o",
                "image_edit_engine": "nano-banana",
                "fallbacks": ["imagen-3.0-capability-001", "opencv"],
            },
            "analysis": analysis_summary,
            "generation_prompt": "",
            "generation_meta": {},
        }
        
        edited_roi_pil = None
        
        # 4. v47 - PRE-PROCESS DOOR FOR AI (Perspective Alignment + Border Cleaning)
        log("4", "v47: Cleaning borders and pre-warping door for better AI reference...")
        try:
            from shop.services import perspective_warp_door_to_corners
            
            # Map normalized corners to absolute pixels for warping
            target_corners = {
                k: (v[0] * rw, v[1] * rh) for k, v in room_analysis['door_corners'].items()
            }
            # We warp into a black canvas the size of the room
            warped_door_cv = perspective_warp_door_to_corners(door_bgra_cv, target_corners, rw, rh)
            
            # Crop the warped door to just its bounds for use as a subject reference
            w_alpha = warped_door_cv[:, :, 3]
            w_coords = cv2.findNonZero(w_alpha)
            if w_coords is not None:
                wx, wy, ww, wh = cv2.boundingRect(w_coords)
                warped_door_crop = warped_door_cv[wy:wy+wh, wx:wx+ww]
                door_pil_for_ai = Image.fromarray(cv2.cvtColor(warped_door_crop, cv2.COLOR_BGRA2RGBA), "RGBA")
                log("4", f"Pre-warp successful: {ww}x{wh}")
            else:
                door_pil_for_ai = door_pil
        except Exception as warp_err:
            log("4", f"Pre-warp failed: {warp_err}")
            door_pil_for_ai = door_pil

        # 5. v47 - FULL SCENE RECONSTRUCTION (Primary Mode)
        # We now re-render the WHOLE room to ensure perfect blending/lighting.
        if edited_roi_pil is None:
            try:
                client = AIService.get_gemini_client(prefer_vertex=True)
                
                # Full room style reference
                room_buf = io.BytesIO()
                room_raw.save(room_buf, format='PNG')
                
                # Pre-warped subject reference
                door_buf = io.BytesIO()
                door_pil_for_ai.convert("RGB").save(door_buf, format='PNG')

                prompt_text = override_prompt if override_prompt else (
                    "TASK: Install the EXACT door from IMAGE [1] into the house entrance shown in IMAGE [0].\n"
                    "REQUIREMENTS:\n"
                    "1. Keep the room in IMAGE [0] EXACTLY the same (walls, floor, furniture, lighting).\n"
                    "2. Use the EXACT design, color, and geometric pattern of the door from IMAGE [1].\n"
                    "3. The door MUST be grounded perfectly on the floor.\n"
                    "4. Maintain professional architectural photography quality, 8K, realistic textures."
                )

                log("5", "🎯 v47 Attempt: Full Scene Reconstruction...")
                resp = client.models.generate_images(
                    model='imagen-3.0-capability-001',
                    prompt=prompt_text,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type='image/png',
                        style_reference_config=types.StyleReferenceConfig(
                            style_reference_images=[
                                types.StyleReferenceImage(
                                    reference_id=0,
                                    image=types.Image(image_bytes=room_buf.getvalue()),
                                )
                            ]
                        ),
                        subject_reference_config=types.SubjectReferenceConfig(
                            subject_reference_images=[
                                types.SubjectReferenceImage(
                                    reference_id=1,
                                    image=types.Image(image_bytes=door_buf.getvalue()),
                                    subject_type='SUBJECT_REFERENCE_TYPE_PRODUCT',
                                )
                            ]
                        ),
                    )
                )

                if resp.generated_images:
                    img_bytes = resp.generated_images[0].image.image_bytes
                    final_full_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    final_full_pil.save(result_image_path, format='JPEG', quality=95)
                    log("✓", "v47 Full Scene Reconstruction SUCCESS")
                    
                    metadata["generation_prompt"] = prompt_text
                    metadata["generation_meta"] = {
                        "engine": "imagen-gen",
                        "model": "imagen-3.0-capability-001",
                        "mode": "reconstruction",
                    }
                    metadata["pipeline"]["image_edit_engine"] = "imagen-gen"
                    
                    save_visualization_metadata(result_image_path, metadata)
                    log("✓", f"PIPELINE COMPLETE in {time.time() - start_t:.1f}s")
                    return result_image_path

            except Exception as gen_err:
                log("5", f"v47 Full Scene failed: {gen_err}")

        # 6. Fallback to Selective Inpaint (Old v46 mode)
        if edited_roi_pil is None:
            try:
                log("6", "Fallback: Selective Inpainting...")
                client = AIService.get_gemini_client(prefer_vertex=True)
                roi_buf = io.BytesIO()
                roi_img.save(roi_buf, format='PNG')
                mask_buf = io.BytesIO()
                mask_pil.save(mask_buf, format='PNG')
                door_buf = io.BytesIO()
                door_pil.save(door_buf, format='PNG')

                room_ref = types.RawReferenceImage(reference_id=0, reference_image=types.Image(image_bytes=roi_buf.getvalue()))
                door_ref = types.SubjectReferenceImage(
                    reference_id=1,
                    reference_image=types.Image(image_bytes=door_buf.getvalue()),
                    config=types.SubjectReferenceConfig(subject_type='SUBJECT_TYPE_PRODUCT'),
                )
                mask_ref = types.MaskReferenceImage(
                    reference_id=2,
                    reference_image=types.Image(image_bytes=mask_buf.getvalue()),
                )

                prompt_text = override_prompt if override_prompt else (
                    "TASK: Inpaint the EXACT door from IMAGE [1] into the masked area of IMAGE [0].\n"
                    "Maintain the EXACT design, color, and pattern of the door. Ground it on the floor."
                )

                # Safest SDK modes for Imagen 3 editing
                for mode in ("EDIT_MODE_INPAINT_INSERTION",):
                    try:
                        resp = client.models.edit_image(
                            model='imagen-3.0-capability-001',
                            prompt=prompt_text,
                            reference_images=[room_ref, door_ref, mask_ref],
                            config=types.EditImageConfig(
                                edit_mode=mode,
                                number_of_images=1,
                                output_mime_type='image/png',
                            )
                        )
                        
                        if resp.generated_images:
                            img_bytes = resp.generated_images[0].image.image_bytes
                            edited_roi_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            if edited_roi_pil.size != (roi_w, roi_h):
                                edited_roi_pil = edited_roi_pil.resize((roi_w, roi_h), Image.LANCZOS)
                            log("2", f"✅ Gemini Inpaint Success (mode: {mode})")
                            metadata["generation_prompt"] = prompt_text
                            metadata["generation_meta"] = {
                                "engine": "imagen-edit",
                                "model": "imagen-3.0-capability-001",
                                "mode": mode,
                            }
                            metadata["pipeline"]["image_edit_engine"] = "imagen-edit"
                            break
                    except Exception as inner_e:
                        log("2", f"Gemini mode {mode} skipped: {inner_e}")

            except Exception as ai_e:
                log("2", f"Gemini setup failed: {ai_e}")

        # 7. Safe OpenCV Fallback (FLAWLESS PLACEMENT)
        if edited_roi_pil is None:
            log("3", "⚠️ Gemini failed. Using v47 Homography OpenCV Composite...")
            try:
                from shop.services import perspective_warp_door_to_corners
                
                roi_bgr = cv2.cvtColor(np.array(roi_img), cv2.COLOR_RGB2BGR)
                roi_pixel_corners = {
                    k: (v[0] * rw - c_left, v[1] * rh - c_top) 
                    for k, v in room_analysis['door_corners'].items()
                }
                
                # Warp the door directly into the ROI canvas
                warped_bgra = perspective_warp_door_to_corners(door_bgra_cv, roi_pixel_corners, roi_w, roi_h)
                
                # Composite
                alpha = warped_bgra[:, :, 3].astype(np.float32) / 255.0
                door_rgb = warped_bgra[:, :, :3].astype(np.float32)
                region = roi_bgr.astype(np.float32)
                blended = (alpha[..., None] * door_rgb) + ((1.0 - alpha[..., None]) * region)
                final_roi_bgr = np.clip(blended, 0, 255).astype(np.uint8)
                
                edited_roi_pil = Image.fromarray(cv2.cvtColor(final_roi_bgr, cv2.COLOR_BGR2RGB))
                metadata["generation_meta"] = {
                    "engine": "opencv-v47",
                    "model": "homography-composite",
                    "mode": "warp-overlay",
                }
                metadata["pipeline"]["image_edit_engine"] = "opencv-v47"
            except Exception as cv_err:
                log("3", f"Homography fallback failed: {cv_err}")
                # Ultimate simple fallback
                final_roi_bgr = cv2.cvtColor(np.array(roi_img), cv2.COLOR_RGB2BGR)
                edited_roi_pil = Image.fromarray(cv2.cvtColor(final_roi_bgr, cv2.COLOR_BGR2RGB))

        # 7. Paste back cleanly!
        log("4", "Pasting edited door back into original room image...")
        final_img = room_raw.copy()
        final_img.paste(edited_roi_pil, (c_left, c_top))
        
        final_img.save(result_image_path, format='JPEG', quality=95)
        save_visualization_metadata(result_image_path, metadata)
        log("✓", f"PIPELINE COMPLETE in {time.time() - start_t:.1f}s")
        return result_image_path

    except Exception as e_fatal:
        log("X", f"Fatal error: {e_fatal}")
        traceback.print_exc()
        try:
            if room_raw:
                room_raw.save(result_image_path, format='JPEG', quality=90)
            else:
                Image.new('RGB', (800, 600), color=(240, 240, 240)).save(result_image_path)
            if metadata:
                save_visualization_metadata(result_image_path, metadata)
        except Exception:
            pass
        return result_image_path


# ═══════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════

def get_gemini_client():
    """Legacy wrapper — delegates to AIService."""
    from shop.services import AIService
    return AIService.get_gemini_client()


def process_product_with_ai(product):
    """Legacy wrapper — delegates to AIService."""
    from shop.services import AIService
    AIService.process_product_background(product)


def analyze_room_for_placement(room_img):
    """Legacy wrapper — calls new analyze_room_advanced."""
    result = analyze_room_advanced(room_img)
    return {
        "design_dna": result.get("design_dna", "Modern interior"),
        "door_box": result.get("door_box", {"ymin": 0.2, "xmin": 0.4, "ymax": 0.8, "xmax": 0.6}),
        "lighting": result["lighting"].get("direction", "natural"),
    }

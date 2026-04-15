import numpy as np
from django.test import SimpleTestCase
from unittest.mock import patch
import tempfile
from pathlib import Path
import cv2

from shop.services import (
    AIService,
    add_floor_contact_shadow,
    build_box_mask,
    compute_floor_aligned_door_box,
    detect_door_opening_box,
    detect_door_box_with_opencv,
    match_door_lighting_to_room,
    normalize_door_opening_box,
    overlay_door_into_room,
    remove_door_from_room_locally,
)


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


class RoomPreviewPipelineTests(SimpleTestCase):
    def test_detect_door_box_with_opencv_finds_room_door(self):
        room = np.full((420, 320, 3), 230, dtype=np.uint8)
        expected_box = (112, 60, 208, 368)

        room[expected_box[1]:expected_box[3], expected_box[0]:expected_box[2]] = (45, 45, 45)
        room[expected_box[1] + 12:expected_box[3] - 12, expected_box[0] + 12:expected_box[2] - 12] = (70, 70, 70)
        room[360:420, :] = (190, 190, 190)

        detected_box = detect_door_box_with_opencv(room, expected_aspect_ratio=96 / 308)

        self.assertIsNotNone(detected_box)
        self.assertGreater(compute_iou(detected_box, expected_box), 0.60)

    def test_remove_door_from_room_locally_brightens_masked_door_area(self):
        room = np.full((240, 180, 3), 220, dtype=np.uint8)
        pixel_box = (65, 35, 120, 210)
        room[pixel_box[1]:pixel_box[3], pixel_box[0]:pixel_box[2]] = (30, 30, 30)

        cleaned = remove_door_from_room_locally(room, pixel_box)

        self.assertGreater(float(cleaned[120, 90].mean()), 120.0)
        self.assertGreater(float(cleaned[120, 90].mean()), float(room[120, 90].mean()))

    def test_overlay_door_into_room_uses_alpha_blend_inside_target_box(self):
        room = np.full((240, 180, 3), 200, dtype=np.uint8)
        door = np.zeros((140, 70, 4), dtype=np.uint8)
        door[:, :, :3] = (40, 90, 160)
        door[:, :, 3] = 255

        box = (55, 30, 125, 210)
        combined = overlay_door_into_room(room, door, box, add_shadow=False)

        self.assertTrue(np.array_equal(combined[10, 10], np.array([200, 200, 200], dtype=np.uint8)))
        self.assertTrue(np.array_equal(combined[120, 90], np.array([40, 90, 160], dtype=np.uint8)))

    def test_compute_floor_aligned_door_box_keeps_top_gap_and_bottom_anchor(self):
        door = np.zeros((200, 80, 4), dtype=np.uint8)
        box = (40, 20, 120, 220)

        placed = compute_floor_aligned_door_box(box, door, image_width=180, image_height=260)

        self.assertEqual(placed[3], box[3])
        self.assertGreaterEqual(placed[1], 30)
        self.assertLess(placed[2] - placed[0], box[2] - box[0])

    def test_match_door_lighting_to_room_tints_and_softens_edges(self):
        room = np.full((220, 180, 3), (90, 120, 150), dtype=np.uint8)
        door = np.zeros((160, 80, 4), dtype=np.uint8)
        door[:, :, :3] = 245
        door[4:-4, 4:-4, 3] = 255

        matched = match_door_lighting_to_room(door, room, (50, 30, 130, 190))

        self.assertLess(int(matched[4, 40, 3]), int(matched[20, 40, 3]))
        self.assertLess(int(matched[4, 40, 0]), int(matched[20, 40, 0]))
        self.assertLess(int(matched[4, 40, 1]), int(matched[20, 40, 1]))
        self.assertLess(int(matched[4, 40, 2]), int(matched[20, 40, 2]))

    def test_add_floor_contact_shadow_darkens_floor_near_door_base(self):
        room = np.full((240, 180, 3), 215, dtype=np.uint8)
        shaded = add_floor_contact_shadow(room, (60, 40, 120, 200), strength=0.35)

        self.assertLess(float(shaded[200, 90].mean()), 215.0)
        self.assertLess(float(shaded[204, 90].mean()), 215.0)
        self.assertEqual(float(shaded[230, 20].mean()), 215.0)

    def test_build_box_mask_marks_detected_area(self):
        mask = build_box_mask(100, 80, (20, 10, 50, 80), pad_x_ratio=0.0, pad_y_ratio=0.0)

        self.assertEqual(int(mask[50, 30]), 255)
        self.assertEqual(int(mask[5, 5]), 0)

    def test_normalize_door_opening_box_widens_too_narrow_detection(self):
        normalized = normalize_door_opening_box((120, 70, 165, 360), 320, 420, expected_aspect_ratio=0.32)

        self.assertLessEqual(normalized[0], 90)
        self.assertGreaterEqual(normalized[2], 190)
        self.assertGreater(normalized[3] - normalized[1], 280)

    def test_detect_door_opening_box_prefers_full_frame_not_inner_leaf(self):
        room = np.full((420, 320, 3), 232, dtype=np.uint8)
        expected_box = (96, 72, 224, 370)

        room[expected_box[1]:expected_box[3], expected_box[0]:expected_box[2]] = (248, 248, 248)
        room[expected_box[1] + 10:expected_box[3] - 10, expected_box[0] + 10:expected_box[2] - 10] = (95, 62, 28)
        room[expected_box[1] + 12:expected_box[3] - 12, 146:182] = (25, 25, 25)
        room[360:420, :] = (192, 192, 192)

        detected_box, method = detect_door_opening_box(room, expected_aspect_ratio=0.32)

        self.assertIn(method, {'opencv-lines', 'opencv', 'default'})
        self.assertGreater(compute_iou(detected_box, expected_box), 0.55)

    def test_generate_room_preview_regression_does_not_raise_name_error(self):
        room = np.full((260, 180, 3), 228, dtype=np.uint8)
        room[40:228, 62:118] = (55, 55, 55)
        room[228:260, :] = (190, 190, 190)

        door_rgba = np.zeros((180, 70, 4), dtype=np.uint8)
        door_rgba[:, :, :3] = (235, 235, 235)
        door_rgba[:, :, 3] = 255

        class DummyProduct:
            id = 999
            width = 80
            height = 200

        with tempfile.TemporaryDirectory() as tmpdir:
            room_path = str(Path(tmpdir) / 'room.png')
            result_path = str(Path(tmpdir) / 'result.png')
            self.assertTrue(cv2.imwrite(room_path, room))

            with (
                patch('shop.services.load_best_door_rgba', return_value=door_rgba),
                patch('shop.services.detect_door_opening_box', return_value=((58, 36, 122, 230), 'mock')),
                patch.object(AIService, 'refine_door_edges_with_ai', side_effect=ValueError('skip ai')),
            ):
                output_path = AIService.generate_room_preview(DummyProduct(), room_path, result_path)

            self.assertEqual(output_path, result_path)
            self.assertTrue(Path(result_path).exists())

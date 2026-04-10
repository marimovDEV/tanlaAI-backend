import numpy as np
from django.test import SimpleTestCase

from shop.services import (
    build_box_mask,
    detect_door_opening_box,
    detect_door_box_with_opencv,
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

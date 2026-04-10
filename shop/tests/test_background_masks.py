import numpy as np
from django.test import SimpleTestCase

from shop.services import merge_candidate_masks, refine_product_mask


class BackgroundMaskTests(SimpleTestCase):
    def test_refine_product_mask_fills_internal_holes_and_removes_noise(self):
        mask = np.zeros((120, 80), dtype=np.uint8)
        mask[10:110, 20:60] = 255
        mask[35:75, 32:48] = 0
        mask[5:10, 5:10] = 255

        refined = refine_product_mask(mask)

        self.assertIsNotNone(refined)
        self.assertEqual(int(refined[55, 40]), 255)
        self.assertEqual(int(refined[7, 7]), 0)
        self.assertEqual(int(refined[0, 0]), 0)

    def test_merge_candidate_masks_restores_full_silhouette_from_polygon(self):
        alpha_mask = np.zeros((120, 80), dtype=np.uint8)
        alpha_mask[10:110, 20:60] = 255
        alpha_mask[35:75, 32:48] = 0

        polygon_mask = np.zeros((120, 80), dtype=np.uint8)
        polygon_mask[8:112, 18:62] = 255

        merged = merge_candidate_masks(alpha_mask, polygon_mask)

        self.assertIsNotNone(merged)
        self.assertEqual(int(merged[55, 40]), 255)
        self.assertEqual(int(merged[15, 20]), 255)
        self.assertEqual(int(merged[0, 0]), 0)

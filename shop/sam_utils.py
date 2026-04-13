import os
import cv2
import numpy as np
import torch
from django.conf import settings
from segment_anything import sam_model_registry, SamPredictor

class SAMService:
    _instance = None
    _predictor = None

    @classmethod
    def get_predictor(cls):
        if cls._predictor is None:
            model_type = "vit_b"
            checkpoint_path = os.path.join(settings.BASE_DIR, 'models', 'sam', 'sam_vit_b_01ec64.pth')
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
            
            # Use MPS (Metal) on Mac if available, else CPU
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"DEBUG: [SAM Service] Loading model to {device}...")
            
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            cls._predictor = SamPredictor(sam)
            print("DEBUG: [SAM Service] Model loaded successfully.")
            
        return cls._predictor

    @staticmethod
    def get_wall_mask(image_bgr, hint_box=None):
        """
        Uses SAM to segment the wall.
        If hint_box is provided, it uses it to find the wall *around* that box.
        """
        predictor = SAMService.get_predictor()
        predictor.set_image(image_bgr)
        
        h, w = image_bgr.shape[:2]
        
        if hint_box:
            # If we know where the door is, the wall is likely around it.
            # We can prompt with points outside the door box.
            x1, y1, x2, y2 = hint_box
            
            # Points likely on the wall: top corner, side corners
            input_points = np.array([
                [w // 10, h // 10], # Top-leftish
                [w * 9 // 10, h // 10], # Top-rightish
                [x1 - 20 if x1 > 20 else 5, (y1 + y2) // 2], # Left of door
                [x2 + 20 if x2 < w - 20 else w - 5, (y1 + y2) // 2], # Right of door
            ])
            input_labels = np.array([1, 1, 1, 1])
        else:
            # Generic wall points
            input_points = np.array([
                [w // 2, h // 10],   # Top center
                [w // 10, h // 2],   # Left center
                [w * 9 // 10, h // 2] # Right center
            ])
            input_labels = np.array([1, 1, 1])

        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Pick the best mask (usually the one with highest score that covers a large area)
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx]

    @staticmethod
    def find_largest_vertical_plane(mask):
        """
        Finds the largest rectangular-ish region in the mask.
        """
        import cv2
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

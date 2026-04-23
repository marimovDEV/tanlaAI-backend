import os
from django.conf import settings

class SAMService:
    _instance = None
    _predictor = None

    @classmethod
    def get_predictor(cls):
        if cls._predictor is None:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            model_type = "vit_b"
            checkpoint_path = os.path.join(settings.BASE_DIR, 'models', 'sam', 'sam_vit_b_01ec64.pth')
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
            
            # Use CPU for background workers to ensure stability in subthreads
            device = "cpu"
            print(f"DEBUG: [SAM Service] Loading model to {device} (Thread-Safe)...")
            
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
        import numpy as np
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
    def get_opening_candidates(image_bgr):
        """
        Structural search for architectural openings:
        1. Find the wall.
        2. Find 'holes' in the wall.
        3. Return candidates.
        """
        import cv2
        import numpy as np
        
        # 1. Get wall mask
        wall_mask = SAMService.get_wall_mask(image_bgr)
        h, w = wall_mask.shape[:2]
        
        # 2. Find the 'inverses'/holes in the wall
        # We look for large vertical voids
        wall_uint8 = (wall_mask * 255).astype(np.uint8)
        
        # Invert: holes are now 255
        holes_mask = cv2.bitwise_not(wall_uint8)
        
        # Clean the holes mask (ignore tiny noise)
        kernel = np.ones((5, 5), np.uint8)
        holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours of holes
        contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area_ratio = (cw * ch) / float(w * h)
            
            # Doors are usually > 3% of image area and vertical
            if area_ratio > 0.02 and ch > cw * 1.2:
                candidates.append((x, y, x + cw, y + ch))
                
        return candidates, wall_mask

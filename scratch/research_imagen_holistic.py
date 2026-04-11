import os
import io
from PIL import Image
from google.genai import types
from shop.services import AIService

def test_holistic_replacement():
    client = AIService.get_gemini_client()
    
    # Mock paths
    room_path = "path/to/test_room.jpg"
    door_path = "path/to/test_door.jpg"
    
    if not os.path.exists(room_path) or not os.path.exists(door_path):
        print("Please provide test images.")
        return

    room_img = Image.open(room_path).convert("RGB")
    door_img = Image.open(door_path).convert("RGB")
    
    # Imagine a mask for the door
    w, h = room_img.size
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    # Central door-ish area
    draw = ImageDraw.Draw(mask)
    draw.rectangle([w*0.4, h*0.2, w*0.6, h*0.85], fill=255)
    
    room_buf = io.BytesIO()
    room_img.save(room_buf, format='PNG')
    
    door_buf = io.BytesIO()
    door_img.save(door_buf, format='PNG')
    
    mask_buf = io.BytesIO()
    mask.save(mask_buf, format='PNG')
    
    prompt = (
        "Re-design the interior doorway of this room. "
        "Replace the area marked in the mask with the EXACT door shown in Reference 2. "
        "The new door must match the proportions, handle material, and glass style of the door in Reference 2. "
        "Maintain the exact lighting and shadows of the room in Reference 1."
    )
    
    response = client.models.edit_image(
        model='imagen-3.0-capability-001',
        prompt=prompt,
        reference_images=[
            types.RawReferenceImage(reference_id=1, reference_image=types.Image(image_bytes=room_buf.getvalue())),
            types.RawReferenceImage(reference_id=2, reference_image=types.Image(image_bytes=door_buf.getvalue())),
            types.MaskReferenceImage(
                reference_id=3, 
                reference_image=types.Image(image_bytes=mask_buf.getvalue()),
                config=types.MaskReferenceConfig(mask_mode='MASK_MODE_USER_PROVIDED')
            ),
        ],
        config=types.EditImageConfig(
            edit_mode='EDIT_MODE_INPAINT_EDIT',
            number_of_images=1,
            output_mime_type='image/png',
        ),
    )
    
    if response.generated_images:
        print("Success! Generated image bytes received.")
    else:
        print("Failed to generate.")

if __name__ == "__main__":
    test_holistic_replacement()

import os, sys, django, io
import cv2
import numpy as np

sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from openai import OpenAI
from shop.services import AIService

def test_dalle2_inpaint():
    client = OpenAI()
    
    # Create dummy room if needed
    if not os.path.exists('dummy.png'):
        img = np.zeros((800, 600, 3), dtype=np.uint8)
        img[:] = (200, 200, 200) # grey wall
        img[600:, :, :] = (100, 100, 100) # dark floor
        cv2.rectangle(img, (200, 100), (400, 600), (0, 0, 255), -1) # red door
        cv2.imwrite('dummy.png', img)
        
    room_bgr = cv2.imread('dummy.png')
    h, w = room_bgr.shape[:2]
    
    # target square size
    sq_size = max(w, h)
    
    # pad room to square
    padded_room = np.zeros((sq_size, sq_size, 3), dtype=np.uint8)
    y_off = (sq_size - h) // 2
    x_off = (sq_size - w) // 2
    padded_room[y_off:y_off+h, x_off:x_off+w] = room_bgr
    
    # create mask
    mask = np.zeros((sq_size, sq_size, 4), dtype=np.uint8)
    mask[:, :, 3] = 255 # opaque mask
    
    # door box on original image
    box_x1, box_y1, box_x2, box_y2 = 200, 100, 400, 600
    
    # hole in mask (fully transparent where we want DALL-E to paint)
    mask[y_off+box_y1:y_off+box_y2, x_off+box_x1:x_off+box_x2] = (0, 0, 0, 0)
    
    # must convert room to RGBA and transparent hole
    img_rgba = cv2.cvtColor(padded_room, cv2.COLOR_BGR2BGRA)
    img_rgba[y_off+box_y1:y_off+box_y2, x_off+box_x1:x_off+box_x2, 3] = 0
    
    room_png = cv2.imencode('.png', img_rgba)[1].tobytes()
    mask_png = cv2.imencode('.png', mask)[1].tobytes()
    
    try:
        res = client.images.edit(
            model="dall-e-2",
            image=room_png,
            mask=mask_png,
            prompt="An empty interior wall with continuous floor. Seamless interior background. Plain wallpaper.",
            n=1,
            size="1024x1024"
        )
        print("SUCCESS:", res.data[0].url)
    except Exception as e:
        print("ERROR:", e)

test_dalle2_inpaint()

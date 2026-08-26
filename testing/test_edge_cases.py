import requests
import cv2
import numpy as np
import io

print("="*50)
print("EDGE CASE TEST")
print("="*50)

def post_image(buf, label):
    img_bytes = buf.tobytes()
    r = requests.post(
        'http://127.0.0.1:8000/api/inspect',
        files={'file': ('test.jpg', img_bytes, 'image/jpeg')},  # <-- fixed
        timeout=10
    )
    data = r.json()
    if r.status_code == 200:
        print(f"  {label:<20} → {data['status']} ({data.get('confidence', '?')}%)")
    else:
        print(f"  {label:<20} → ERROR {r.status_code}: {data.get('detail', data)}")

# Test 1 - Pure black image
black = np.zeros((224, 224, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', black)
post_image(buf, "Black image")

# Test 2 - Pure white image
white = np.ones((224, 224, 3), dtype=np.uint8) * 255
_, buf = cv2.imencode('.jpg', white)
post_image(buf, "White image")

# Test 3 - Random noise image
noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', noise)
post_image(buf, "Random noise")

# Test 4 - Very small image
small = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', small)
post_image(buf, "32x32 image")

# Test 5 - Very large image
large = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', large)
post_image(buf, "2048x2048 image")

# Test 6 - Grayscale image
gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
_, buf = cv2.imencode('.jpg', gray_bgr)
post_image(buf, "Grayscale image")
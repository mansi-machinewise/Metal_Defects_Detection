# Save as scripts\test_robustness.py
import cv2
import numpy as np
import requests
import glob
import os

TEST_DIR = r"D:\Metal_Defects\dataset_multiclass\test\images"
img_files = glob.glob(os.path.join(TEST_DIR, "*.jpg"))[:30]

def test_images(images_dict):
    results = {}
    for test_name, img_list in images_dict.items():
        correct = 0
        total = 0
        for img_bytes, expected_status in img_list:
            files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
            r = requests.post('http://127.0.0.1:8000/api/inspect', files=files)
            if r.status_code == 200:
                data = r.json()
                if data['status'] == expected_status:
                    correct += 1
                total += 1
        results[test_name] = f"{correct}/{total} ({correct/total*100:.1f}%)"
    return results

# Prepare test variations
normal = []
noisy = []
blurred = []
brightened = []
darkened = []

for img_path in img_files:
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Normal
    _, buf = cv2.imencode('.jpg', img)
    normal.append((buf.tobytes(), 'BAD'))

    # Noisy
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    noisy_img = cv2.add(img, noise)
    _, buf = cv2.imencode('.jpg', noisy_img)
    noisy.append((buf.tobytes(), 'BAD'))

    # Blurred
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    _, buf = cv2.imencode('.jpg', blurred_img)
    blurred.append((buf.tobytes(), 'BAD'))

    # Brightened
    bright_img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    _, buf = cv2.imencode('.jpg', bright_img)
    brightened.append((buf.tobytes(), 'BAD'))

    # Darkened
    dark_img = cv2.convertScaleAbs(img, alpha=0.5, beta=-30)
    _, buf = cv2.imencode('.jpg', dark_img)
    darkened.append((buf.tobytes(), 'BAD'))

print("Running robustness tests...")
print("Make sure server is running at http://127.0.0.1:8000\n")

results = test_images({
    'Normal Images':     normal,
    'Noisy Images':      noisy,
    'Blurred Images':    blurred,
    'Brightened Images': brightened,
    'Darkened Images':   darkened,
})

print("="*50)
print("ROBUSTNESS TEST RESULTS")
print("="*50)
for test_name, result in results.items():
    print(f"  {test_name:<25} {result}")
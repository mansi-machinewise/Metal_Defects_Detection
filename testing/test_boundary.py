# Save as scripts\test_boundary.py
import requests
import cv2
import numpy as np
import os
import glob

API_URL = "http://127.0.0.1:8000/api/inspect"
TEST_DIR = r"D:\Metal_Defects\dataset_multiclass\test\images"

def send_image(img_bgr, filename="test.jpg"):
    _, buf = cv2.imencode('.jpg', img_bgr)
    files = {'file': (filename, buf.tobytes(), 'image/jpeg')}
    r = requests.post(API_URL, files=files)
    if r.status_code == 200:
        return r.json()
    return None

img_files = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
sample_img = cv2.imread(img_files[0])

print("="*60)
print("BOUNDARY TEST RESULTS")
print("="*60)

# ---------------------------------------------------------------
# Test 1 — Minimum confidence threshold
# Change conf in config.yaml temporarily and test
# ---------------------------------------------------------------
print("\n1. CONFIDENCE THRESHOLD TEST")
print("-"*40)

for img_path in img_files[:10]:
    img = cv2.imread(img_path)
    result = send_image(img)
    if result:
        print(f"  {os.path.basename(img_path)[:40]:<40} "
              f"Status: {result['status']} | "
              f"Conf: {result['confidence']}% | "
              f"Defects: {result['defect_count']}")

# ---------------------------------------------------------------
# Test 2 — Single pixel defect (tiny defect region)
# ---------------------------------------------------------------
print("\n2. TINY DEFECT REGION TEST")
print("-"*40)

# Create image with very small defect area
tiny_defect_images = []
for img_path in img_files[:5]:
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    
    # Crop only 10x10 pixels from center and paste on blank
    cx, cy = W//2, H//2
    tiny = img[cy-5:cy+5, cx-5:cx+5]
    
    # Create blank image same size
    blank = np.ones((H, W, 3), dtype=np.uint8) * 128
    blank[cy-5:cy+5, cx-5:cx+5] = tiny
    
    result = send_image(blank, "tiny_defect.jpg")
    if result:
        print(f"  Tiny defect (10x10 px) → "
              f"Status: {result['status']} | "
              f"Conf: {result['confidence']}% | "
              f"Defects: {result['defect_count']}")

# ---------------------------------------------------------------
# Test 3 — Full image defect (entire image is defective)
# ---------------------------------------------------------------
print("\n3. FULL IMAGE DEFECT TEST")
print("-"*40)

# Use highly defective images (rolled-in_scale has most defects)
rs_images = glob.glob(os.path.join(TEST_DIR, "RS_*.jpg"))
for img_path in rs_images[:5]:
    img = cv2.imread(img_path)
    result = send_image(img, "full_defect.jpg")
    if result:
        print(f"  {os.path.basename(img_path)[:40]:<40} "
              f"Status: {result['status']} | "
              f"Conf: {result['confidence']}% | "
              f"Defects: {result['defect_count']}")

# ---------------------------------------------------------------
# Test 4 — Multiple overlapping defects
# ---------------------------------------------------------------
print("\n4. OVERLAPPING DEFECTS TEST")
print("-"*40)

# Combine two defective images by blending
for i in range(5):
    img1 = cv2.imread(img_files[i])
    img2 = cv2.imread(img_files[i+10])
    
    # Resize to same size
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Blend images (creates overlapping defects)
    blended = cv2.addWeighted(img1, 0.6, img2_resized, 0.4, 0)
    
    result = send_image(blended, "overlapping.jpg")
    if result:
        print(f"  Blended image {i+1} → "
              f"Status: {result['status']} | "
              f"Conf: {result['confidence']}% | "
              f"Defects: {result['defect_count']}")

# ---------------------------------------------------------------
# Test 5 — Summary
# ---------------------------------------------------------------
print("\n" + "="*60)
print("BOUNDARY TEST SUMMARY")
print("="*60)
print("  Test 1 — Confidence Threshold : completed")
print("  Test 2 — Tiny Defect Region   : completed")
print("  Test 3 — Full Image Defect    : completed")
print("  Test 4 — Overlapping Defects  : completed")
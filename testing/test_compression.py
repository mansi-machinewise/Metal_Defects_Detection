# Save as scripts\test_compression.py
import requests
import cv2
import os

img_path = r"D:\Metal_Defects\dataset_multiclass\test\images\crazing_103_jpg.rf.86ae815afa93e66177fa4e37646d7cf3.jpg"
img = cv2.imread(img_path)

print("="*50)
print("COMPRESSION TEST")
print("="*50)

for quality in [10, 30, 50, 70, 90, 100]:
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    files = {'file': ('test.jpg', buf.tobytes(), 'image/jpeg')}
    r = requests.post('http://127.0.0.1:8000/api/inspect', files=files)
    if r.status_code == 200:
        data = r.json()
        size_kb = len(buf) / 1024
        print(f"  Quality {quality:3d}% | Size: {size_kb:6.1f}KB | {data['status']} | Conf: {data['confidence']}%")
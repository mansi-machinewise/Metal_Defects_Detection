# Save as scripts\test_formats.py
import requests
import cv2
import os

img_path = r"D:\Metal_Defects\dataset_multiclass\test\images\crazing_103_jpg.rf.86ae815afa93e66177fa4e37646d7cf3.jpg"
img = cv2.imread(img_path)

formats = {
    'JPG':  ('.jpg',  [cv2.IMWRITE_JPEG_QUALITY, 95]),
    'PNG':  ('.png',  []),
    'BMP':  ('.bmp',  []),
    'WEBP': ('.webp', []),
}

print("="*50)
print("FILE FORMAT TEST")
print("="*50)

for fmt_name, (ext, params) in formats.items():
    _, buf = cv2.imencode(ext, img, params)
    files = {'file': (f'test{ext}', buf.tobytes(), 'image/jpeg')}
    r = requests.post('http://127.0.0.1:8000/api/inspect', files=files)
    if r.status_code == 200:
        data = r.json()
        print(f"  {fmt_name:<10} → {data['status']} | {data['confidence']}%")
    else:
        print(f"  {fmt_name:<10} → ERROR {r.status_code}")
# Save as test_speed.py
import requests
import time
import glob

img_files = glob.glob(r'D:\Metal_Defects\dataset_multiclass\test\images\*.jpg')[:50]

times = []
for img_path in img_files:
    with open(img_path, 'rb') as f:
        start = time.time()
        r = requests.post('http://127.0.0.1:8000/api/inspect', files={'file': f})
        end = time.time()
        times.append((end - start) * 1000)

print(f"Average response time: {sum(times)/len(times):.1f}ms")
print(f"Min response time: {min(times):.1f}ms")
print(f"Max response time: {max(times):.1f}ms")
print(f"Images processed per second: {1000/(sum(times)/len(times)):.2f}")
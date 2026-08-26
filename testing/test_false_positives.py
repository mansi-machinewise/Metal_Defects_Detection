import requests
import glob
import os

# Test with known GOOD images (no defects)
good_images = glob.glob(r'D:\Metal_Defects\dataset_multiclass\test\images\*.jpg')

false_positives = 0
true_negatives = 0

for img_path in good_images[:20]:  # test first 20
    with open(img_path, 'rb') as f:
        r = requests.post(
            'http://127.0.0.1:8000/api/inspect',
            files={'file': (img_path, f, 'image/jpeg')},
            timeout=10
        )
        data = r.json()
        filename = os.path.basename(img_path)  # <-- fixed
        print(f"{filename}: {data['status']} ({data['confidence']}%)")
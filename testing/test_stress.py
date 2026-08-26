import requests
import glob
import os

TEST_DIR = r"D:\Metal_Defects\dataset_multiclass\test\images"
img_files = glob.glob(os.path.join(TEST_DIR, "*.jpg"))

passed = 0
failed = 0
errors = []

print(f"Stress testing {len(img_files)} images...\n")

for i, img_path in enumerate(img_files):
    with open(img_path, 'rb') as f:
        r = requests.post(
            'http://127.0.0.1:8000/api/inspect',
            files={'file': (img_path, f, 'image/jpeg')},  # <-- fixed
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            required = ['status', 'confidence', 'defect_type', 'defects', 'annotated_image']
            missing = [k for k in required if k not in data]
            if missing:
                errors.append(f"{os.path.basename(img_path)}: missing fields {missing}")
                failed += 1
            else:
                passed += 1
        else:
            failed += 1
            errors.append(f"{os.path.basename(img_path)}: HTTP {r.status_code}")

    if (i+1) % 50 == 0:
        print(f"  Processed {i+1}/{len(img_files)}...")

print("\n" + "="*50)
print("STRESS TEST RESULTS")
print("="*50)
print(f"  Total images    : {len(img_files)}")
print(f"  Passed          : {passed}")
print(f"  Failed          : {failed}")
print(f"  Success rate    : {passed/len(img_files)*100:.1f}%")
if errors:
    print(f"\n  Errors:")
    for e in errors[:10]:
        print(f"    {e}")
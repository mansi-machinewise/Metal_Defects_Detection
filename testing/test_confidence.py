import requests
import glob

img_files = glob.glob(r'D:\Metal_Defects\dataset_multiclass\test\images\*.jpg')

print(f"Found {len(img_files)} image(s)")
if not img_files:
    print("ERROR: No .jpg files found. Check the path exists and has images.")
    exit(1)

confidences = []
errors = []

for img_path in img_files:
    with open(img_path, 'rb') as f:
        try:
            r = requests.post(
                'http://127.0.0.1:8000/api/inspect',
                files={'file': (img_path, f, 'image/jpeg')},  # <-- changed here
                timeout=10
            )
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect. Is the server running on port 8000?")
            exit(1)

        if r.status_code == 200:
            data = r.json()
            if 'confidence' in data:
                confidences.append(data['confidence'])
            else:
                print(f"WARNING: 'confidence' key missing in response for {img_path}")
                print(f"  Response was: {data}")
                errors.append(img_path)
        else:
            print(f"WARNING: Status {r.status_code} for {img_path}")
            print(f"  Response: {r.text[:200]}")
            errors.append(img_path)

print(f"\nProcessed: {len(confidences)} succeeded, {len(errors)} failed")

if not confidences:
    print("ERROR: No confidence values collected. See warnings above.")
    exit(1)

print(f"Average confidence: {sum(confidences)/len(confidences):.2f}%")
print(f"Min confidence:     {min(confidences):.2f}%")
print(f"Max confidence:     {max(confidences):.2f}%")
print(f"Below 30%:          {sum(1 for c in confidences if c < 30)}")
print(f"Above 90%:          {sum(1 for c in confidences if c > 90)}")
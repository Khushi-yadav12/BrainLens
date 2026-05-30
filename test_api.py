import urllib.request
import json
import os

url = "http://127.0.0.1:5000/analyze"
image_path = r"c:\Users\Administrator\OneDrive\Desktop\Projects\braintumor\brats_healthy_tumor_9k\images\healthy\healthy_BraTS2021_00003_AX_070.png"

# We must send multipart/form-data. Since we don't have requests, we'll construct it manually.
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = []
body.append(f"--{boundary}\r\n")
body.append(f"Content-Disposition: form-data; name=\"image\"; filename=\"{os.path.basename(image_path)}\"\r\n")
body.append("Content-Type: image/png\r\n\r\n")

with open(image_path, "rb") as f:
    img_data = f.read()

payload = "".join(body).encode("utf-8") + img_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(url, data=payload)
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

try:
    with urllib.request.urlopen(req) as response:
        print("Status Code:", response.status)
        print("Response JSON:", response.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Response JSON:", e.read().decode("utf-8"))

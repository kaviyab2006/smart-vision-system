import urllib.request
import requests
import os

SAMPLE_IMG = "sample.jpg"

if not os.path.exists(SAMPLE_IMG):
    print("Downloading sample image...")
    # Bus image from ultralytics repo, has person, bus, etc.
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", SAMPLE_IMG)

print("Sending to API...")
with open(SAMPLE_IMG, "rb") as f:
    res = requests.post("http://127.0.0.1:8000/detect", files={"file": f})

if res.status_code == 200:
    data = res.json()
    print("Hints:", data.get("hints"))
    print("Objects detected:", [o['name'] for o in data.get("objects", [])])
else:
    print("Error:", res.status_code, res.text)

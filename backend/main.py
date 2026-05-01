from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from detector import ObjectDetector

app = FastAPI(title="Object Detection Accessibility API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = ObjectDetector()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    hints, objects = detector.process_image(contents)
    
    return {
        "hints": hints,
        "objects": objects
    }

@app.get("/")
def read_root():
    return {"status": "Backend is running and YOLOv8 is loaded"}

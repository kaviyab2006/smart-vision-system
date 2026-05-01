from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='../yolov8n.pt'):
        # Load the YOLOv8 model from the ultralytics package
        self.model = YOLO(model_path)
        self.reference_classes = ['chair', 'dining table', 'couch', 'bed', 'tv', 'laptop', 'sink', 'refrigerator', 'book', 'bottle', 'cup']

        # Synonyms or similar items mapping 
        self.class_mapping = {
            'cell phone': 'phone',
            'handbag': 'bag',
            'suitcase': 'bag',
            'backpack': 'bag',
            'dining table': 'table',
            'potted plant': 'plant',
            'tv': 'television'
        }

    def process_image(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return [], []

        # Run YOLO inference
        results = self.model(img)
        
        detected_objects = []
        img_height, img_width = img.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls_id]
                
                # Apply mapping
                name = self.class_mapping.get(name, name)
                
                detected_objects.append({
                    'name': name,
                    'confidence': conf,
                    'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2,
                    'area': (x2 - x1) * (y2 - y1)
                })
                
        return self._generate_hints(detected_objects, img_width, img_height), detected_objects

    def _generate_hints(self, objects, img_width, img_height):
        # Separate reference objects (furniture, fixed items) and target objects (keys, wallet, phone, bag, etc.)
        references = [obj for obj in objects if obj['name'] in self.reference_classes or obj['name'] == 'table']
        targets = [obj for obj in objects if obj not in references and obj['name'] != 'person']
        
        hints = []
        
        for target in targets:
            name = target['name']
            cx = target['center_x']
            cy = target['center_y']
            
            # Determine horizontal position relative to user
            if cx < img_width * 0.33:
                pos = "on your left"
            elif cx > img_width * 0.67:
                pos = "on your right"
            else:
                pos = "in front of you"
                
            # Find closest reference object
            closest_ref = None
            min_dist = float('inf')
            relation = "near"
            
            for ref in references:
                ref_cx, ref_cy = ref['center_x'], ref['center_y']
                dist = np.sqrt((cx - ref_cx)**2 + (cy - ref_cy)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_ref = ref
                    
                    # Check if "on" (target bottom is within reference top half, target center x within reference x bounds)
                    if (ref['box']['x1'] < cx < ref['box']['x2']) and (ref['box']['y1'] < target['box']['y2'] < ref['box']['y2']):
                        relation = "on"
                    else:
                        relation = "near"
                        
            if closest_ref:
                ref_name = closest_ref['name']
                hint = f"The {name} is {pos}, {relation} the {ref_name}."
            else:
                hint = f"The {name} is {pos}."
                
            hints.append(hint)
            
        return hints

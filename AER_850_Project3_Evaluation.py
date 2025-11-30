# ===============================================================
#  AER850 Project 3 - Evaluation Script
# ===============================================================

import cv2
import os
from ultralytics import YOLO

#Configuration of our folders
MODEL_PATH = "runs/detect/Project3_model/weights/best.pt"   
EVAL_FOLDER = "data/data/evaluation"                        
OUTPUT_FOLDER = "evaluation_outputs"                        

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Loading the trained YOLO model 
model = YOLO(MODEL_PATH)
print(f"[INFO] Loaded trained model: {MODEL_PATH}")


def get_color(label):
    lname = label.lower()

    if "connector" in lname:
        return (255, 170, 80)    
    elif "ic" in lname:
        return (120, 220, 120)   
    elif "capacitor" in lname:
        return (255, 230, 120)   
    elif "resistor" in lname:
        return (180, 140, 255)   
    elif "diode" in lname:
        return (120, 200, 255)   
    elif "led" in lname:
        return (255, 120, 180)    
    elif "button" in lname:
        return (255, 150, 120)    
    else:
        return (200, 200, 255)    

#Drawing our bounding boxes
def draw_boxes(img, results, names):

    img_h, img_w = img.shape[:2]
    base_scale = img_w / 2000

    font_scale = max(0.4, 0.7 * base_scale)
    thickness = max(1, int(2 * base_scale))

    prediction_summary = []

    for box in results[0].boxes:

        cls = int(box.cls[0])
        label = names[cls]
        conf = float(box.conf[0])
        text = f"{label} {conf:.2f}"

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = get_color(label)

        # Storing the prediction info
        prediction_summary.append({
            "class": label,
            "confidence": round(conf, 3),
            "bbox": (x1, y1, x2, y2)
        })

   
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 10, y1), color, -1)


        cv2.putText(
            img, text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness
        )

    return img, prediction_summary

#Evaluating all of the images in our folder
all_results = {}

for filename in os.listdir(EVAL_FOLDER):

    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(EVAL_FOLDER, filename)
    print(f"[INFO] Evaluating: {filename}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not read {filename}")
        continue

    # Our YOLO prediction
    results = model(img, conf=0.35, iou=0.7, verbose=False)

    # Draw boxes with our different colours 
    annotated, preds = draw_boxes(img.copy(), results, model.names)

    # Saving the annotated result
    out_path = os.path.join(OUTPUT_FOLDER, f"annotated_{filename}")
    cv2.imwrite(out_path, annotated)
    print(f"[INFO] Saved annotated image → {out_path}")

    all_results[filename] = preds

#Creating a summary text file for analysis in the report
summary_path = os.path.join(OUTPUT_FOLDER, "evaluation_summary.txt")

with open(summary_path, "w") as f:
    f.write("AER850 Project 3 Evaluation Summary\n")
    f.write("===================================\n\n")

    for img_name, preds in all_results.items():
        f.write(f"\nImage: {img_name}\n")
        f.write("-----------------------------------\n")

        if len(preds) == 0:
            f.write("No detections.\n")
        else:
            for p in preds:
                f.write(f"- {p['class']} (conf {p['confidence']}) | bbox={p['bbox']}\n")

print(f"\n[INFO] Summary saved → {summary_path}")
print("[INFO] Evaluation Completed Successfully.")


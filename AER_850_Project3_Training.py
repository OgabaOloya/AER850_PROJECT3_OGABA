# ===============================================================
#  AER850 Project 3 - YOLOv11 Training Script
# ===============================================================

from ultralytics import YOLO

def main():

    #Loading the pretrained YOLO model 
    model = YOLO("yolo11n.pt")   

#Training our model 
    model.train(
        data="data/data/data.yaml",  
        epochs=200,                  # Epoch count -- maximum allowed by our project to hopefully yield the best results 
        batch=4,                     # Batch size 
        imgsz=1200,                  # Image size
        device=0,                    # Our GPU device ID -- we set it to 0 to run off GPU
        patience=20,                 # Early stopping patience 
        name="Project3_model"       # Our training model save name 
    )

if __name__ == "__main__":
    main()

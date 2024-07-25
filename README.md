# Sign-Language-Detection-and-Image-Generation
This project focuses on developing and training a YOLOv8 model for sign language detection, followed by generating related images using Stable Diffusion. Below is a step-by-step guide on how to set up, train, validate, and use the custom model, as well as integrate it with Stable Diffusion for image generation.
# Perequisites
+ Ensure you have access to a GPU. You can check this by running the following command:
`!nvidia-smi`
+ If there are any issues, navigate to Edit -> Notebook settings -> Hardware accelerator, set it to GPU, and then click Save.
# Projecr Structure
### 1. Install Dependencies:
Install the required libraries including YOLOv8 and Roboflow.
`!pip install ultralytics==8.0.20
!pip install roboflow
`
### 2. Import Libraries and Initialize YOLOv8:
`import ultralytics
from ultralytics import YOLO
`
### 3. Download Custom Dataset:
Download the dataset using Roboflow.
`from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key_here")
project = rf.workspace("projects-sbfcm").project("sign-language-detection-u0eat")
version = project.version(1)
dataset = version.download("yolov8")
`
### 4. Train Custom YOLOv8 Model:
Train the YOLOv8 model on the custom dataset.
`!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=30 batch=4 imgsz=640 plots=True
`
### 5. Validate the Custom Model:
Validate the trained model.
`!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
`
### 6. Inference with Custom Model:
Run inference on a sample image and save the results.
`!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.25 source="your_image.jpg" save=True save_txt=True
`
### 7. Generate Images using Stable Diffusion:
Generate images based on the detected classes.

```from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast

prompt = "Detected Classes: [your_detected_classes_here]"
stable_diff_pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
stable_diff_pipe.to("cuda")

with autocast("cuda"):
    image = stable_diff_pipe(prompt, guidance_scale=12.5).images[0]
```

### 8. Live Detection using YOLOv8:
Implement live detection using the camera.
``` import cv2
from ultralytics import YOLO

model = YOLO('path_to_your_model.pt')
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
```
# Repository Structure
+ `datasets/`: Directory containing the dataset.
+ `runs/detect/train/`: Directory containing the training results.
+ `runs/detect/predict/`: Directory containing the inference results.

# Notes
+ Ensure to replace placeholders with actual paths and values.
+ The project is configured to run on GPU for faster processing
+ Make sure to have a valid API key from Roboflow.
# References
+ [Ultralytics YOLO Docs](https://docs.ultralytics.com/usage/cli/)
+ [Roboflow](https://roboflow.com/)

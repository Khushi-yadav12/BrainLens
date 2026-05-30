import os
import glob
import random
import torch
import torchvision.transforms.functional as TF
from torchvision import models
import torch.nn as nn
from PIL import Image

CLASSIFIER_PATH = r"c:\Users\Administrator\OneDrive\Desktop\Projects\braintumor\model\brain_tumor_classifier_multiplanar.pth"

def evaluate_fast():
    device = torch.device("cpu")
    print(f"Loading Model on {device}...", flush=True)
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    state_dict = torch.load(CLASSIFIER_PATH, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Find images
    healthy_files = glob.glob(r"c:\Users\Administrator\OneDrive\Desktop\Projects\braintumor\brats_healthy_tumor_9k\images\healthy\*.png")
    tumor_files = glob.glob(r"c:\Users\Administrator\OneDrive\Desktop\Projects\braintumor\brats_healthy_tumor_9k\images\tumor\*.png")
    
    # Shuffle and pick 100 of each to be fast
    random.shuffle(healthy_files)
    random.shuffle(tumor_files)
    healthy_files = healthy_files[:100]
    tumor_files = tumor_files[:100]
    
    print(f"Evaluating {len(healthy_files)} Healthy and {len(tumor_files)} Tumor images...")

    correct_healthy = 0
    correct_tumor = 0

    with torch.no_grad():
        for i, img_path in enumerate(healthy_files):
            if i % 10 == 0: print(f"Processing Healthy {i}/{len(healthy_files)}", flush=True)
            pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
            tensor_img = TF.to_tensor(pil_img)
            tensor_img = TF.normalize(tensor_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)
            out = model(tensor_img)
            _, predicted = out.max(1)
            if predicted.item() == 0:
                correct_healthy += 1
                
        for i, img_path in enumerate(tumor_files):
            if i % 10 == 0: print(f"Processing Tumor {i}/{len(tumor_files)}", flush=True)
            pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
            tensor_img = TF.to_tensor(pil_img)
            tensor_img = TF.normalize(tensor_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)
            out = model(tensor_img)
            _, predicted = out.max(1)
            if predicted.item() == 1:
                correct_tumor += 1
                
    acc_h = (correct_healthy / len(healthy_files)) * 100
    acc_t = (correct_tumor / len(tumor_files)) * 100
    total_acc = ((correct_healthy + correct_tumor) / (len(healthy_files) + len(tumor_files))) * 100
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS (N=200)")
    print("="*40)
    print(f"Overall Accuracy:   {total_acc:.2f}%")
    print(f"Healthy Accuracy:   {acc_h:.2f}% ({correct_healthy}/{len(healthy_files)})")
    print(f"Tumor Accuracy:     {acc_t:.2f}% ({correct_tumor}/{len(tumor_files)})")
    print("="*40)

if __name__ == "__main__":
    evaluate_fast()

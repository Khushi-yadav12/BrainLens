import os
import glob
import torch
import torchvision.transforms.functional as TF
from torchvision import models
import torch.nn as nn
from PIL import Image

CLASSIFIER_PATH = r"c:\Users\Administrator\OneDrive\Desktop\Projects\braintumor\model\brain_tumor_classifier_multiplanar.pth"

def test_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    state_dict = torch.load(CLASSIFIER_PATH, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    pil_cls = Image.open(image_path).convert('RGB').resize((224, 224))
    tensor_cls = TF.to_tensor(pil_cls)
    tensor_cls = TF.normalize(tensor_cls, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor_cls)
        prob = torch.softmax(out, dim=1)
        _, predicted = out.max(1)
        
    print(f"File: {os.path.basename(image_path)}")
    print(f"Logits: {out.cpu().numpy()}")
    print(f"Probabilities: {prob.cpu().numpy()}")
    print(f"Predicted Class: {predicted.item()}")
    print("-" * 50)

if __name__ == "__main__":
    healthy_files = glob.glob(r"c:\Users\Administrator\OneDrive\Desktop\Projects\braintumor\brats_healthy_tumor_9k\images\healthy\*.png")
    tumor_files = glob.glob(r"c:\Users\Administrator\OneDrive\Desktop\Projects\braintumor\brats_healthy_tumor_9k\images\tumor\*.png")
    
    if healthy_files:
        test_inference(healthy_files[0])
    if tumor_files:
        test_inference(tumor_files[0])

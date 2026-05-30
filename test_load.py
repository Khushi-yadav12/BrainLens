import torch
from acu_net import ACUNet
import sys
import traceback

try:
    print("Loading model...")
    model = ACUNet(in_channels=3, out_channels=1)
    state_dict = torch.load('model/brain_tumor_acunet.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    print("Success!")
except Exception as e:
    print("Error:", e)
    traceback.print_exc()

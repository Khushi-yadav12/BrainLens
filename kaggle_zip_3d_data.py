# ==============================================================================
# BrainLens — 3D Dataset Zipping Utility for Kaggle
# ==============================================================================
# Instructions:
#   1. Copy this script into a NEW Kaggle cell at the bottom of your notebook.
#   2. Run the cell AFTER you have run the main training script.
#   3. This will gather EXACTLY the 300 Tumor and 300 Healthy 3D NIfTI files
#      that were used for training and zip them into a single file!
# ==============================================================================

import os
import glob
import shutil
from tqdm import tqdm

def zip_full_3d_dataset(num_patients=300):
    print(f"\n[INFO] Gathering {num_patients} Healthy and {num_patients} Tumor 3D brains...")
    
    # Define directories
    brats_root = "/kaggle/working/BraTS2021_Extracted"
    oasis_root = "/kaggle/working/huggingface_oasis"
    
    zip_dir = "/kaggle/working/Full_3D_Dataset"
    yes_dir = os.path.join(zip_dir, "Yes_Tumor_3D")
    no_dir = os.path.join(zip_dir, "No_Tumor_3D")
    
    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)
    
    # 1. Copy Tumor Brains (BraTS)
    t1_files = sorted(glob.glob(os.path.join(brats_root, "**", "*t1ce.nii.gz"), recursive=True))
    patient_folders = list(dict.fromkeys([os.path.dirname(f) for f in t1_files]))[-num_patients:]
    
    if not patient_folders:
        print("[ERROR] Could not find BraTS extracted data. Did you run the main script first?")
        return
        
    print(f"[INFO] Copying {len(patient_folders)} BraTS Tumor NIfTI files...")
    for pf in tqdm(patient_folders):
        t1_path = glob.glob(os.path.join(pf, "*t1ce.nii.gz"))
        if t1_path:
            shutil.copy(t1_path[0], yes_dir)
            
    # 2. Copy Healthy Brains (OASIS)
    healthy_files = sorted(glob.glob(os.path.join(oasis_root, "**", "*masked.nii.gz"), recursive=True))[:num_patients]
    
    if not healthy_files:
        print("[ERROR] Could not find OASIS extracted data. Did you run the main script first?")
        return
        
    print(f"[INFO] Copying {len(healthy_files)} OASIS Healthy NIfTI files...")
    for f in tqdm(healthy_files):
        shutil.copy(f, no_dir)
        
    # 3. Zip it all up!
    print(f"\n[INFO] Compressing all {num_patients * 2} 3D brains into a ZIP file...")
    print(f"[INFO] (This may take 5-10 minutes. Do not close Kaggle!)")
    
    shutil.make_archive("/kaggle/working/Full_3D_Dataset", 'zip', zip_dir)
    print(f"[DONE] Successfully created Full_3D_Dataset.zip!")
    
    # Clean up the unzipped copy to save Kaggle disk space
    shutil.rmtree(zip_dir)

if __name__ == "__main__":
    zip_full_3d_dataset(num_patients=400)

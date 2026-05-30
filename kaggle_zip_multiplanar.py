# ==============================================================================
# BrainLens — Zip Extracted Multi-Planar Dataset for Download
# ==============================================================================
# Instructions:
#   1. Copy this script into a NEW cell at the bottom of your Kaggle notebook.
#   2. Run the cell AFTER the training script has finished (or at least after 
#      the extraction phase has completed).
#   3. It will zip the 1,800 optimal slices and masks into a single file.
#   4. Refresh your Kaggle Output panel and download "MultiPlanar_Dataset.zip".
# ==============================================================================

import os
import shutil

def zip_multiplanar_dataset():
    source_dir = "/kaggle/working/extracted_multiplanar"
    output_zip = "/kaggle/working/MultiPlanar_Dataset"  # shutil.make_archive adds .zip automatically
    
    if not os.path.exists(source_dir):
        print(f"[ERROR] The directory {source_dir} does not exist!")
        print("Please make sure you ran the main training script first so the images are extracted.")
        return
        
    print(f"\n[INFO] Zipping the mathematically extracted multi-planar dataset...")
    print(f"       Source: {source_dir}")
    
    # Create the zip archive
    shutil.make_archive(output_zip, 'zip', source_dir)
    
    print(f"\n[DONE] Successfully created: {output_zip}.zip")
    print("       You can now download this from the Kaggle Output panel!")

if __name__ == "__main__":
    zip_multiplanar_dataset()

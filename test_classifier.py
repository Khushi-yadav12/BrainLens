import classifier
print("Classifier loaded successfully")
label, conf = classifier.classify("data_slices/yes/BraTS-GLI-00000-000-t1c_z070.png")
print("Classified:", label, conf)

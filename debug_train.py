import sys, traceback, random
from pathlib import Path

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    class BinaryMRIDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = list(image_paths)
            self.labels = list(labels)
            self.transform = transform
        def __len__(self): return len(self.image_paths)
        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert("RGB")
            label = int(self.labels[idx])
            if self.transform: img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)

    yes_files = list(Path("data_slices/yes").glob("*.png"))
    no_files  = list(Path("data_slices/no").glob("*.png"))
    print(f"yes: {len(yes_files)}, no: {len(no_files)}")

    all_paths  = yes_files + no_files
    all_labels = [1]*len(yes_files) + [0]*len(no_files)
    combined = list(zip(all_paths, all_labels))
    random.seed(42); random.shuffle(combined)
    unzipped = list(zip(*combined))
    all_paths  = list(unzipped[0])
    all_labels = list(unzipped[1])

    split = int(0.8 * len(all_paths))
    train_paths, val_paths   = all_paths[:split], all_paths[split:]
    train_labels, val_labels = all_labels[:split], all_labels[split:]

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = BinaryMRIDataset(train_paths, train_labels, train_tfm)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    print(f"DataLoader OK: {len(train_ds)} samples")

    model = models.vgg16_bn(weights="IMAGENET1K_V1")
    model.classifier[6] = nn.Linear(4096, 2)
    for p in model.features.parameters(): p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    model.train()
    for i, (imgs, labels) in enumerate(train_loader):
        print(f"  batch {i}: imgs={imgs.shape}, labels={labels.shape}, dtype={labels.dtype}")
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        print(f"  loss={loss.item():.4f}")
        if i >= 1: break

    print("SUCCESS")
except Exception:
    traceback.print_exc()
    sys.exit(1)

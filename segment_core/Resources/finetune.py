
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler

import torch
import numpy as np
import albumentations as A

import argparse
import os
import math

def read_dataset_list(path):
    dataset_dirs = os.listdir(path)
    # data = {}
    images = []
    masks = []
    for data_dir in dataset_dirs:
        masks_path = os.path.join(path, data_dir, 'masks.npy')
        slices_path = os.path.join(path, data_dir, 'slices.npy')

        if (not os.path.exists(masks_path)) or (not os.path.exists(slices_path)):
            print(f'"{data_dir}" dataset folder doesnt contain masks.npy or slices.npy')
            continue

        img_list = [im for im in np.load(slices_path)]
        msk_list = [ms for ms in np.load(masks_path)]

        if len(img_list) != len(msk_list):
            raise ValueError(f'"{data_dir}" {len(img_list)} images but {len(msk_list)} masks')
        
        images.extend(img_list)
        masks.extend(msk_list)
        # data[prefix + ' ' + data_dir] = {'images': np.load(slices_path), 'masks': np.load(masks_path)}

    return {'images': images, 'masks': masks}


def train_val_split(data, val_prop):
    pivot = math.ceil(len(data['images']) * val_prop)
    data_train = {'images': data['images'][pivot:], 'masks': data['masks'][pivot:]}
    data_val = {'images': data['images'][:pivot], 'masks': data['masks'][:pivot]}
    return data_train, data_val


class PercentileNormalize(A.ImageOnlyTransform):
    def __init__(self, p_low=2.5, p_high=97.5, always_apply=True, p=1.0):
        super().__init__(p=p)
        self.p_low = p_low
        self.p_high = p_high

    def apply(self, img, **params):
        img = img.astype(np.float32)

        low, high = np.percentile(img, [self.p_low, self.p_high])

        img = np.clip(img, low, high)

        img = (img - low) / (high - low + 1e-8)

        img = (img * 255.0).astype(np.uint8)

        return img


class SegmentationDataset(Dataset):

    def __init__(self, data, transform=None):

        self.images = data['images']
        self.masks = data['masks']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        if self.transform:
            transformed = self.transform(image=image, target=mask)
            image = transformed['image']
            mask = transformed['target']

        return image, mask
    

req_size = 512
train_transform = A.Compose([
    A.RandomSizedCrop(min_max_height=[256, 1024], size = [512, 512], w2h_ratio=1, p = 1),
    A.RandomBrightnessContrast(p = 0.5, brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.1], brightness_by_max=False),
    A.GaussNoise(p = 0.1, std_range=(0.02, 0.05)),
    A.GaussianBlur(p = 0.1, sigma_limit = [0.1, 0.2]),

    A.CLAHE(
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    p=0.3
    ),
    A.GridDistortion(
    num_steps=5,
    distort_limit=0.2,
    p=0.2
    ),

    PercentileNormalize(p_low=0),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomRotate90(p=0.5),
    
    A.Normalize(),
], additional_targets={'target': 'mask'})

eval_transform = A.Compose([
    A.RandomSizedCrop(min_max_height=[512, 512], size = [512, 512], w2h_ratio=1, p = 1),
    PercentileNormalize(p_low=0),
    A.Normalize(),
], additional_targets={'target': 'mask'})


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, reduction = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, 1, H, W]
        # targets: [B, H, W]

        probs = torch.sigmoid(logits)
        targets = targets.unsqueeze(1).float()

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def Train(model, device, lr, base_data_path, user_data_path, base_prop, val_prop, batchsize, max_epochs):
    base_data = read_dataset_list(base_data_path)
    user_data = read_dataset_list(user_data_path)

    base_data_train, base_data_val = train_val_split(base_data, val_prop)
    user_data_train, user_data_val = train_val_split(user_data, val_prop)

    base_train_dataset = SegmentationDataset(base_data_train, train_transform)
    base_val_dataset = SegmentationDataset(base_data_val, eval_transform)

    user_train_dataset = SegmentationDataset(user_data_train, train_transform)
    user_val_dataset = SegmentationDataset(user_data_val, eval_transform)

    train_dataset = ConcatDataset([base_train_dataset, user_train_dataset])
    val_dataset = ConcatDataset([base_val_dataset, user_val_dataset])

    base_weights = torch.ones(len(base_train_dataset)) * base_prop / len(base_train_dataset)
    user_weights = torch.ones(len(user_train_dataset)) * (1 - base_prop) / len(user_train_dataset)

    train_weights = torch.cat([base_weights, user_weights])

    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         
        factor=0.2,    
        patience=5,            
        min_lr=1e-9,
        threshold = 1e-3          
    )

    for epoch in range(max_epochs):
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model.forward(images)

            optimizer.zero_grad()
            loss = bce_loss(preds, masks.unsqueeze(1)) + dice_loss(preds, masks)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--base_data_path", required=True)
    parser.add_argument("--user_data_path", required=True)
    parser.add_argument("--base_prop", type=float, required=True)
    parser.add_argument("--val_prop", type=float, required=True)
    parser.add_argument("--batchsize", type=int, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(
        args.model_path,
        map_location=device,
        weights_only=False,
    )
    model = model.to(device)

    Train(
        model=model,
        device=device,
        lr=args.lr,
        base_data_path=args.base_data_path,
        user_data_path=args.user_data_path,
        base_prop=args.base_prop,
        val_prop=args.val_prop,
        batchsize=args.batchsize,
        max_epochs=args.max_epochs,
    )

    torch.save(model, args.model_path)

from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np
import albumentations as A

import argparse
import os
import math
import ctypes
from datetime import datetime

import metrics

def read_dataset_list(path):
    dataset_dirs = os.listdir(path)
    # data = {}
    images = []
    masks = []
    datasets = []
    for data_dir in dataset_dirs:
        masks_path = os.path.join(path, data_dir, 'numpy', 'masks.npy')
        slices_path = os.path.join(path, data_dir, 'numpy', 'slices.npy')

        if (not os.path.exists(masks_path)) or (not os.path.exists(slices_path)):
            print(f'"{data_dir}" dataset folder doesnt contain masks.npy or slices.npy')
            continue

        img_list = [im for im in np.load(slices_path)]
        msk_list = [ms for ms in np.load(masks_path)]

        if len(img_list) != len(msk_list):
            raise ValueError(f'"{data_dir}" {len(img_list)} images but {len(msk_list)} masks')
        
        images.extend(img_list)
        masks.extend(msk_list)
        datasets.append(data_dir)
        # data[prefix + ' ' + data_dir] = {'images': np.load(slices_path), 'masks': np.load(masks_path)}

    return {'images': images, 'masks': masks}, datasets


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

    def __init__(self, data, source, transform=None):

        self.images = data['images']
        self.masks = data['masks']
        self.transform = transform
        self.source = source

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.asarray(self.images[idx], dtype=np.float32)
        mask = np.asarray(self.masks[idx], dtype=np.uint8)

        if self.transform:
            transformed = self.transform(image=image, target=mask)
            image = transformed['image']
            mask = transformed['target']

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if image.ndim == 2:
            image = image.unsqueeze(0)

        return image, mask, torch.tensor(self.source)
    

req_size = 512
train_transform = A.Compose([
    A.RandomSizedCrop(min_max_height=[req_size // 2, req_size * 2], size = [req_size, req_size], w2h_ratio=1, p = 1),
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
    A.RandomSizedCrop(min_max_height=[req_size, req_size], size = [req_size, req_size], w2h_ratio=1, p = 1),
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


def toLongPath(path):
    path = os.path.abspath(path)

    if os.name != "nt":
        return path

    buffer = ctypes.create_unicode_buffer(4096)

    result = ctypes.windll.kernel32.GetLongPathNameW(
        path,
        buffer,
        4096
    )

    if result == 0:
        return path

    return buffer.value


def Train(model, device, lr, base_data_path, user_data_path, base_prop, val_prop, batchsize, max_epochs, tb_logger):
    print("Writer id:", id(tb_logger))
    base_data, base_datasets = read_dataset_list(base_data_path)
    user_data, user_datasets = read_dataset_list(user_data_path)

    print(f'base {len(base_data['images'])} samples\nuser {len(user_data['images'])} samples', flush=True)

    base_data_train, base_data_val = train_val_split(base_data, val_prop)
    user_data_train, user_data_val = train_val_split(user_data, val_prop)

    base_train_dataset = SegmentationDataset(base_data_train, 0, train_transform)
    base_val_dataset = SegmentationDataset(base_data_val, 0, eval_transform)

    user_train_dataset = SegmentationDataset(user_data_train, 1, train_transform)
    user_val_dataset = SegmentationDataset(user_data_val, 1, eval_transform)

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

    early_stop_lr = 1e-8

    metric_names = [
        "iou",
        "prauc",
        "f1",
        "precision",
        "recall",
    ]

    for epoch in range(max_epochs):
        metrics_dict = {'base': {metric: 0.0 for metric in metric_names}, 
                        'user': {metric: 0.0 for metric in metric_names},
                        'train loss': 0.0,
                        'validation loss': 0.0
                        }
        model.train()
        for images, masks, sources in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model.forward(images)

            optimizer.zero_grad()
            loss = bce_loss(preds, masks.unsqueeze(1).float()) + dice_loss(preds, masks)
            loss.backward()
            optimizer.step()

            metrics_dict['train loss'] += loss.item()
        
        metrics_dict['train loss'] /= len(train_loader)

        model.eval()
        for images, masks, sources in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model.forward(images)
        
            metrics_dict['validation loss'] += (bce_loss(preds, masks.unsqueeze(1)) + dice_loss(preds, masks)).item()

            base_mask = (sources == 0)
            user_mask = (sources == 1)

            if base_mask.any():
                base_preds = preds[base_mask]
                base_masks = masks[base_mask]

                metrics_dict['base']['iou'] += metrics.compute_binary_iou(base_preds, base_masks)
                metrics_dict['base']['prauc'] += metrics.pr_auc_score(base_preds, base_masks)

                f1, precision, recall = metrics.metrix(base_preds, base_masks)
                metrics_dict['base']['f1'] += f1
                metrics_dict['base']['precision'] += precision
                metrics_dict['base']['recall'] += recall

            if user_mask.any():
                user_preds = preds[user_mask]
                user_masks = masks[user_mask]

                metrics_dict['user']['iou'] += metrics.compute_binary_iou(user_preds, user_masks)
                metrics_dict['user']['prauc'] += metrics.pr_auc_score(user_preds, user_masks)

                f1, precision, recall = metrics.metrix(user_preds, user_masks)
                metrics_dict['user']['f1'] += f1
                metrics_dict['user']['precision'] += precision
                metrics_dict['user']['recall'] += recall

        for group in ['base', 'user']:
            for metric in metric_names:
                metrics_dict[group][metric] /= len(val_loader)
        metrics_dict['validation loss'] /= len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]

        tb_logger.add_scalar("Loss/train", metrics_dict["train loss"], epoch)
        tb_logger.add_scalar("Loss/val", metrics_dict["validation loss"], epoch)
        tb_logger.add_scalar("LR", current_lr, epoch)
        tb_logger.flush()

        for metric in metric_names:
            tb_logger.add_scalar(f"Base/{metric}", metrics_dict["base"][metric], epoch)
            tb_logger.add_scalar(f"User/{metric}", metrics_dict["user"][metric], epoch)
        
        print(f"PROGRESS:{epoch+1}:{max_epochs}", flush=True)

        plateau_scheduler.step(metrics_dict['validation loss'])
        if current_lr <= early_stop_lr:
            print(f"Early stopping: lr reached {current_lr}", flush=True)
            break
    
    tb_logger.close()
        

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
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--tensorboard_path", required=True)
    args = parser.parse_args()

    print(f'received {args.model_path}', flush=True)
    print(f'received {args.output_model_path}', flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(
        args.model_path,
        map_location=device,
        weights_only=False,
    )
    model = model.to(device)

    tensorboard_logger = SummaryWriter(log_dir=os.path.join(args.tensorboard_path, f'{datetime.now().strftime('%d_%m %H_%M')}'))

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
        tb_logger=tensorboard_logger,
    )


    output_model_path = args.output_model_path
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(output_model_path, f"finetuned_model_{timestamp}.pth")
    torch.save(model, model_save_path)
    print(f'model saved at {model_save_path}')
    
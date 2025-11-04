import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

try:
    # reuse existing helper if available
    from danial.dataloader import load_image
except Exception:
    def load_image(path, target_size=(224, 224), normalize_imagenet=True):
        img = Image.open(path).convert('RGB')
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        arr = np.array(img).astype('float32') / 255.0
        if normalize_imagenet:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).float()


class BoxDataset(Dataset):
    """Dataset that reads a COCO-like annotation JSON and yields (image_tensor, annotation_dict).

    Items:
        image_tensor: torch.FloatTensor (C, H, W)
        annotation_dict: dict with keys 'annotations' (list of dicts with 'bbox' and 'category_id'),
                         and image meta 'width' and 'height'.
    """

    def __init__(self, annotation_json, images_root=None, target_size=(224, 224), normalize=True):
        with open(annotation_json, 'r') as f:
            self.coco = json.load(f)

        self.images = {}
        for im in self.coco.get('images', []):
            # image dict should contain at least 'id' and 'file_name' (or 'filename')
            img_id = im.get('id')
            fname = im.get('file_name') or im.get('filename') or im.get('file')
            self.images[img_id] = {
                'file_name': fname,
                'width': im.get('width'),
                'height': im.get('height')
            }

        # group annotations by image_id
        self.anns_per_image = {}
        for ann in self.coco.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id not in self.anns_per_image:
                self.anns_per_image[img_id] = []
            self.anns_per_image[img_id].append(ann)

        # create list of image ids that exist and have file names
        self.image_ids = [i for i, v in self.images.items() if v.get('file_name')]
        if len(self.image_ids) == 0:
            raise ValueError('No images found in annotation file or missing file_name keys')

        self.images_root = images_root or os.path.dirname(annotation_json)
        self.target_size = target_size
        self.normalize = normalize

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.images[img_id]
        fname = info['file_name']

        img_path = os.path.join(self.images_root, fname) if not os.path.isabs(fname) else fname
        img = load_image(img_path, target_size=self.target_size, normalize_imagenet=self.normalize)

        # Some load_image helpers return a batched tensor with shape (1, C, H, W).
        # Ensure we return a plain (C, H, W) tensor for stacking in collate_fn.
        try:
            # torch tensor
            if hasattr(img, 'dim') and img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)
        except Exception:
            pass

        # get annotations for this image
        anns = self.anns_per_image.get(img_id, [])
        # keep only bbox and category_id (COCO style bbox: [x,y,w,h] in pixels)
        anns_simple = []
        for a in anns:
            anns_simple.append({
                'bbox': a.get('bbox'),
                'category_id': a.get('category_id')
            })

        annotation_dict = {
            'annotations': anns_simple,
            'width': info.get('width'),
            'height': info.get('height')
        }

        return img, annotation_dict


def collate_fn(batch):
    """Collate function for DataLoader. Batch is list of (img_tensor, annotation_dict).
    Returns: images tensor (B, C, H, W), list of annotation_dicts (length B)
    """
    imgs = [b[0] for b in batch]
    anns = [b[1] for b in batch]
    imgs = torch.stack(imgs, dim=0)
    # If imgs ended up with an extra singleton dimension (B,1,C,H,W) because
    # individual elements were themselves batched, squeeze that dimension.
    if imgs.dim() == 5 and imgs.size(1) == 1:
        imgs = imgs.squeeze(1)
    return imgs, anns


if __name__ == '__main__':
    # quick smoke test
    import sys
    if len(sys.argv) < 2:
        print('Usage: python box_dataloader.py path/to/annotation.json')
        sys.exit(0)
    ds = BoxDataset(sys.argv[1])
    imgs, anns = ds[0]
    print('Image shape:', imgs.shape)
    print('Annotations:', anns)

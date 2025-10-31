import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataLoader(Dataset):
    """
    Simple dataset that loads images from a directory and converts them to numpy arrays
    in VGG-compatible format.
    """
    
    def __init__(self, image_dir, target_size=(512, 512), normalize_imagenet=True,
                 extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        """
        Args:
            image_dir: path to directory containing images
            target_size: tuple (height, width) to resize images to
            normalize_imagenet: apply ImageNet normalization (mean/std)
            extensions: tuple of valid image extensions to load
        """
        self.image_dir = image_dir
        self.target_size = target_size
        self.normalize_imagenet = normalize_imagenet
        
        # ImageNet normalization values
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Get all image files
        self.image_paths = []
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(extensions):
                self.image_paths.append(os.path.join(image_dir, fname))
        
        self.image_paths.sort()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: torch tensor of shape (C, H, W)
            path: path to the image file
        """
        img_path = self.image_paths[idx]
        
        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        
        # Convert to numpy array (H, W, C) in range [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization if requested
        if self.normalize_imagenet:
            img_array = (img_array - self.mean) / self.std
        
        # Convert to (C, H, W) format
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img_array).float()
        
        return img_tensor, img_path
    
    def get_dataloader(self, batch_size=32, shuffle=False, num_workers=0):
        """
        Create a DataLoader for batch processing.
        
        Returns:
            DataLoader that yields batches of shape (B, C, H, W)
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )


def load_image(image_path, target_size=(512, 512), normalize_imagenet=True):
    """
    Load a single image and convert it to VGG-compatible format.
    
    Args:
        image_path: path to image file
        target_size: tuple (height, width)
        normalize_imagenet: apply ImageNet normalization
    
    Returns:
        torch tensor of shape (1, C, H, W) ready for VGG
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    img_array = np.array(img).astype(np.float32) / 255.0
    
    if normalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
    
    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)
    
    return img_tensor

class SegmentationDataset(Dataset):
    """
    Dataset that returns (image_tensor, mask_tensor) pairs for segmentation.

    - Images and masks are paired by basename (image.jpg <-> image.png).
    - Masks are converted to single-channel float tensors in {0,1} and
      returned with shape (1, H, W) to be compatible with BCEWithLogitsLoss.
    """

    def __init__(self, image_dir, mask_dir, target_size=(224, 224), normalize_imagenet=True,
                 image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                 mask_extensions=('.png', '.bmp', '.tiff')):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.normalize_imagenet = normalize_imagenet

        self.image_extensions = image_extensions
        self.mask_extensions = mask_extensions

        # collect images and masks and pair by basename
        images = {}
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(image_extensions):
                images[os.path.splitext(fname)[0]] = os.path.join(image_dir, fname)

        masks = {}
        for fname in os.listdir(mask_dir):
            if fname.lower().endswith(mask_extensions):
                masks[os.path.splitext(fname)[0]] = os.path.join(mask_dir, fname)

        # keep only basenames present in both
        common = sorted([b for b in images.keys() if b in masks.keys()])

        if len(common) == 0:
            raise ValueError(f"No matching image/mask pairs found in {image_dir} and {mask_dir}")

        self.pairs = [(images[b], masks[b]) for b in common]
        print(f"Found {len(self.pairs)} image-mask pairs")

        # ImageNet stats
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0

        if self.normalize_imagenet:
            img_array = (img_array - self.mean) / self.std

        img_array = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.from_numpy(img_array).float()

        # load mask as grayscale
        mask = Image.open(mask_path).convert('L')
        # For masks, use nearest interpolation to avoid introducing intermediate values
        mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        mask_array = np.array(mask).astype(np.float32)

        # Binarize mask: any non-zero pixel becomes 1.0
        mask_array = (mask_array > 0).astype(np.float32)

        # Convert to (1, H, W)
        mask_array = np.expand_dims(mask_array, axis=0)
        mask_tensor = torch.from_numpy(mask_array).float()

        return img_tensor, mask_tensor

    def get_dataloader(self, batch_size=32, shuffle=False, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    

# Example usage
if __name__ == "__main__":
    image_dir = "/Users/dania/code/fyp/MHNet/test-images/images"
    mask_dir = "/Users/dania/code/fyp/MHNet/test-images/masks"
    dataset = SegmentationDataset(image_dir, mask_dir)
    loader = dataset.get_dataloader(batch_size=1, shuffle=True)

    img, mask = next(iter(loader))

    print("Image shape:", img.shape)       # Expect (1, 3, 224, 224)
    print("Mask shape:", mask.shape)       # Expect (1, 1, 224, 224)
    print("Mask unique values:", torch.unique(mask))
    print("Mask min:", mask.min().item(), "Mask max:", mask.max().item())



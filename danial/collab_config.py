# @title
train = '''

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import shutil

from danial import model, dataloader, loss_d, cod10k




def main():
	parser = argparse.ArgumentParser(description='Train MHNet')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
	parser.add_argument('--save-name', type=str, default='first', help='Subdirectory under models to save checkpoints (e.g. first, second)')
	args = parser.parse_args()

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	net = model.Model().to(device)

	# Create data loader for training data (COD10K-v2)
	print(cod10k.fuyo())
	train_image_dir = '/content/dasatet/COD10K-v2/Train/Images/camo_images'
	train_mask_dir = '/content/dasatet/COD10K-v2/Train/GT_Objects/GT_Object'
	train_dataset = dataloader.SegmentationDataset(
		image_dir=train_image_dir,
		mask_dir=train_mask_dir,
		target_size=(224, 224),
		normalize_imagenet=True
	)
	train_loader = train_dataset.get_dataloader(
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=4
	)

	criterion = loss_fn = loss_d.FeatureMapLoss(
        mse_weight=1.0,
        cosine_weight=0.5,
        pearson_weight=0.3,
        gram_weight=0.1,
        reduction="mean",
    ).to(device)
	optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)


	# Prepare models directory (use save-name arg so user can choose first/second/etc.)
	model_dir = os.path.join('models', args.save_name)
	os.makedirs(model_dir, exist_ok=True)
	print(f"Saving checkpoints to: {model_dir}")
	best_loss = float('inf')

	for epoch in range(args.epochs):
		net.train()
		running_loss = 0.0
		for i, (inputs, targets) in enumerate(train_loader):
			inputs = inputs.to(device)
			# targets shape: (B, 1, H, W) and dtype float
			targets = targets.to(device)
			optimizer.zero_grad()
			binary_pred, boundary_pred, fused_feat = net(inputs)
			
			# targets: [B,1,224,224]

			# loss for map only
			loss = criterion(fused_feat, targets)
			
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			if (i+1) % 10 == 0:
				print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
		avg_loss = running_loss / len(train_loader)
		print(f'Epoch [{epoch+1}/{args.epochs}] finished. Avg Loss: {avg_loss:.4f}')

		# Append epoch loss to log.txt
		log_path = os.path.join(model_dir, 'log.txt')
		try:
			with open(log_path, 'a') as lf:
				lf.write(f"""
			{epoch+1},{avg_loss}
			 """)
		except Exception as e:
			print('Warning: failed to write log:', e)

		# Save checkpoint for this epoch
		ckpt = {
			'epoch': epoch + 1,
			'model_state_dict': net.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': avg_loss,
		}
		epoch_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
		torch.save(ckpt, epoch_path)
		# update last
		last_path = os.path.join(model_dir, 'last.pth')
		torch.save(ckpt, last_path)
		# update best if improved
		if avg_loss < best_loss:
			best_loss = avg_loss
			best_path = os.path.join(model_dir, 'best.pth')
			torch.save(ckpt, best_path)

	# Training finished - plot loss curve and save image
	def plot_loss(model_dir):
		log_file = os.path.join(model_dir, 'log.txt')
		if not os.path.exists(log_file):
			print('No log file found at', log_file)
			return
		# Read CSV lines of epoch,loss
		epochs = []
		losses = []
		with open(log_file, 'r') as rf:
			for line in rf:
				line = line.strip()
				if not line:
					continue
				parts = line.split(',')
				try:
					e = int(parts[0])
					l = float(parts[1])
				except Exception:
					continue
				epochs.append(e)
				losses.append(l)
		if len(epochs) == 0:
			print('No valid entries found in log file')
			return
		# sort by epoch just in case
		pairs = sorted(zip(epochs, losses), key=lambda x: x[0])
		epochs, losses = zip(*pairs)
		plt.figure()
		plt.plot(list(epochs), list(losses), marker='o')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('Training Loss')
		plt.grid(True)
		outpath = os.path.join(model_dir, 'loss.png')
		plt.savefig(outpath)
		plt.close()
		print('Saved loss plot to', outpath)

	# Call plot function
	try:
		plot_loss(model_dir)
	except Exception as e:
		print('Warning: plotting failed:', e)

if __name__ == '__main__':
	main()



'''
code10k = '''
import os
import shutil
# Paths

# os.chdir("..")

source_folder = "/content/dasatet/COD10K-v2/Train/Images/Image"  # change to your source folder
target_folder = "/content/dasatet/COD10K-v2/Train/Images/camo_images"  # new folder to store COD10K-CAM images

def fuyo():
    if os.path.exists(target_folder):
        return"folder already exists"
    else:
        # Create target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Initialize counter
        count = 0

        # Iterate through files
        for filename in os.listdir(source_folder):
            if filename.startswith("COD10K-CAM"):
                source_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)
                shutil.copy2(source_path, target_path)  # copy file with metadata
                count += 1

        return f"Total COD10K-CAM images copied: {count}"
'''
dl = '''
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
    mod = SegmentationDataset(image_dir, mask_dir)
    
    sample = mod[0]
    print("Image shape:", sample[0].shape)
    print("Mask shape:", sample[1].shape)


'''

def mini_zombie():
    return train, code10k, dl
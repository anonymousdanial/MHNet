
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from danial import model
from danial import dataloader

def main():
	parser = argparse.ArgumentParser(description='Train MHNet')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
	parser.add_argument('--smoke', action='store_true', help='Run one-batch smoke test and exit')
	args = parser.parse_args()

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	net = model.Model().to(device)

	# Create data loader for training data (COD10K-v2)
	train_image_dir = 'datasets/COD10K-v2/Train/Images/Image'
	train_mask_dir = 'datasets/COD10K-v2/Train/GT_Objects/GT_Object'
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

	criterion = nn.BCEWithLogitsLoss()  # Binary classification for segmentation
	optimizer = optim.Adam(net.parameters(), lr=args.lr)

	# Lightweight segmentation head that maps the fused feature map to a 1-channel
	# spatial prediction and upsamples to the input image size. This is a stop-gap
	# so the training script can compute a segmentation loss against available masks.
	seg_head = nn.Sequential(
		nn.Conv2d(64, 1, kernel_size=1),
		nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
	).to(device)

	# Add seg_head params to optimizer so it's trained together with the model
	optimizer.add_param_group({'params': seg_head.parameters()})

	for epoch in range(args.epochs):
		net.train()
		running_loss = 0.0
		for i, (inputs, targets) in enumerate(train_loader):
			inputs = inputs.to(device)
			# targets shape: (B, 1, H, W) and dtype float
			targets = targets.to(device)
			optimizer.zero_grad()
			binary_pred, boundary_pred, fused_feat = net(inputs)
			
			# If running a smoke test just print shapes and exit (model here returns
			# classification/regression heads plus recovered features, not spatial masks)
			if args.smoke:
				print('binary_pred shape:', getattr(binary_pred, 'shape', None))
				print('boundary_pred shape:', getattr(boundary_pred, 'shape', None))
				print('fused_feat shape:', getattr(fused_feat, 'shape', None))
				# Compute seg prediction and loss for the smoke test
				seg_pred = seg_head(fused_feat)
				print('seg_pred shape:', seg_pred.shape)
				try:
					seg_loss = criterion(seg_pred, targets.to(device))
				except Exception as e:
					seg_loss = e
				print('seg_loss:', seg_loss)
				return
			
			# Compute a segmentation prediction from the recovered fused feature map
			# fused_feat shape is typically [B, 64, 7, 7]; map -> [B,1,224,224]
			seg_pred = seg_head(fused_feat)
			# targets: [B,1,224,224]
			seg_loss = criterion(seg_pred, targets)
			
			# For now the training loss will be the segmentation loss
			loss = seg_loss
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			if (i+1) % 10 == 0:
				print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
			# If smoke test requested, run only one batch and exit
			if args.smoke:
				print('Smoke test: processed one batch, exiting')
				return
		print(f'Epoch [{epoch+1}/{args.epochs}] finished. Avg Loss: {running_loss/len(train_loader):.4f}')

if __name__ == '__main__':
	main()



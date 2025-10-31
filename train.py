
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from danial import model, dataloader, loss_d

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
	train_image_dir = 'dasatet/COD10k-v2/Train/Images/Image'
	train_mask_dir = 'dasatet/COD10k-v2/Train/GT_Objects/GT_Object'
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
				lf.write(f"{epoch+1},{avg_loss}\n")
		except Exception as e:
			print('Warning: failed to write log:', e)

		# Save checkpoint for this epoch
		ckpt = {
			'epoch': epoch + 1,
			'model_state_dict': net.state_dict(),
			'seg_head_state_dict': seg_head.state_dict(),
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



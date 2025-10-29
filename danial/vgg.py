import torch
import torch.nn as nn
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    VGG Feature Extractor that returns feature maps from each of the 5 main blocks.
    Ignores the fully connected head layers and only uses the convolutional features.
    """
    
    def __init__(self, model_name='vgg16', pretrained=True, requires_grad=False):
        """
        Args:
            model_name: 'vgg16' or 'vgg19'
            pretrained: whether to load pretrained weights
            requires_grad: whether to allow gradient computation
        """
        super(VGGFeatureExtractor, self).__init__()
        
        # Load VGG model
        if model_name == 'vgg16':
            vgg = models.vgg16(pretrained=pretrained)
        elif model_name == 'vgg19':
            vgg = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError("model_name must be 'vgg16' or 'vgg19'")
        
        # Extract only the features (convolutional layers), ignore classifier
        features = vgg.features
        
        # Define the 5 feature blocks based on max pooling layers
        # VGG16: [0-4], [5-9], [10-16], [17-23], [24-30]
        # VGG19: [0-4], [5-9], [10-18], [19-27], [28-36]
        self.block1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.block3 = nn.Sequential()
        self.block4 = nn.Sequential()
        self.block5 = nn.Sequential()
        
        # Split layers into blocks (before each maxpool)
        if model_name == 'vgg16':
            blocks = [4, 9, 16, 23, 30]
        else:  # vgg19
            blocks = [4, 9, 18, 27, 36]
        
        block_idx = 0
        current_block = self.block1
        
        for idx, layer in enumerate(features):
            current_block.add_module(str(idx), layer)
            
            if idx == blocks[block_idx]:
                block_idx += 1
                if block_idx == 1:
                    current_block = self.block2
                elif block_idx == 2:
                    current_block = self.block3
                elif block_idx == 3:
                    current_block = self.block4
                elif block_idx == 4:
                    current_block = self.block5
        
        # Freeze parameters if specified
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x, return_all=True):
        """
        Forward pass through VGG blocks.
        
        Args:
            x: input tensor (B, C, H, W)
            return_all: if True, returns dict of all feature maps
                       if False, returns only the final feature map
        
        Returns:
            if return_all=True: dict with keys 'block1' through 'block5'
            if return_all=False: final feature map tensor
        """
        if return_all:
            features = {}
            
            x = self.block1(x)
            features['block1'] = x
            
            x = self.block2(x)
            features['block2'] = x
            
            x = self.block3(x)
            features['block3'] = x
            
            x = self.block4(x)
            features['block4'] = x
            
            x = self.block5(x)
            features['block5'] = x
            
            return features
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            return x
    
    def get_block1(self, x):
        """Extract features from block 1 only"""
        return self.block1(x)
    
    def get_block2(self, x):
        """Extract features from blocks 1-2"""
        x = self.block1(x)
        return self.block2(x)
    
    def get_block3(self, x):
        """Extract features from blocks 1-3"""
        x = self.block1(x)
        x = self.block2(x)
        return self.block3(x)
    
    def get_block4(self, x):
        """Extract features from blocks 1-4"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)
    
    def get_block5(self, x):
        """Extract features from blocks 1-5"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.block5(x)

def backbone():
    return VGGFeatureExtractor(model_name='vgg16', pretrained=True)



if __name__ == "__main__":
    # Create the feature extractor
    vgg_features = backbone()
    vgg_features.eval()
    
    # Create dummy input (batch_size=2, channels=3, height=224, width=224)
    x = torch.randn(2, 3, 224, 224)
    
    # Method 1: Get all feature maps at once
    # print("Get all features")
    features = vgg_features(x, return_all=True)
    # for name, feat in features.items():
    #     print(f"{name}: {feat.shape}")
    f1 = features['block1']
    f2 = features['block2']
    f3 = features['block3']
    print(f1.shape)
    print(f2.shape)
    print(f3.shape)
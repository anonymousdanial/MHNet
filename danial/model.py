from . import f3 as conv_f3
from . import pre_fcs
from . import fcs
from . import vgg
from . import prm as perm
from . import spg

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, config=None):
        super(Model, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        # VGG Backbone
        self.backbone = vgg.backbone()
        
        # SPG (Selective Weighted Attention)
        self.spg = spg.SelectiveWeightedAttention(in_channels1=64, in_channels2=128)
        
        # F3 Convolution
        self.conv_f3 = conv_f3.F3Conv(in_channels=256, out_channels=128)
        
        # PRM
        self.prm = perm.PRM(f3_channels=128, spg_channels=32, fused_channels=128)
        
        # PreFCS
        self.pre_fcs = pre_fcs.PreFCS()
        
        # CRFCS
        self.crfcs = fcs.CRFCS(roi_channels=64, boundary_channels=64)

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x, return_all=True)
        f1 = features['block1']
        f2 = features['block2']
        f3 = features['block3']
        
        # SPG processing
        spg_out = self.spg(f1, f2)
        
        # F3 processing
        f3_conv = self.conv_f3(f3)
        
        # PRM processing
        prm_out = self.prm(f3_conv, spg_out)
        
        # PreFCS processing
        roi_out = self.pre_fcs(f3_conv, prm_out)
        branch = self.pre_fcs.branch
        
        # CRFCS processing
        binary_pred, boundary_pred, fused_feat = self.crfcs(roi_out, branch)
        
        return binary_pred, boundary_pred, fused_feat

    def summary(self):
        print("Model Components:")
        print("- VGG Backbone")
        print("- Selective Weighted Attention (SPG)")
        print("- F3 Convolution")
        print("- PRM (Perception Refinement Module)")
        print("- PreFCS")
        print("- CRFCS (Conditional Random Field based Camouflage Segmentation)")


if __name__ == "__main__":
    import dataloader
    test = dataloader.load_image("/Users/dania/code/fyp/MHNet/assets/RR.png")
    mod = Model()
    mod.eval()
    with torch.no_grad():
        binary_pred, boundary_pred, fused_feat = mod(test)
    print(mod)
    print(binary_pred.shape, boundary_pred.shape, fused_feat.shape)
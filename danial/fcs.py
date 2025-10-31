import torch
import torch.nn as nn
import torch.nn.functional as F


class CRFCS(nn.Module):
    """
    Concept Recovery / Feature Clue Supplement Module
    
    Based on the paper: "MHNet: Military high-level camouflage object detection"
    This module recovers missing features by using reverse attention combined
    with boundary feature constraints.
    
    Args:
        roi_channels (int): Number of channels in the ROI feature map (default: 512)
        boundary_channels (int): Number of channels in the boundary feature map (default: 512)
        output_size (int): Size of the output feature map (default: 224)
    """
    
    def __init__(self, roi_channels=64, boundary_channels=512, output_size=224):
        super(CRFCS, self).__init__()
        
        self.output_size = output_size
        
        # 1x1 convolution to generate masked feature map
        # Use a slightly deeper conv to get more spatial variation
        self.mask_conv = nn.Sequential(
            nn.Conv2d(roi_channels, roi_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(roi_channels // 2, 1, kernel_size=1),
        )
        
        # 1x1 convolution for boundary feature constraint
        self.boundary_conv = nn.Conv2d(boundary_channels, 1, kernel_size=1, bias=True)
        
        # 3x3 convolution for feature recovery (Eq. 6 in paper)
        self.recovery_conv = nn.Conv2d(roi_channels, roi_channels, 
                                       kernel_size=3, padding=1, bias=True)
        
        # 1x1 convolution for selective weighted attention
        self.attention_conv = nn.Conv2d(roi_channels * 2, roi_channels, 
                                        kernel_size=1, bias=True)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
        # CI head (classification) - now uses pooled features
        self.cls_head = nn.Linear(roi_channels * 7 * 7, 5) # 5 refers to number of classes

        # Reg head (bounding box) - now uses pooled features
        self.reg_head = nn.Linear(roi_channels * 7 * 7, 4)

        
    def forward(self, roi_features, boundary_features):
        """
        Forward pass of CR/FCS module
        
        Args:
            roi_features: ROI feature map from deep layer with RPN [B, C, H_roi, W_roi]
                         e.g., [1, 64, 7, 7]
            boundary_features: Boundary features from F5^1 layer [B, C, H_bound, W_bound]
                              e.g., [1, 64, 56, 56]
            
        Returns:
            cls_out: Classification predictions [B, num_classes]
            reg_out: Bounding box regression [B, 4]
            recovered_features: Feature map with recovered missing cues [B, C, 224, 224]
        """
        B, C, H_roi, W_roi = roi_features.shape
        _, _, H_bound, W_bound = boundary_features.shape
        
        # Step 1: Generate masked feature map M_t (Eq. 5)
        # This represents the predicted/visible regions
        M_t = self.mask_conv(roi_features)  # [B, 1, 7, 7]
        
        # Step 2: Compute reverse attention map a_t^C (Eq. 5)
        # Reverse attention highlights non-predicted regions (missing features)
        reverse_attention = 1 - self.sigmoid(M_t)  # [B, 1, 7, 7]
        
        # Step 3: Generate information constraint f_t^C from boundary features
        # This provides boundary constraints for the reverse attention search
        boundary_constraint = self.boundary_conv(boundary_features)  # [B, 1, 56, 56]
        boundary_constraint = self.sigmoid(boundary_constraint)
        
        # Resize boundary constraint to match ROI size
        boundary_constraint = F.interpolate(
            boundary_constraint, 
            size=(H_roi, W_roi), 
            mode='bilinear', 
            align_corners=False
        )  # [B, 1, 7, 7]
        
        # Step 4: Apply reverse attention with boundary constraints (Eq. 6)
        # Element-wise multiplication to focus on missing features within boundaries
        constrained_attention = reverse_attention * boundary_constraint
        
        # Apply to ROI features
        attended_features = constrained_attention * roi_features
        
        # 3x3 convolution to recover missing body features
        x_t = self.recovery_conv(attended_features)
        x_t = self.relu(x_t)
        
        # Step 5: Selective weighted attention for mutual information compensation
        # Concatenate original ROI features with recovered features
        combined = torch.cat([roi_features, x_t], dim=1)
        
        # Generate attention weights
        attention_weights = self.attention_conv(combined)
        attention_weights = self.sigmoid(attention_weights)
        
        # Apply weighted combination (still at 7x7)
        recovered_features_7x7 = attention_weights * x_t + (1 - attention_weights) * roi_features
        
        # For classification and regression heads, use the 7x7 features
        x_flat = recovered_features_7x7.view(B, -1)
        cls_out = self.cls_head(x_flat)
        reg_out = self.reg_head(x_flat)
        
        # Upsample recovered features to 224x224
        recovered_features = F.interpolate(
            recovered_features_7x7,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False
        )  # [B, C, 224, 224]
        
        return cls_out, reg_out, recovered_features
    
    def get_reverse_attention_map(self, roi_features):
        """
        Utility function to visualize the reverse attention map
        
        Args:
            roi_features: ROI feature map [B, C, H, W]
            
        Returns:
            reverse_attention: Reverse attention map [B, 1, H, W]
        """
        M_t = self.mask_conv(roi_features)
        reverse_attention = 1 - self.sigmoid(M_t)
        return reverse_attention


# Example usage
if __name__ == "__main__":
    # Create module
    crfcs = CRFCS(roi_channels=64, boundary_channels=64, output_size=224)
    
    # Example inputs
    batch_size = 1
    channels = 64
    roi_height, roi_width = 7, 7  # ROI features are typically 7x7
    boundary_height, boundary_width = 56, 56  # Boundary features can be larger
    
    roi_features = torch.randn(batch_size, channels, roi_height, roi_width)
    boundary_features = torch.randn(batch_size, channels, boundary_height, boundary_width)
    
    # Forward pass
    cls_out, reg_out, recovered = crfcs(roi_features, boundary_features)
    
    print(f"Input ROI features shape: {roi_features.shape}")
    print(f"Boundary features shape: {boundary_features.shape}")
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Regression output shape: {reg_out.shape}")
    print(f"Recovered features shape: {recovered.shape}")
    
    # Visualize reverse attention
    reverse_attn = crfcs.get_reverse_attention_map(roi_features)
    print(f"Reverse attention map shape: {reverse_attn.shape}")
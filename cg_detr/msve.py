import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, swin_t
import math
from typing import Dict, List, Tuple, Optional

class PositionalEncoding3D(nn.Module):
    """3D positional encoding for video sequences"""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]

class SlowFastPathway(nn.Module):
    """Simplified SlowFast pathway implementation"""
    def __init__(self, input_dim: int, hidden_dim: int, alpha: int = 8, beta: float = 0.125):
        super().__init__()
        self.alpha = alpha  # Temporal sampling ratio
        self.beta = beta    # Channel ratio
        
        # Slow pathway (high spatial, low temporal resolution)
        self.slow_pathway = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fast pathway (low spatial, high temporal resolution)
        fast_channels = int(hidden_dim * beta)
        self.fast_pathway = nn.Sequential(
            nn.Conv3d(input_dim, fast_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(fast_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(fast_channels, fast_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(fast_channels),
            nn.ReLU(inplace=True)
        )
        
        # Lateral connections
        self.lateral = nn.Conv3d(fast_channels, hidden_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            fused features: (B, C, T, H, W)
        """
        # Slow pathway - subsample temporally
        slow_input = x[:, :, ::self.alpha]  # Subsample temporal dimension
        slow_features = self.slow_pathway(slow_input)
        
        # Fast pathway - full temporal resolution
        fast_features = self.fast_pathway(x)
        
        # Lateral connection - upsample fast features temporally to match slow
        fast_upsampled = F.interpolate(
            fast_features, 
            size=slow_features.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        lateral_features = self.lateral(fast_upsampled)
        
        # Fuse pathways
        fused = slow_features + lateral_features
        
        # Upsample back to original temporal resolution if needed
        if fused.shape[2] != x.shape[2]:
            fused = F.interpolate(
                fused, 
                size=(x.shape[2], fused.shape[3], fused.shape[4]),
                mode='trilinear',
                align_corners=False
            )
        
        return fused

class I3DBlock(nn.Module):
    """Simplified I3D block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//2, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(out_channels//2, out_channels//2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(out_channels//2, out_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
        )
        
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv3d(x)
        out += self.shortcut(x)
        return self.relu(out)

class TemporalTransformerBlock(nn.Module):
    """Temporal attention block similar to TimeSformer"""
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - temporal sequence
        """
        # Temporal self-attention
        attn_out, _ = self.temporal_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x

class MultiStreamVideoEncoder(nn.Module):
    """
    Multi-stream video encoder combining different temporal modeling approaches
    """
    def __init__(
        self, 
        input_channels: int = 3,
        feature_dim: int = 512,
        max_frames: int = 100,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_positional_encoding = use_positional_encoding
        
        # 1. ResNet-based spatial feature extractor
        self.spatial_encoder = self._build_resnet_encoder()
        
        # 2. SlowFast pathway for motion dynamics
        self.slowfast_encoder = SlowFastPathway(
            input_dim=input_channels, 
            hidden_dim=feature_dim//2
        )
        
        # 3. I3D pathway for spatio-temporal features
        self.i3d_encoder = nn.Sequential(
            I3DBlock(input_channels, 64),
            I3DBlock(64, 128),
            I3DBlock(128, feature_dim//2),
            nn.AdaptiveAvgPool3d((None, 1, 1))  # Spatial pooling, keep temporal
        )
        
        # 4. Temporal Transformer (TimeSformer-like)
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding3D(feature_dim, max_frames)
            
        self.temporal_transformer = nn.Sequential(
            TemporalTransformerBlock(feature_dim, nhead=8),
            TemporalTransformerBlock(feature_dim, nhead=8),
        )
        
        # Feature fusion and projection layers
        self.fusion_layers = nn.ModuleDict({
            'spatial_proj': nn.Linear(2048, feature_dim//4),  # ResNet features
            'slowfast_proj': nn.Linear(feature_dim//2, feature_dim//4),
            'i3d_proj': nn.Linear(feature_dim//2, feature_dim//4),
            'temporal_proj': nn.Linear(feature_dim, feature_dim//4),
        })
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Motion energy computation
        self.motion_energy_conv = nn.Conv1d(feature_dim, 1, kernel_size=3, padding=1)
        
    def _build_resnet_encoder(self):
        """Build ResNet-50 based spatial encoder"""
        resnet = resnet50(pretrained=True)
        # Remove final classification layers
        return nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
    
    def extract_spatial_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features using ResNet
        Args:
            video: (B, T, C, H, W)
        Returns:
            spatial_features: (B, T, D)
        """
        B, T, C, H, W = video.shape
        
        # Process each frame independently
        video_flat = video.view(B * T, C, H, W)
        spatial_feats = self.spatial_encoder(video_flat)  # (B*T, 2048, H', W')
        
        # Global average pooling
        spatial_feats = F.adaptive_avg_pool2d(spatial_feats, (1, 1))  # (B*T, 2048, 1, 1)
        spatial_feats = spatial_feats.view(B, T, 2048)  # (B, T, 2048)
        
        return spatial_feats
    
    def compute_motion_energy(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute motion energy from temporal features
        Args:
            features: (B, T, D)
        Returns:
            motion_energy: (B, T)
        """
        # Transpose for conv1d: (B, D, T)
        features_t = features.transpose(1, 2)
        motion_energy = self.motion_energy_conv(features_t)  # (B, 1, T)
        return motion_energy.squeeze(1)  # (B, T)
    
    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-stream encoder
        
        Args:
            video: (B, T, C, H, W) - batch of videos
            
        Returns:
            Dictionary containing:
                - 'features': (B, T, D) - fused multi-stream features
                - 'motion_energy': (B, T) - temporal motion energy
                - 'stream_features': dict with individual stream features
        """
        B, T, C, H, W = video.shape
        
        # 1. Spatial stream (ResNet)
        spatial_features = self.extract_spatial_features(video)  # (B, T, 2048)
        spatial_proj = self.fusion_layers['spatial_proj'](spatial_features)  # (B, T, D//4)
        
        # 2. SlowFast stream
        video_3d = video.transpose(1, 2)  # (B, C, T, H, W) for 3D conv
        slowfast_features = self.slowfast_encoder(video_3d)  # (B, D//2, T, H', W')
        slowfast_pooled = F.adaptive_avg_pool3d(slowfast_features, (None, 1, 1))  # (B, D//2, T, 1, 1)
        slowfast_features = slowfast_pooled.squeeze(-1).squeeze(-1).transpose(1, 2)  # (B, T, D//2)
        slowfast_proj = self.fusion_layers['slowfast_proj'](slowfast_features)  # (B, T, D//4)
        
        # 3. I3D stream
        i3d_features = self.i3d_encoder(video_3d)  # (B, D//2, T, 1, 1)
        i3d_features = i3d_features.squeeze(-1).squeeze(-1).transpose(1, 2)  # (B, T, D//2)
        i3d_proj = self.fusion_layers['i3d_proj'](i3d_features)  # (B, T, D//4)
        
        # 4. Concatenate for temporal transformer input
        concat_features = torch.cat([
            spatial_proj, slowfast_proj, i3d_proj, 
            torch.zeros(B, T, self.feature_dim//4, device=video.device)  # Placeholder for temporal
        ], dim=-1)  # (B, T, D)
        
        # Apply positional encoding
        if self.use_positional_encoding:
            concat_features = self.pos_encoding(concat_features)
        
        # Temporal transformer
        temporal_features = self.temporal_transformer(concat_features)  # (B, T, D)
        temporal_proj = self.fusion_layers['temporal_proj'](temporal_features)  # (B, T, D//4)
        
        # 5. Final feature fusion
        final_concat = torch.cat([
            spatial_proj, slowfast_proj, i3d_proj, temporal_proj
        ], dim=-1)  # (B, T, D)
        
        fused_features = self.final_fusion(final_concat)  # (B, T, D)
        
        # 6. Compute motion energy
        motion_energy = self.compute_motion_energy(fused_features)  # (B, T)
        
        # Prepare outputs
        stream_features = {
            'spatial': spatial_features,
            'slowfast': slowfast_features, 
            'i3d': i3d_features,
            'temporal': temporal_features
        }
        
        return {
            'features': fused_features,
            'motion_energy': motion_energy,
            'stream_features': stream_features
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the multi-stream encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy video input
    batch_size = 2
    num_frames = 32
    height, width = 224, 224
    video_input = torch.randn(batch_size, num_frames, 3, height, width).to(device)
    
    # Initialize encoder
    encoder = MultiStreamVideoEncoder(
        input_channels=3,
        feature_dim=512,
        max_frames=100
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = encoder(video_input)
    
    print("Multi-Stream Video Encoder Output Shapes:")
    print(f"Fused features: {outputs['features'].shape}")
    print(f"Motion energy: {outputs['motion_energy'].shape}")
    print("Individual stream features:")
    for stream_name, features in outputs['stream_features'].items():
        print(f"  {stream_name}: {features.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
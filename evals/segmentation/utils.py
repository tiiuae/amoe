# Segmentation utilities for Falcon Vision evaluation
# Uses the standalone falcon_vision model

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from amoe import AMOE
from amoe.utils import FEATURE_DIM_DICT, PATCH_SIZE, load_amoe_model


def build_backbone_and_processor(
    ckpt_path: str,
    configs: str,
    device: torch.device = torch.device("cuda"),
    feature_type: str = "dinov3",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Build the backbone model and image processor.
    
    Args:
        ckpt_path: Path to model checkpoint
        device: Device to load model on
        feature_type: Type of features to extract
        image_size: Image size for processing
    
    Returns:
        Tuple of (backbone, image_processor)
    """
    # Load model
    model, image_processor = load_amoe_model(
        checkpoint_path=ckpt_path,
        config_name=configs,
        device=device,
        dtype=dtype,
        do_resize=False
    )
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Wrap in backbone interface
    backbone = AMOEBackbone(model, feature_type=feature_type)
    
    return backbone, image_processor


class AMOEBackbone(nn.Module):
    """Wrapper that provides a unified interface for feature extraction."""
    
    def __init__(self, model: AMOE, feature_type: str = "dinov3"):
        super().__init__()
        self.model = model
        self.feature_type = feature_type
        self.patch_size = model.patch_size
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
        compile: bool = True,
    ) -> dict:
        """
        Extract features from images.
        
        Args:
            pixel_values: Preprocessed image patches (N, L, C*P*P)
            spatial_shape: Patch grid shape per image (N, 2)
            compile: Whether to use compiled attention
        
        Returns:
            Dictionary with output features
        """
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                spatial_shapes=spatial_shapes,
                compile=compile,
            )
        return outputs


class AMOELinearSeg(nn.Module):
    """Linear segmentation head on top of Falcon Vision backbone."""
    
    def __init__(
        self,
        backbone: AMOEBackbone,
        num_classes: int = 21,
        feature_dim: int = 1024,
        feature_type: str = "dinov3",
        image_size: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.feature_type = feature_type
        self.image_size = image_size
        self.patch_size = backbone.patch_size
        
        # Linear segmentation head
        self.head = nn.LazyConv2d(num_classes, kernel_size=1)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shape: torch.Tensor,
        upsample: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for segmentation.

        Args:
            upsample: If True, bilinear-upsample logits to (image_size, image_size).
                      If False, return logits at patch resolution (fast, for training).
        """
        # Extract features (use no_grad, NOT inference_mode — inference_mode
        # tensors break autograd for the head's backward pass)
        with torch.no_grad():
            outputs = self.backbone(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shape,
            compile=True,
            )

        # Get patch features for the desired type — detach to cut graph
        feats = outputs["patch_features"][self.feature_type].detach()  # (N, L, D)

        N, L, D = feats.shape
        H_patch = int(spatial_shape[0, 0].item())
        W_patch = int(spatial_shape[0, 1].item())

        # Reshape to spatial grid
        feats = feats.view(N, H_patch, W_patch, D)
        feats = feats.permute(0, 3, 1, 2).contiguous()  # (N, D, H_patch, W_patch)

        logits = self.head(feats)  # (N, C, H_patch, W_patch)
        if upsample:
            logits = F.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def forward_from_precomputed(self, feats: torch.Tensor, upsample: bool = False) -> torch.Tensor:
        """Forward using precomputed features."""
        logits = self.head(feats)
        if upsample:
            logits = F.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits


class PrecomputedFeatureDataset(Dataset):
    """Dataset wrapping precomputed spatial feature maps and target masks."""

    def __init__(self, features: list[torch.Tensor], targets: list[torch.Tensor]):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


@torch.no_grad()
def precompute_features(
    backbone: AMOEBackbone,
    dataloader,
    feature_type: str,
    device: torch.device,
):
    """Run backbone once over the entire dataloader, return (features_list, targets_list).

    Each feature tensor is (D, H_patch, W_patch) already reshaped to spatial grid.
    Stored on CPU to avoid GPU OOM.
    """
    backbone.eval()
    all_features = []
    all_targets = []
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        spatial_shape = batch["spatial_shape"].to(device, non_blocking=True)
        targets = batch["targets"]  # keep on CPU

        with torch.inference_mode():
            outputs = backbone(
                pixel_values=pixel_values,
                spatial_shapes=spatial_shape,
                compile=False,
            )

        feats = outputs["patch_features"][feature_type]  # (N, L, D)
        N, L, D = feats.shape
        H_patch = int(spatial_shape[0, 0].item())
        W_patch = int(spatial_shape[0, 1].item())
        feats = feats.view(N, H_patch, W_patch, D).permute(0, 3, 1, 2).contiguous()  # (N, D, H, W)

        for i in range(N):
            all_features.append(feats[i].cpu())
            all_targets.append(targets[i])

    return all_features, all_targets


def make_collate_fn(
    image_processor,
    max_num_patches: int = 256,
    output_dtype: torch.dtype = torch.bfloat16,
):
    """Create a collate function for the dataloader."""
    
    def collate_fn(batch):
        images, masks = zip(*batch)
        
        # Process images (patchify + pad + mask)
        enc = image_processor(
            list(images),
            max_num_patches=max_num_patches,
            pad=True,
            output_dtype=output_dtype,
            mask_dtype=output_dtype,
        )
        
        # Stack masks
        masks = torch.stack(masks)
        
        return {
            "pixel_values": enc["pixel_values"],
            "spatial_shape": enc["spatial_shape"],
            "targets": masks,
        }
    
    return collate_fn


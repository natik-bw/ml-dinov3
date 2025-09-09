#!/usr/bin/env python3
"""
Video-based segmentation tracking using DINOv3 features.

This script implements temporal segmentation tracking by:
1. Loading frames from the tracking cache (similar to run_tracking_processor.py)
2. Setting up the first frame as reference segmentation
3. Tracking segmentation masks through subsequent frames using DINOv3 features
4. Creating visualizations of the tracking results

Based on the segmentation tracking approach from segmentation_tracking.ipynb
"""

import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt

from create_tracking_frame_data import load_tracking_cache
from dino_bw.utils.dino_embeddings_utils import get_class_idx_from_name
from bw_ml_common.datasets.data_accessor_factory import create_dataset_accessor

from dino_bw.utils.tracking_processing import propogate_context_masked_probs, make_neighborhood_mask, VideoFrameData, \
    TreesTracker, normalize_probs, extract_frames_from_cache


# ============================================================================
# TREE INSTANCE TRACKING DATA STRUCTURE
# ============================================================================


def track_first_to_second_frame(
        first_frame: VideoFrameData,
        second_frame: VideoFrameData,
        reference_probs: Tensor,  # [h, w, M]
        num_classes: int,
        neighborhood_size: float = 12.0,
        neighborhood_shape: str = "circle",
        topk: int = 5,
        temperature: float = 0.2
) -> Tuple[Tensor, np.ndarray, np.ndarray]:
    """
    Track tree instances from first frame to second frame.
    
    Args:
        first_frame: Reference frame with tree instances
        second_frame: Target frame to track to
        reference_probs: Reference probabilities [h, w, M]
        num_classes: Number of classes/instances
        neighborhood_size: Size of spatial neighborhood
        neighborhood_shape: Shape of neighborhood ("circle" or "square") 
        topk: Number of top similar patches to consider
        temperature: Temperature for softmax weighting
        
    Returns:
        Tuple of (current_probs, current_pred_patches, current_probs_np)
        - current_probs: Predicted probabilities [h, w, M] (patch resolution)
        - current_pred_patches: Predicted mask as numpy array [h, w] (patch resolution)
        - current_probs_np: Probabilities as numpy array [M, H', W'] (target resolution)
    """
    print("=== TRACKING FIRST TO SECOND FRAME ===")

    # Get features (already normalized)
    first_feats = first_frame.features  # [h, w, D]
    second_feats = second_frame.features  # [h, w, D]

    print(f"First frame features: {first_feats.shape}")
    print(f"Second frame features: {second_feats.shape}")
    print(f"Reference probs: {reference_probs.shape}")

    # Create neighborhood mask
    h, w = first_feats.shape[:2]
    neighborhood_mask = make_neighborhood_mask(h, w, neighborhood_size, neighborhood_shape)
    print(f"Neighborhood mask: {neighborhood_mask.shape}")

    # Ensure features have proper shape and mark as dynamic for torch.compile
    torch._dynamo.maybe_mark_dynamic(first_feats, (0, 1))
    torch._dynamo.maybe_mark_dynamic(second_feats, (0, 1))

    # Run propagation
    current_probs = propogate_context_masked_probs(
        second_feats,  # [h", w", D] - current frame
        context_features=first_feats.unsqueeze(0),  # [1, h, w, D] - context
        context_probs=reference_probs.unsqueeze(0),  # [1, h, w, M] - context probs
        neighborhood_mask=neighborhood_mask,  # [h", w", h, w]
        topk=topk,
        temperature=temperature,
    )  # [h", w", M]

    print(f"Propagated probs: {current_probs.shape}")

    # Upsample to original image resolution for visualization
    # Convert from [h, w, M] to [1, M, h, w] for interpolation
    p = current_probs.movedim(-1, -3).unsqueeze(0)  # [1, M, h, w]

    # Get target size from second frame
    target_h, target_w = second_frame.target_image_size[1], second_frame.target_image_size[0]  # (height, width)
    p = F.interpolate(p, size=(target_h, target_w), mode="nearest")  # [1, M, H', W']

    # Postprocess probabilities
    p = normalize_probs(p).squeeze(0)  # [M, H', W']

    # Get predictions at both resolutions
    # Patch resolution predictions for consistent mask handling
    current_pred_patches = current_probs.argmax(-1).cpu().numpy()  # [h, w]

    # Target resolution predictions and probabilities for detailed visualization
    current_pred_np = p.argmax(0).cpu().numpy()  # [H', W']
    current_probs_np = p.cpu().numpy()  # [M, H', W']

    print(f"Patch resolution prediction shape: {current_pred_patches.shape}")
    print(f"Target resolution prediction shape: {current_pred_np.shape}")
    print(f"Final probabilities shape: {current_probs_np.shape}")

    return current_probs, current_pred_patches, current_probs_np


def process_consecutive_frames(
        video_frames: List[VideoFrameData],
        tree_class_id: int,
        patch_size: int = 16,
        max_context_length: int = 4,
        neighborhood_size: float = 12.0,
        neighborhood_shape: str = "circle",
        topk: int = 5,
        temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Process consecutive frames with memory-based tree instance tracking.
    
    Args:
        video_frames: List of video frames to process
        tree_class_id: Class ID for trees 
        patch_size: DINO patch size
        max_context_length: Maximum number of context frames to keep
        neighborhood_size: Spatial neighborhood size for propagation
        neighborhood_shape: Shape of neighborhood ("circle" or "square")
        topk: Number of top similar patches
        temperature: Temperature for softmax
        
    Returns:
        Dictionary containing tracking results for all frames
    """
    print("=== PROCESSING CONSECUTIVE FRAMES ===")
    print(f"Total frames: {len(video_frames)}")

    # Initialize tracker
    tracker = TreesTracker(tree_class_id=tree_class_id)

    # Storage for results
    results: Dict[str, Any] = {
        'frames': [],
        'tracker': tracker,
        'metadata': {
            'max_context_length': max_context_length,
            'neighborhood_size': neighborhood_size,
            'topk': topk,
            'temperature': temperature
        }
    }

    # Context queues (similar to segmentation_tracking.ipynb)
    features_queue: List[Tensor] = []
    probs_queue: List[Tensor] = []

    # Process first frame (reference)
    first_frame = video_frames[0]
    if not first_frame.has_segmentation or not first_frame.has_detections:
        raise ValueError("First frame must have both segmentation and detections")

    print(f"\n--- PROCESSING FRAME 0 (REFERENCE) ---")

    # Create initial instances using TreesTracker
    first_seg_mask = torch.from_numpy(first_frame.segmentation).long().cuda()
    first_instance_mask, first_class_mapping = tracker.create_initial_instances(
        segmentation_mask=first_seg_mask,
        detections=first_frame.detections_scaled,
        patch_size=patch_size,
        target_image_size=first_frame.target_image_size
    )

    # Convert to one-hot probabilities (mask is already remapped)
    num_classes = len(first_class_mapping)
    first_probs = F.one_hot(first_instance_mask, num_classes).float()

    # Store first frame results
    frame_result = {
        'frame_idx': 0,
        'instance_mask': first_instance_mask.cpu().numpy(),
        'class_mapping': first_class_mapping,
        'probs': first_probs.cpu().numpy(),
        'num_classes': num_classes
    }
    results['frames'].append(frame_result)

    # Set up neighborhood mask
    h, w = first_frame.features.shape[:2]
    neighborhood_mask = make_neighborhood_mask(h, w, neighborhood_size, neighborhood_shape)

    # Process remaining frames
    for frame_idx in range(1, len(video_frames)):
        current_frame = video_frames[frame_idx]
        print(f"\n--- PROCESSING FRAME {frame_idx} ---")

        # Prepare context
        if frame_idx == 1:
            # First tracking frame: use only reference frame as context
            context_features = first_frame.features.unsqueeze(0)  # [1, h, w, D]
            context_probs = first_probs.unsqueeze(0)  # [1, h, w, M]
        else:
            # Use reference + recent frames as context
            context_features = torch.stack([first_frame.features] + features_queue, dim=0)
            context_probs = torch.stack([first_probs] + probs_queue, dim=0)

        print(f"Context shape: {context_features.shape}")

        # Mark features as dynamic for torch.compile
        torch._dynamo.maybe_mark_dynamic(current_frame.features, (0, 1))
        torch._dynamo.maybe_mark_dynamic(context_features, 0)
        torch._dynamo.maybe_mark_dynamic(context_probs, (0, 3))

        # Run propagation to get predicted probabilities
        predicted_probs = propogate_context_masked_probs(
            current_frame.features,  # [h, w, D]
            context_features,  # [t, h, w, D]
            context_probs,  # [t, h, w, M]
            neighborhood_mask,  # [h, w, h, w]
            topk=topk,
            temperature=temperature
        )  # [h, w, M]

        print(f"Predicted probs shape: {predicted_probs.shape}")

        # Convert semantic segmentation to instance segmentation using tracker
        if current_frame.has_segmentation and current_frame.has_detections:
            current_seg_mask = torch.from_numpy(current_frame.segmentation).long().cuda()

            current_instance_mask, current_class_mapping = tracker.assign_instances_from_predictions(
                semantic_mask=current_seg_mask,
                predicted_probs=predicted_probs,
                detections=current_frame.detections_scaled,
                patch_size=patch_size,
                target_image_size=current_frame.target_image_size
            )

            # Update class mapping and probabilities for new instances
            current_num_classes = len(current_class_mapping)

            # Convert to one-hot probabilities (mask is already remapped)
            current_probs = F.one_hot(current_instance_mask, current_num_classes).float()
        else:
            # No segmentation/detections: use predicted probabilities directly
            print("Warning: No segmentation/detections found, using predictions directly")
            current_instance_mask = predicted_probs.argmax(dim=-1)
            current_class_mapping = {i: i for i in range(predicted_probs.shape[-1])}
            current_num_classes = predicted_probs.shape[-1]
            current_probs = predicted_probs

        # Store frame results
        frame_result = {
            'frame_idx': frame_idx,
            'instance_mask': current_instance_mask.cpu().numpy(),
            'class_mapping': current_class_mapping,
            'probs': current_probs.cpu().numpy(),
            'num_classes': current_num_classes,
            'predicted_probs': predicted_probs.cpu().numpy()
        }
        results['frames'].append(frame_result)

        # Update context queues
        features_queue.append(current_frame.features)
        probs_queue.append(current_probs)

        # Maintain queue size
        if len(features_queue) > max_context_length:
            features_queue.pop(0)
        if len(probs_queue) > max_context_length:
            probs_queue.pop(0)

    print(f"\n✓ Processed {len(video_frames)} frames successfully")
    print(f"Final active instances: {list(tracker.active_instances.keys())}")

    return results


def visualize_first_to_second_tracking_results(
        first_frame: VideoFrameData,
        second_frame: VideoFrameData,
        reference_mask: np.ndarray,
        predicted_mask: np.ndarray,
        predicted_probs: np.ndarray,
        num_classes: int,
        data_folder: Path,
        save_path: Optional[Path] = None,
        ignore_class: Optional[int] = None,
        overlay_alpha: float = 0.6,
        patch_size: int = 16
):
    """
    Visualize tracking results similar to segmentation_tracking.ipynb.
    
    Shows:
    1. First frame with reference segmentation
    2. Second frame with predicted segmentation
    3. Overlay visualization with alpha blending for non-ignore classes
    4. Individual probability maps for each tree instance
    
    Args:
        ignore_class: Class to ignore in overlay (if None, uses max class index)
        overlay_alpha: Alpha value for overlay blending
    """
    print("=== VISUALIZING TRACKING RESULTS ===")

    # Load original images
    first_image = load_image_for_visualization(first_frame, data_folder)
    second_image = load_image_for_visualization(second_frame, data_folder)

    # Create RGB masks
    reference_rgb = mask_to_rgb(reference_mask, num_classes)
    predicted_rgb = mask_to_rgb(predicted_mask, num_classes)

    # Determine ignore class (background/invalid class)
    if ignore_class is None:
        ignore_class = num_classes - 1  # Use maximum class index as ignore class

    # Create overlay visualizations
    def create_overlay(image, mask, alpha=overlay_alpha, patch_size=16):
        """Create overlay where non-ignore classes are highlighted with alpha blending."""
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        # Upsample mask from patch resolution to pixel resolution
        # mask is [H_patches, W_patches], need to make it [H_pixels, W_pixels]
        mask_upsampled = np.repeat(np.repeat(mask, patch_size, axis=0), patch_size, axis=1)

        # Ensure mask matches image dimensions exactly
        img_h, img_w = image_np.shape[:2]
        mask_h, mask_w = mask_upsampled.shape

        if mask_h != img_h or mask_w != img_w:
            # Crop or pad mask to match image size exactly
            min_h, min_w = min(mask_h, img_h), min(mask_w, img_w)
            mask_final = np.full((img_h, img_w), ignore_class, dtype=mask_upsampled.dtype)
            mask_final[:min_h, :min_w] = mask_upsampled[:min_h, :min_w]
        else:
            mask_final = mask_upsampled

        # Create mask for non-ignore areas
        non_ignore_mask = mask_final != ignore_class

        # Create colored overlay from upsampled mask
        overlay = mask_to_rgb(mask_final, num_classes)

        # Apply alpha blending only where mask is not ignore class
        result = image_np.copy().astype(float)
        overlay_float = overlay.astype(float)

        # Blend only non-ignore areas
        for c in range(3):
            result[non_ignore_mask, c] = (
                    (1 - alpha) * image_np[non_ignore_mask, c] +
                    alpha * overlay_float[non_ignore_mask, c]
            )

        return result.astype(np.uint8)

    reference_overlay = create_overlay(first_image, reference_mask, patch_size=patch_size)
    predicted_overlay = create_overlay(second_image, predicted_mask, patch_size=patch_size)

    # Main comparison plot - now 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: Original images + overlays
    axes[0, 0].imshow(first_image)
    axes[0, 0].set_title(f"Reference Frame {first_frame.frame_idx}")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(reference_overlay)
    axes[0, 1].set_title(f"Reference Overlay (α={overlay_alpha})")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(reference_rgb)
    axes[0, 2].set_title("Reference Tree Instances")
    axes[0, 2].axis('off')

    # Bottom row: Target frame + results
    axes[1, 0].imshow(second_image)
    axes[1, 0].set_title(f"Target Frame {second_frame.frame_idx}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(predicted_overlay)
    axes[1, 1].set_title(f"Predicted Overlay (α={overlay_alpha})")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(predicted_rgb)
    axes[1, 2].set_title("Predicted Tree Instances")
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / "tracking_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved tracking comparison to {save_path / 'tracking_comparison.png'}")

    plt.show()

    # Individual probability maps
    if predicted_probs.shape[0] > 1:  # Only if we have multiple classes
        n_classes = min(predicted_probs.shape[0], 8)  # Limit to 8 for display
        cols = min(4, n_classes)
        rows = (n_classes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_classes):
            row, col = i // cols, i % cols
            axes[row, col].imshow(predicted_probs[i], cmap='hot', vmin=0, vmax=1)
            axes[row, col].set_title(f"Tree Instance {i}")
            axes[row, col].axis('off')

        # Hide unused subplots
        for i in range(n_classes, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path / "tree_instance_probabilities.png", dpi=150, bbox_inches='tight')
            print(f"Saved probability maps to {save_path / 'tree_instance_probabilities.png'}")

        plt.show()


def create_reference_segmentation(
        first_frame: VideoFrameData,
        tree_class_id: int,
        patch_size: int = 16
) -> Tuple[torch.Tensor, int, Dict, torch.Tensor] | None:
    """
    DEPRECATED: Use TreesTracker.create_initial_instances() instead.
    
    This function uses the old approach with complex ID remapping.
    The new TreesTracker class uses a simpler approach:
    - background = 0
    - tree instances start from 1
    - no complex remapping needed
    
    Args:
        first_frame: First video frame with segmentation data and detections
        tree_class_id: Class ID for trees (should be 2)
        patch_size: Size of patches for coordinate conversion
        
    Returns:
        Tuple of (one_hot_probs, actual_num_classes, class_mapping, instance_mask)
    """
    print("=== CREATING TREE INSTANCE SEGMENTATION ===")

    if not first_frame.has_segmentation:
        raise ValueError("First frame must have segmentation data")

    if not first_frame.has_detections:
        print("Warning: No detections found, returning None")
        return None

    # Get segmentation mask [H_patches, W_patches]
    seg_mask = torch.from_numpy(first_frame.segmentation).long().cuda()
    h_patches, w_patches = seg_mask.shape

    # Get scaled detections (in DINO coordinates)
    detections = first_frame.detections_scaled

    # Debug: print detection structure
    if len(detections) > 0:
        print(f"Detection sample keys: {list(detections[0].keys())}")
        print(f"First detection: {detections[0]}")

    # Filter for tree detections - try multiple possible key names
    tree_detections = []
    for det in detections:
        if det.get('name') == 'trunk':
            tree_detections.append(det)

    print(f"Total detections: {len(detections)}")
    print(f"Found {len(tree_detections)} tree detections")
    print(f"Segmentation mask shape: {seg_mask.shape}")
    print(f"Target image size: {first_frame.target_image_size}")

    # Create new instance mask
    instance_mask = seg_mask.clone()

    # Convert tree patches (class 2) to individual instances based on bounding boxes
    tree_patches = (seg_mask == tree_class_id)
    print(f"Found {tree_patches.sum().item()} tree patches")

    if len(tree_detections) > 0 and tree_patches.sum() > 0:
        # Start instance numbering from a high number to avoid conflicts
        instance_id = 100

        # Convert each detection box to patch coordinates
        for i, detection in enumerate(tree_detections):
            box = detection['box']
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

            # Convert from image coordinates to patch coordinates
            target_w, target_h = first_frame.target_image_size
            patch_x1 = int(x1 / patch_size)
            patch_y1 = int(y1 / patch_size)
            patch_x2 = int(x2 / patch_size)
            patch_y2 = int(y2 / patch_size)

            # Clamp to patch grid bounds
            patch_x1 = max(0, min(patch_x1, w_patches - 1))
            patch_y1 = max(0, min(patch_y1, h_patches - 1))
            patch_x2 = max(0, min(patch_x2, w_patches - 1))
            patch_y2 = max(0, min(patch_y2, h_patches - 1))

            # Find tree patches within this bounding box
            box_tree_patches = tree_patches[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

            if box_tree_patches.sum() > 0:
                # Assign unique instance ID to tree patches in this box
                instance_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1][box_tree_patches] = instance_id
                print(f"  Tree instance {instance_id}: bbox=({patch_x1},{patch_y1})->({patch_x2},{patch_y2}), "
                      f"{box_tree_patches.sum().item()} patches out of total {box_tree_patches.nelement()}")
                instance_id += 1

        # Mark remaining tree patches (outside all bounding boxes) as background
        remaining_tree_patches = (instance_mask == tree_class_id)
        if remaining_tree_patches.sum() > 0:
            print(f"Marking {remaining_tree_patches.sum().item()} tree patches outside detections as background")
            instance_mask[remaining_tree_patches] = 255  # Background

    # Get final unique classes
    unique_classes = instance_mask.unique()
    actual_num_classes = len(unique_classes)

    print(f"Final unique classes: {unique_classes.cpu().numpy()}")
    print(f"Total classes: {actual_num_classes}")

    # Create contiguous class mapping (0, 1, 2, ...)
    class_mapping = {}
    for i, class_id in enumerate(unique_classes):
        class_mapping[class_id.item()] = i

    # Remap to contiguous class IDs
    remapped_mask = torch.zeros_like(instance_mask)
    for orig_id, new_id in class_mapping.items():
        remapped_mask[instance_mask == orig_id] = new_id

    # Convert to one-hot probabilities [H, W, M]
    one_hot_probs = F.one_hot(remapped_mask, actual_num_classes).float()

    print(f"Instance segmentation shape: {one_hot_probs.shape}")

    # Print class distribution
    class_counts = [(class_mapping[k.item()], (remapped_mask == class_mapping[k.item()]).sum().item())
                    for k in unique_classes]
    print(f"Class distribution: {class_counts}")

    return one_hot_probs, actual_num_classes, class_mapping, remapped_mask


def load_image_for_visualization(frame: VideoFrameData, data_folder: Path) -> Image.Image:
    """Load the image and resize to DINO target size for visualization purposes."""
    image_path = data_folder / "images" / frame.filename
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Resize to DINO target size to match mask resolution
    target_size = frame.target_image_size  # (width, height)
    image_resized = image.resize(target_size, Image.Resampling.BICUBIC)

    return image_resized


def mask_to_rgb(mask: np.ndarray, num_masks: int) -> np.ndarray:
    """
    Convert segmentation mask to RGB visualization.
    Adapted from segmentation_tracking.ipynb
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Exclude background
    background = mask == 0
    mask = mask - 1
    num_masks = num_masks - 1

    # Choose palette
    if num_masks <= 10:
        mask_rgb = plt.get_cmap("tab10")(mask)[..., :3]
    elif num_masks <= 20:
        mask_rgb = plt.get_cmap("tab20")(mask)[..., :3]
    else:
        mask_rgb = plt.get_cmap("gist_rainbow")(mask / (num_masks - 1))[..., :3]

    mask_rgb = (mask_rgb * 255).astype(np.uint8)
    mask_rgb[background, :] = 0
    return mask_rgb


def main(
        data_folder: Path,
        cache_path: Path,
        tree_class_id: int,
        frame_indices: np.ndarray,
        debugging_folder: Optional[Path] = None,
        patch_size: int = 16,
        image_size: int = 1024
):
    """
    Main video tracking processor.
    
    Args:
        data_folder: Path to dataset folder
        cache_path: Path to cached DINO features and segmentation  
        tree_class_id: Class ID for trees
        debugging_folder: Optional folder for saving debug visualizations
        frame_indices: List of frame indices to process (default: auto-select)
        patch_size: DINO patch size (should match cache)
        image_size: Target image size (should match cache)
    """
    if not (cache_result := load_tracking_cache(cache_path, patch_size, image_size))[0]:
        warnings.warn(
            "Cache validation failed - parameters mismatch or missing metadata. Please run tracking_data_processor.py with correct parameters first",
            RuntimeWarning)
        return
    cache_data = cache_result[1]

    video_frames = extract_frames_from_cache(
        cache_data,
        frame_indices=frame_indices,
        require_segmentation=True,
        require_detections=True
    )

    if len(video_frames) == 0:
        warnings.warn("No valid frames found for processing", RuntimeWarning)
        return

    print(f"Successfully extracted {len(video_frames)} frames")

    # STAGE 2: Create reference segmentation from first frame
    print("\n=== STAGE 2: CREATING REFERENCE SEGMENTATION ===")
    first_frame = video_frames[0]

    try:
        # Use TreesTracker for consistent ID management
        tracker = TreesTracker(tree_class_id=tree_class_id)
        first_seg_mask = torch.from_numpy(first_frame.segmentation).long().cuda()
        reference_instance_mask, class_mapping = tracker.create_initial_instances(
            segmentation_mask=first_seg_mask,
            detections=first_frame.detections_scaled,
            patch_size=patch_size,
            target_image_size=first_frame.target_image_size
        )

        # Convert to one-hot probabilities
        num_classes = len(class_mapping)
        reference_probs = F.one_hot(reference_instance_mask, num_classes).float()

        print("✓ Reference segmentation created successfully")
    except Exception as e:
        print(f"Error creating reference segmentation: {e}")
        return

    # STAGE 3: Prepare for tracking (foundation for next steps)
    print("\n=== STAGE 3: PREPARING TRACKING FOUNDATION ===")

    # Normalize features (similar to segmentation_tracking.ipynb)
    print("Normalizing features...")
    for frame in video_frames:
        # Features are already loaded as tensors, just normalize
        frame.features = F.normalize(frame.features, dim=-1, p=2)

    print(f"✓ Normalized features for {len(video_frames)} frames")

    # STAGE 4: Choose tracking mode
    if len(video_frames) >= 2 and reference_probs is not None:

        # Option A: Simple two-frame demo
        if len(video_frames) == 2:
            print("\n=== STAGE 4: DEMONSTRATING TREE TRACKING (2 FRAMES) ===")

            first_frame = video_frames[0]
            second_frame = video_frames[1]

            # Track from first to second frame
            try:
                current_probs, predicted_mask_patches, predicted_probs = track_first_to_second_frame(
                    first_frame=first_frame,
                    second_frame=second_frame,
                    reference_probs=reference_probs,
                    num_classes=num_classes,
                    neighborhood_size=12.0,
                    neighborhood_shape="circle",
                    topk=5,
                    temperature=0.2
                )

                print("✓ Tree tracking completed successfully")

                # Visualize results
                if debugging_folder:
                    # Use the modified instance mask instead of original segmentation
                    reference_mask = reference_instance_mask.cpu().numpy()
                    visualize_first_to_second_tracking_results(
                        first_frame=first_frame,
                        second_frame=second_frame,
                        reference_mask=reference_mask,
                        predicted_mask=predicted_mask_patches,
                        predicted_probs=predicted_probs,
                        num_classes=num_classes,
                        data_folder=data_folder,
                        save_path=debugging_folder,
                        ignore_class=None,  # Will use max class index
                        overlay_alpha=0.6,
                        patch_size=patch_size
                    )
                    print("✓ Tracking visualization saved")

            except Exception as e:
                print(f"Error in tracking demo: {e}")
                import traceback
                traceback.print_exc()

        # Option B: Multi-frame consecutive processing
        else:
            print("\n=== STAGE 4: CONSECUTIVE FRAMES TREE TRACKING ===")

            try:
                # Process all frames with memory-based tracking
                tracking_results = process_consecutive_frames(
                    video_frames=video_frames,
                    tree_class_id=tree_class_id,
                    patch_size=patch_size,
                    max_context_length=3,
                    neighborhood_size=12.0,
                    neighborhood_shape="circle",
                    topk=5,
                    temperature=0.2
                )

                print("✓ Consecutive frames tracking completed successfully")

                # Save results if debugging folder is provided
                if debugging_folder:
                    import pickle
                    results_path = debugging_folder / "consecutive_tracking_results.pkl"
                    with open(results_path, 'wb') as f:
                        pickle.dump(tracking_results, f)
                    print(f"✓ Tracking results saved to {results_path}")

                    # Create summary visualization for first few frames
                    print("Creating summary visualizations...")
                    for i in range(min(3, len(tracking_results['frames']) - 1)):
                        frame_result = tracking_results['frames'][i + 1]  # Skip reference frame

                        try:
                            reference_mask = tracking_results['frames'][0]['instance_mask']
                            predicted_mask = frame_result['instance_mask']

                            save_path_frame = debugging_folder / f"frame_{i + 1}_tracking.png"

                            # Simple visualization (you can extend this)
                            import matplotlib.pyplot as plt
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                            ax1.imshow(mask_to_rgb(reference_mask, tracking_results['frames'][0]['num_classes']))
                            ax1.set_title("Reference (Frame 0)")
                            ax1.axis('off')

                            ax2.imshow(mask_to_rgb(predicted_mask, frame_result['num_classes']))
                            ax2.set_title(f"Tracked (Frame {i + 1})")
                            ax2.axis('off')

                            plt.tight_layout()
                            plt.savefig(save_path_frame, dpi=150, bbox_inches='tight')
                            plt.close()

                        except Exception as viz_e:
                            print(f"Warning: Could not create visualization for frame {i + 1}: {viz_e}")

            except Exception as e:
                print(f"Error in consecutive frames tracking: {e}")
                import traceback
                traceback.print_exc()

    # Create debugging folder if specified
    if debugging_folder:
        debugging_folder.mkdir(exist_ok=True, parents=True)
        print(f"✓ Debugging folder ready: {debugging_folder}")

        # Save reference frame visualization
        try:
            ref_image = load_image_for_visualization(first_frame, data_folder)
            # Use the modified instance mask
            ref_mask_rgb = mask_to_rgb(
                reference_instance_mask.cpu().numpy(),
                num_classes
            )

            # Create side-by-side visualization
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.imshow(ref_image)
            ax1.set_title(f"Reference Frame {first_frame.frame_idx}")
            ax1.axis('off')

            ax2.imshow(ref_mask_rgb)
            ax2.set_title("Reference Segmentation")
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(debugging_folder / "reference_frame.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Saved reference frame visualization")

        except Exception as e:
            print(f"Warning: Could not save reference frame visualization: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("VIDEO TRACKING COMPLETE")
    print("=" * 80)
    print(f"Total frames loaded: {len(video_frames)}")
    print(f"Reference frame: {first_frame.frame_idx} ({first_frame.filename})")
    print(f"Feature dimensions: {first_frame.features.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {class_mapping}")

    if len(video_frames) >= 2:
        print(f"Tracking demo: Frame {video_frames[0].frame_idx} -> Frame {video_frames[1].frame_idx}")

    if debugging_folder:
        print(f"Debug outputs saved to: {debugging_folder}")

    print("\nTree instance tracking successfully implemented!")
    print("Next: Extend to full video sequence processing")

    return {
        'video_frames': video_frames,
        'reference_probs': reference_probs,
        'reference_instance_mask': reference_instance_mask,
        'num_classes': num_classes,
        'class_mapping': class_mapping,
        'cache_data': cache_data
    }


if __name__ == "__main__":
    # Example usage - update these paths for your setup
    data_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0")
    cache_path = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/tracking_cache_img_1024_patch_16.pkl")
    debugging_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/video_tracking_debug")

    datasets_folder = "/home/nati/source/datasets"
    data_accessor = create_dataset_accessor("bw2508", datasets_folder)
    tree_class_id = get_class_idx_from_name(data_accessor, "tree")

    # Run with custom frame indices (first 20 frames for testing)
    frame_indices = np.arange(200, 220).astype('int')

    result = main(
        data_folder=data_folder,
        cache_path=cache_path,
        tree_class_id=tree_class_id,
        frame_indices=frame_indices,
        debugging_folder=debugging_folder,
        patch_size=16,
        image_size=1024
    )

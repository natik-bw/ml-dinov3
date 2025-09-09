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
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from create_tracking_frame_data import load_tracking_cache
from dino_bw.utils.dino_embeddings_utils import get_class_idx_from_name
from bw_ml_common.datasets.data_accessor_factory import create_dataset_accessor

from dino_bw.utils.tracking_processing import propogate_context_masked_probs, make_neighborhood_mask, VideoFrameData, \
    TreesTracker, normalize_probs, extract_frames_from_cache, \
    TrunkVideoTracker  # Note: TreesTracker used in reference function


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
        trunk_class_id: int,
        frame_indices: np.ndarray,
        debugging_folder: Optional[Path] = None,
        patch_size: int = 16,
        image_size: int = 1024
):
    """
    Simplified main video tracking processor using VideoTracker.
    
    Args:
        data_folder: Path to dataset folder
        cache_path: Path to cached DINO features and segmentation  
        trunk_class_id: Class ID for trunks
        debugging_folder: Optional folder for saving debug visualizations
        frame_indices: Array of frame indices to process
        patch_size: DINO patch size (should match cache)
        image_size: Target image size (should match cache)
    """
    print("=== SIMPLIFIED VIDEO TRACKING WITH VIDEOTRACKER ===")

    # Check cache validity and load data
    if not (cache_result := load_tracking_cache(cache_path, patch_size, image_size))[0]:
        warnings.warn(
            "Cache validation failed - parameters mismatch or missing metadata. Please run tracking_data_processor.py with correct parameters first",
            RuntimeWarning)
        return
    cache_data = cache_result[1]

    # Extract video frames
    video_frames = extract_frames_from_cache(
        cache_data,
        frame_indices=frame_indices,
        require_segmentation=True,
        require_detections=True
    )

    if len(video_frames) == 0:
        warnings.warn("No valid frames found for processing", RuntimeWarning)
        return
    print(f"Processing {len(video_frames)} frames with VideoTracker")
    target_image_size_wh = cache_data['frames'][0]['target_image_size']

    # Initialize VideoTracker
    tracker = TrunkVideoTracker(tree_class_id=trunk_class_id, context_window_size=5)
    # Process each frame
    results = []
    for i, video_frame in enumerate(tqdm(video_frames, desc="Processing frames", unit="frame")):
        # Process frame through VideoTracker
        trunk_instance_mask = tracker.process_frame(
            video_frame,
            patch_size,
            target_image_size_wh
        )

        # Store results
        frame_result = {
            'frame_idx': video_frame.frame_idx,
            'filename': video_frame.filename,
            'trunk_instance_mask': trunk_instance_mask,
            'num_trunk_instances': len(trunk_instance_mask.unique()) - 1,  # Exclude background
            'active_trunk_indexes': sorted(tracker.trunk_indexes)
        }
        results.append(frame_result)

        print(f"  Result: {frame_result['num_trunk_instances']} trunk instances")
        print(f"  Active trunk indexes: {frame_result['active_trunk_indexes']}")

    # Optionally save debug visualizations
    if debugging_folder:
        debugging_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving debug visualizations to {debugging_folder}")

        for i, result in enumerate(results):
            save_path = debugging_folder / f"frame_{i:03d}_trunks.png"

            # Simple visualization
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            trunk_mask = result['trunk_instance_mask'].cpu().numpy()
            ax.imshow(mask_to_rgb(trunk_mask, result['num_trunk_instances'] + 1))
            ax.set_title(f"Frame {result['frame_idx']}: {result['num_trunk_instances']} trunk instances")
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

    print(f"\n=== TRACKING COMPLETE ===")
    print(f"Processed {len(results)} frames")
    print(f"Total trunk indexes assigned: {len(tracker.trunk_indexes)}")
    print(f"Final context window size: {len(tracker.context_frames)}")

    return {
        'results': results,
        'tracker': tracker,
        'video_frames': video_frames,
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
        trunk_class_id=tree_class_id,
        frame_indices=frame_indices,
        debugging_folder=debugging_folder,
        patch_size=16,
        image_size=1024
    )

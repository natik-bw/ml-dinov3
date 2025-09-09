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
import cv2
import colorsys
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
    Convert segmentation mask to RGB visualization using consistent color mapping.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Create RGB image
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # Apply colors for each unique index using our consistent color function
    unique_indices = np.unique(mask)
    for idx in unique_indices:
        color = index_to_color(idx)
        mask_rgb[mask == idx] = color

    return mask_rgb


def index_to_color(index: int) -> Tuple[int, int, int]:
    """
    Convert an index to a distinct RGB color with high contrast between adjacent indices.
    
    Uses a combination of golden ratio spacing and HSV color space to ensure
    that consecutive indices have maximally different colors.
    
    Args:
        index: Integer index (0 for background/black, 1+ for distinct colors)
        
    Returns:
        RGB color tuple (R, G, B) with values in [0, 255]
    """
    if index == 0:
        return (0, 0, 0)  # Black for background

    # Use golden ratio conjugate to create maximum color separation
    # This avoids patterns and ensures adjacent indices have very different colors
    golden_ratio_conjugate = 0.618033988749
    hue = (index * golden_ratio_conjugate) % 1.0

    # High saturation and value for vivid, distinct colors
    saturation = 0.8 + (index % 3) * 0.1  # Vary saturation slightly (0.8-1.0)
    value = 0.8 + (index % 2) * 0.2  # Vary brightness slightly (0.8-1.0)

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    # Convert to 0-255 range
    return (int(r * 255), int(g * 255), int(b * 255))


def create_enhanced_frame_visualization(
        video_frame: 'VideoFrameData',
        result: Dict[str, Any],
        context_based_mask: Optional[torch.Tensor] = None,
        patch_size: int = 16,
        data_folder: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create enhanced per-frame visualization with three components.
    
    Args:
        video_frame: Original video frame data
        result: Frame result containing trunk_instance_mask and metadata
        context_based_mask: Raw context-based inference result (optional)
        patch_size: Size of patches for coordinate conversion
        
    Returns:
        Tuple of (bbox_overlay, final_segmentation, context_overlay) as RGB numpy arrays
    """
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap

    # Get dimensions
    trunk_mask = result['trunk_instance_mask'].cpu().numpy()
    h_patches, w_patches = trunk_mask.shape

    # Create base image (patch resolution)
    base_img_height = h_patches * patch_size
    base_img_width = w_patches * patch_size

    # Load actual image once and reuse for all overlays
    base_image = None
    if data_folder is not None:
        image_path = data_folder / "images" / video_frame.filename
        if image_path.exists():
            actual_image = cv2.imread(str(image_path))
            if actual_image is not None:
                actual_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB)
                base_image = cv2.resize(actual_image, (base_img_width, base_img_height))

    # Create base image fallback if loading failed
    if base_image is None:
        base_image = np.ones((base_img_height, base_img_width, 3), dtype=np.uint8) * 128  # Gray fallback

    # 1. Bounding Box Overlay - start with actual image
    bbox_overlay = base_image.copy()

    # Draw bounding boxes with colors and labels using tracking info
    bbox_tracking_info = result.get('bbox_tracking_info', [])
    for bbox_info in bbox_tracking_info:
        box = bbox_info['bbox']
        bbox_idx = bbox_info['bbox_index']
        tracking_idx = bbox_info['tracking_index']

        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

        # Generate color based on tracking index using color function
        color = index_to_color(tracking_idx)

        # Draw bounding box
        cv2.rectangle(bbox_overlay, (x1, y1), (x2, y2), color, 2)

        # Add text label below the bounding box
        label = f"{tracking_idx}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        # Position label below the box (y2 + offset)
        label_y = y2 + text_size[1] + 5
        cv2.rectangle(bbox_overlay, (x1, y2 + 5), (x1 + text_size[0], label_y), color, -1)
        cv2.putText(bbox_overlay, label, (x1, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 2. Final Segmentation Overlay - blend with actual image using alpha
    final_seg_colored = mask_to_rgb(trunk_mask, result['num_trunk_instances'] + 1)
    final_seg_colored_resized = cv2.resize(final_seg_colored, (base_img_width, base_img_height),
                                           interpolation=cv2.INTER_NEAREST)

    # Create alpha blended segmentation: show pure image for background, semi-transparent overlay for segments
    trunk_mask_resized = cv2.resize(trunk_mask.astype(np.uint8), (base_img_width, base_img_height),
                                    interpolation=cv2.INTER_NEAREST)
    background_pixels = trunk_mask_resized == 0

    # Alpha blending: 70% color overlay, 30% original image for segmented areas
    alpha = 0.7
    final_segmentation = base_image.copy().astype(np.float32)
    segmented_pixels = ~background_pixels

    final_segmentation[segmented_pixels] = (
            alpha * final_seg_colored_resized[segmented_pixels].astype(np.float32) +
            (1 - alpha) * base_image[segmented_pixels].astype(np.float32)
    )
    final_segmentation = final_segmentation.astype(np.uint8)

    # 3. Context-based Mask Overlay - blend with actual image using alpha
    context_overlay: np.ndarray
    if context_based_mask is not None:
        context_mask_np = context_based_mask.cpu().numpy()
        context_colored = mask_to_rgb(context_mask_np, int(context_mask_np.max()) + 1)
        context_colored_resized = cv2.resize(context_colored, (base_img_width, base_img_height),
                                             interpolation=cv2.INTER_NEAREST)

        # Create alpha blended context mask
        context_mask_resized = cv2.resize(context_mask_np.astype(np.uint8), (base_img_width, base_img_height),
                                          interpolation=cv2.INTER_NEAREST)
        background_pixels_context = context_mask_resized == 0

        # Alpha blending: 70% color overlay, 30% original image for context areas
        alpha_context = 0.7
        context_overlay_float = base_image.copy().astype(np.float32)
        context_pixels = ~background_pixels_context

        context_overlay_float[context_pixels] = (
                alpha_context * context_colored_resized[context_pixels].astype(np.float32) +
                (1 - alpha_context) * base_image[context_pixels].astype(np.float32)
        )
        context_overlay = context_overlay_float.astype(np.uint8)
    else:
        # Create empty overlay with image background if no context mask provided
        context_overlay = base_image.copy().astype(np.uint8)

    return bbox_overlay, final_segmentation, context_overlay


def save_frame_visualizations(
        bbox_overlay: np.ndarray,
        final_segmentation: np.ndarray,
        context_overlay: np.ndarray,
        save_dir: Path,
        frame_idx: int,
        create_united: bool = True
) -> None:
    """
    Save frame visualizations either as separate images or united horizontal concatenation.
    
    Args:
        bbox_overlay: Bounding box overlay image
        final_segmentation: Final segmentation overlay image
        context_overlay: Context-based mask overlay image
        save_dir: Directory to save images
        frame_idx: Frame index for naming
        create_united: If True, create horizontal concatenation; if False, save separately
    """
    import matplotlib.pyplot as plt

    if create_united:
        # Create united horizontal visualization
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        axes[0].imshow(bbox_overlay)
        axes[0].set_title(f"Frame {frame_idx}: Bounding Boxes", fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(final_segmentation)
        axes[1].set_title(f"Frame {frame_idx}: Final Segmentation", fontsize=14)
        axes[1].axis('off')

        axes[2].imshow(context_overlay)
        axes[2].set_title(f"Frame {frame_idx}: Context-based Mask", fontsize=14)
        axes[2].axis('off')

        plt.tight_layout()
        united_path = save_dir / f"frame_{frame_idx:03d}_united.png"
        plt.savefig(united_path, dpi=150, bbox_inches='tight')
        plt.close()

    else:
        # Save separate images
        import cv2

        bbox_path = save_dir / f"bboxes_{frame_idx:03d}.png"
        seg_path = save_dir / f"segmentation_{frame_idx:03d}.png"
        context_path = save_dir / f"context_{frame_idx:03d}.png"

        cv2.imwrite(str(bbox_path), cv2.cvtColor(bbox_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(seg_path), cv2.cvtColor(final_segmentation, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(context_path), cv2.cvtColor(context_overlay, cv2.COLOR_RGB2BGR))


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
    tracker = TrunkVideoTracker(tree_class_id=trunk_class_id, context_window_size=1)
    # Process each frame
    results = []
    for i, video_frame in enumerate(tqdm(video_frames, desc="Processing frames", unit="frame")):
        # Process frame through VideoTracker
        trunk_instance_mask, context_based_mask, bbox_tracking_info = tracker.process_frame(
            video_frame,
            patch_size,
            target_image_size_wh
        )

        # Store results including context mask and bbox tracking info
        frame_result = {
            'frame_idx': video_frame.frame_idx,
            'filename': video_frame.filename,
            'trunk_instance_mask': trunk_instance_mask,
            'context_based_mask': context_based_mask,
            'bbox_tracking_info': bbox_tracking_info,
            'num_trunk_instances': len(trunk_instance_mask.unique()) - 1,  # Exclude background
            'active_trunk_indexes': sorted(tracker.trunk_indexes)
        }
        results.append(frame_result)

        print(f"  Result: {frame_result['num_trunk_instances']} trunk instances")
        print(f"  Active trunk indexes: {frame_result['active_trunk_indexes']}")

    # Enhanced debug visualizations
    if debugging_folder:
        debugging_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving enhanced debug visualizations to {debugging_folder}")

        for i, result in enumerate(results):
            video_frame = video_frames[i]

            # Use the stored context_based_mask from the result
            context_based_mask = result.get('context_based_mask', None)

            # Create enhanced visualizations
            bbox_overlay, final_segmentation, context_overlay = create_enhanced_frame_visualization(
                video_frame=video_frame,
                result=result,
                context_based_mask=context_based_mask,
                patch_size=patch_size,
                data_folder=data_folder
            )

            # Save visualizations
            save_frame_visualizations(
                bbox_overlay=bbox_overlay,
                final_segmentation=final_segmentation,
                context_overlay=context_overlay,
                save_dir=debugging_folder,
                frame_idx=result['frame_idx'],
                create_united=tracker.create_united_visualization
            )

            print(f"  Saved visualization for frame {result['frame_idx']}")

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
    frame_indices = np.arange(200, 249).astype('int')

    result = main(
        data_folder=data_folder,
        cache_path=cache_path,
        trunk_class_id=tree_class_id,
        frame_indices=frame_indices,
        debugging_folder=debugging_folder,
        patch_size=16,
        image_size=1024
    )

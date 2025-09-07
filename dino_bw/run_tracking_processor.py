#!/usr/bin/env python3
"""
Enhanced tracking processor for tree object correspondence using DINOv3 features.

This script implements tree object tracking between two frames by:
1. Selecting two frame indices from the tracking dataset
2. Extracting tree objects using bounding boxes and segmentation masks
3. Filtering patches to only include tree class features
4. Computing averaged feature vectors per tree object
5. Finding correspondences using Hungarian matching algorithm
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tracking_data_processor import load_tracking_cache
from dino_bw.dino_embeddings_utils import get_class_idx_from_name
from bw_ml_common.datasets.data_accessor_factory import create_dataset_accessor


@dataclass
class TreeObject:
    """Data structure for a tree object with its features and metadata."""
    object_id: int
    bbox: Dict[str, float]  # x1, y1, x2, y2 in DINO coordinates
    original_bbox: Dict[str, float]  # x1, y1, x2, y2 in original coordinates
    patch_indices: List[Tuple[int, int]]  # (row, col) indices of tree patches
    feature_vectors: torch.Tensor  # Features of all tree patches in this object
    averaged_feature: torch.Tensor  # Mean feature vector for this object
    confidence: float  # Detection confidence
    frame_id: int

    def __post_init__(self):
        """Compute averaged feature after initialization."""
        if len(self.feature_vectors) > 0:
            self.averaged_feature = torch.mean(self.feature_vectors, dim=0)
        else:
            self.averaged_feature = torch.zeros(
                self.feature_vectors.shape[1] if len(self.feature_vectors.shape) > 1 else 384)


def select_frame_indices(cache_data: Dict[str, Any], first_frame_index: Optional[int] = None, frame_gap: int = 10) -> Tuple[int, int]:
    """
    Select two frame indices for tracking analysis.
    
    Args:
        cache_data: Loaded tracking cache data
        first_frame_index: Optional specific first frame index to use
        frame_gap: Minimum gap between selected frames
        
    Returns:
        Tuple of (frame1_idx, frame2_idx)
    """
    frames = cache_data['frames']
    num_frames = len(frames)

    # Find frames with both detections and segmentation
    valid_frames = []
    for i, frame in enumerate(frames):
        has_detections = len(frame.get('detections', [])) > 0
        has_segmentation = frame.get('segmentation') is not None
        if has_detections and has_segmentation:
            valid_frames.append(i)

    if len(valid_frames) < 2:
        print(f"Warning: Only {len(valid_frames)} frames with both detections and segmentation")
        # Fallback to frames with at least segmentation
        raise ValueError("Need at least 2 frames with detection data")

    # Select frames with appropriate gap
    frame1_idx = first_frame_index or valid_frames[len(valid_frames) // 3]  # First third
    frame2_idx = min(frame1_idx + frame_gap, valid_frames[-1])

    # Ensure frame2 is in valid_frames
    if frame2_idx not in valid_frames:
        frame2_idx = valid_frames[-1]

    print(f"Selected frames: {frame1_idx} and {frame2_idx}")
    print(f"Frame {frame1_idx}: {frames[frame1_idx]['filename']}")
    print(f"Frame {frame2_idx}: {frames[frame2_idx]['filename']}")

    return frame1_idx, frame2_idx


def extract_tree_objects_from_frame(
        frame_data: Dict[str, Any],
        tree_class_id: int,
        patch_size: int,
        min_tree_patches: int = 3
) -> List[TreeObject]:
    """
    Extract tree objects from a single frame using bounding boxes and segmentation.
    
    Args:
        frame_data: Single frame data from cache
        tree_class_id: Class ID for tree objects (should be 2 for bw2508)
        patch_size: DINO patch size
        min_tree_patches: Minimum number of tree patches required for valid object
        
    Returns:
        List of TreeObject instances
    """
    features = torch.from_numpy(frame_data['features'])  # Shape: [num_patches, feature_dim]
    segmentation = frame_data['segmentation']  # Shape: [h_patches, w_patches]
    detections = frame_data.get('detections_scaled', [])
    frame_id = frame_data['frame_id']
    target_size = frame_data['target_image_size']  # (width, height)

    if segmentation is None:
        print(f"Warning: No segmentation data for frame {frame_id}")
        return []

    h_patches, w_patches = segmentation.shape
    feature_dim = features.shape[1]

    # Reshape features to match segmentation grid
    features_2d = features.view(h_patches, w_patches, feature_dim)

    # Create tree mask
    tree_mask = (segmentation == tree_class_id)

    tree_objects = []

    # Process each detection to see if it contains trees
    for det_idx, detection in enumerate(detections):
        if 'box' not in detection or detection['name'] != 'trunk':
            continue

        bbox = detection['box']
        confidence = detection.get('confidence', 1.0)

        # Convert bounding box to patch coordinates
        x1_patch = max(0, int(bbox['x1'] / patch_size))
        y1_patch = max(0, int(bbox['y1'] / patch_size))
        x2_patch = min(w_patches, int(bbox['x2'] / patch_size) + 1)
        y2_patch = min(h_patches, int(bbox['y2'] / patch_size) + 1)

        # Extract tree patches within bounding box
        bbox_trunk_mask = tree_mask[y1_patch:y2_patch, x1_patch:x2_patch]

        if not torch.any(torch.from_numpy(bbox_trunk_mask)):
            continue  # No tree patches in this bounding box

        # Get patch indices and features for tree patches in this bbox
        patch_indices = []
        patch_features = []

        for i in range(y1_patch, y2_patch):
            for j in range(x1_patch, x2_patch):
                if tree_mask[i, j]:
                    patch_indices.append((i, j))
                    patch_features.append(features_2d[i, j])

        if len(patch_features) < min_tree_patches:
            continue  # Not enough tree patches

        # Create tree object
        tree_obj = TreeObject(
            object_id=det_idx,
            bbox=bbox,
            original_bbox=detection.get('original_box', bbox),
            patch_indices=patch_indices,
            feature_vectors=torch.stack(patch_features),
            averaged_feature=torch.zeros(feature_dim),  # Will be computed in __post_init__
            confidence=confidence,
            frame_id=frame_id
        )

        tree_objects.append(tree_obj)

    # If no detections contain trees, create objects from tree clusters
    if not tree_objects:
        print(f"No detections contain trees in frame {frame_id}, creating objects from tree clusters")
        tree_objects = create_tree_objects_from_clusters(
            features_2d, tree_mask, frame_id, patch_size, target_size, min_tree_patches
        )

    print(f"Extracted {len(tree_objects)} tree objects from frame {frame_id}")
    return tree_objects


def create_tree_objects_from_clusters(
        features_2d: torch.Tensor,
        tree_mask: np.ndarray,
        frame_id: int,
        patch_size: int,
        target_size: Tuple[int, int],
        min_tree_patches: int
) -> List[TreeObject]:
    """
    Create tree objects by clustering connected tree patches when no detections are available.
    """
    from scipy import ndimage

    # Find connected components in tree mask
    labeled_mask, num_components = ndimage.label(tree_mask)

    tree_objects = []
    h_patches, w_patches = tree_mask.shape

    for component_id in range(1, num_components + 1):
        component_mask = (labeled_mask == component_id)
        component_indices = np.where(component_mask)

        if len(component_indices[0]) < min_tree_patches:
            continue

        # Get patch indices and features
        patch_indices = list(zip(component_indices[0], component_indices[1]))
        patch_features = [features_2d[i, j] for i, j in patch_indices]

        # Create bounding box around component
        min_row, max_row = component_indices[0].min(), component_indices[0].max()
        min_col, max_col = component_indices[1].min(), component_indices[1].max()

        bbox = {
            'x1': min_col * patch_size,
            'y1': min_row * patch_size,
            'x2': (max_col + 1) * patch_size,
            'y2': (max_row + 1) * patch_size
        }

        tree_obj = TreeObject(
            object_id=component_id - 1,
            bbox=bbox,
            original_bbox=bbox,  # Same as bbox since we don't have original detections
            patch_indices=patch_indices,
            feature_vectors=torch.stack(patch_features),
            averaged_feature=torch.zeros(features_2d.shape[2]),
            confidence=1.0,
            frame_id=frame_id
        )

        tree_objects.append(tree_obj)

    return tree_objects


def visualize_tree_objects(
        frame_data: Dict[str, Any],
        tree_objects: List[TreeObject],
        data_folder: Path,
        save_path: Optional[Path] = None
) -> None:
    """
    Visualize tree objects on the original image.
    """
    # Load original image
    image_path = data_folder / "images" / frame_data['filename']
    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert('RGB')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(f"Frame {frame_data['frame_id']}: {len(tree_objects)} Tree Objects")

    # Draw bounding boxes
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = [color_list[i % len(color_list)] for i in range(len(tree_objects))]

    for i, tree_obj in enumerate(tree_objects):
        bbox = tree_obj.original_bbox
        rect = patches.Rectangle(
            (bbox['x1'], bbox['y1']),
            bbox['x2'] - bbox['x1'],
            bbox['y2'] - bbox['y1'],
            linewidth=2,
            edgecolor=colors[i],
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        ax.text(
            bbox['x1'], bbox['y1'] - 10,
            f"Tree {tree_obj.object_id}\n({len(tree_obj.patch_indices)} patches)",
            color=colors[i],
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def compute_tree_correspondences(
        trees_frame1: List[TreeObject],
        trees_frame2: List[TreeObject],
        similarity_threshold: float = 0.7
) -> List[Tuple[int, int, float]]:
    """
    Compute correspondences between tree objects using Hungarian matching.
    
    Args:
        trees_frame1: Tree objects from first frame
        trees_frame2: Tree objects from second frame  
        similarity_threshold: Minimum cosine similarity for valid match
        
    Returns:
        List of (tree1_idx, tree2_idx, similarity_score) tuples
    """
    if not trees_frame1 or not trees_frame2:
        return []

    # Compute similarity matrix using cosine similarity
    n1, n2 = len(trees_frame1), len(trees_frame2)
    similarity_matrix = torch.zeros(n1, n2)

    for i, tree1 in enumerate(trees_frame1):
        for j, tree2 in enumerate(trees_frame2):
            # Normalize features for cosine similarity
            feat1_norm = F.normalize(tree1.averaged_feature.unsqueeze(0), p=2, dim=1)
            feat2_norm = F.normalize(tree2.averaged_feature.unsqueeze(0), p=2, dim=1)

            # Cosine similarity
            similarity = torch.mm(feat1_norm, feat2_norm.t()).item()
            similarity_matrix[i, j] = similarity

    # Convert to cost matrix (Hungarian algorithm minimizes cost)
    cost_matrix = 1.0 - similarity_matrix.numpy()

    # Apply Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Filter matches by similarity threshold
    correspondences = []
    for i, j in zip(row_indices, col_indices):
        similarity = similarity_matrix[i, j].item()
        if similarity >= similarity_threshold:
            correspondences.append((i, j, similarity))

    print(f"Found {len(correspondences)} valid correspondences out of {len(row_indices)} potential matches")
    return correspondences


def main(data_folder, cache_path, debugging_folder=None, patch_size=8, image_size=1024):
    """
    Main function for enhanced tracking processor with improved tree trunk detection.
    
    Args:
        data_folder: Path to data folder (default: turn8_0)
        cache_path: Path to cache file (auto-generated if None)
        debugging_folder: Optional debugging output folder
        patch_size: DINO patch size (8 for better tree trunk detection, default was 16)
        image_size: DINO image size (1024 for higher resolution, default was 768)
    """
      
    print(f"=== ENHANCED TREE TRACKING CONFIGURATION ===")
    print(f"Patch size: {patch_size}x{patch_size} (smaller = better for narrow trunks)")
    print(f"Image size: {image_size}px (higher = more detail)")
    print(f"Spatial resolution improvement: {(16/patch_size)**2:.1f}x better than default")
    print(f"Cache file: {cache_path.name}")

    # Check if we need to regenerate cache with new parameters
    if not cache_path.exists():
        print(f"Cache not found at {cache_path}")
        warnings.warn("Please run the basic tracking processor first to generate the cache.")
        return

    # Load tracking cache
    print("\nLoading tracking cache...")
    cache_data = load_tracking_cache(cache_path)
    
    # Verify cache parameters match our requirements
    config = cache_data.get('config', {})
    cache_patch_size = config.get('patch_size', 16)
    cache_image_size = config.get('image_size', 768)
    
    if cache_patch_size != patch_size or cache_image_size != image_size:
        print(f"⚠️  WARNING: Cache parameters don't match!")
        print(f"   Cache: patch_size={cache_patch_size}, image_size={cache_image_size}")
        print(f"   Requested: patch_size={patch_size}, image_size={image_size}")
        warnings.warn(f"   Regenerate cache with correct parameters (suing `tracking_data_processor`)...")
        return
        
    print(f"Loaded cache with {len(cache_data['frames'])} frames")

    # Get tree class ID
    dataset_path = Path("/home/nati/source/datasets/bw2508")
    accessor = create_dataset_accessor(dataset_name=dataset_path.stem, data_root=str(dataset_path.parent),
                                       split_name="train")
    tree_class_id = get_class_idx_from_name(accessor, "tree")
    print(f"Tree class ID: {tree_class_id}")

    # Stage 1: Select two frame indices
    print("\n=== STAGE 1: SELECTING FRAMES ===")
    frame1_idx, frame2_idx = select_frame_indices(cache_data, first_frame_index=150, frame_gap=5)

    frame1_data = cache_data['frames'][frame1_idx]
    frame2_data = cache_data['frames'][frame2_idx]

    # Stage 2: Extract tree objects from both frames
    print("\n=== STAGE 2: EXTRACTING TREE OBJECTS ===")
    patch_size = cache_data['config']['patch_size']

    print(f"Processing frame {frame1_idx}...")
    trees_frame1 = extract_tree_objects_from_frame(frame1_data, tree_class_id, patch_size)

    print(f"Processing frame {frame2_idx}...")
    trees_frame2 = extract_tree_objects_from_frame(frame2_data, tree_class_id, patch_size)

    if not trees_frame1 or not trees_frame2:
        print("Error: No tree objects found in one or both frames")
        return

    # Display tree object information
    print(f"\nFrame {frame1_idx} tree objects:")
    for i, tree in enumerate(trees_frame1):
        print(f"  Tree {i}: {len(tree.patch_indices)} patches, confidence: {tree.confidence:.3f}")

    print(f"\nFrame {frame2_idx} tree objects:")
    for i, tree in enumerate(trees_frame2):
        print(f"  Tree {i}: {len(tree.patch_indices)} patches, confidence: {tree.confidence:.3f}")

    # Visualize tree objects
    print("\n=== VISUALIZING TREE OBJECTS ===")
    save_path = debugging_folder / "tree_objects_frame1.png" if debugging_folder is not None else None
    visualize_tree_objects(frame1_data, trees_frame1, data_folder,
                           save_path=save_path)
    
    save_path = debugging_folder / "tree_objects_frame2.png" if debugging_folder is not None else None
    visualize_tree_objects(frame2_data, trees_frame2, data_folder,
                           save_path=save_path)

    # Approval checkpoint
    print("\n" + "=" * 50)
    print("APPROVAL CHECKPOINT")
    print("=" * 50)
    print(f"Successfully extracted tree objects:")
    print(f"  Frame {frame1_idx}: {len(trees_frame1)} tree objects")
    print(f"  Frame {frame2_idx}: {len(trees_frame2)} tree objects")
    print(f"  Feature dimension: {trees_frame1[0].averaged_feature.shape[0] if trees_frame1 else 'N/A'}")
    print("\nTree object visualizations have been saved as:")
    print("  - tree_objects_frame1.png")
    print("  - tree_objects_frame2.png")
    print("\nPlease review the extracted tree objects before proceeding.")

    response = input("\nDo you approve proceeding with the matching logic? (y/n): ").lower().strip()

    if response != 'y':
        print("Stopping at approval checkpoint.")
        return

    # Stage 3: Compute correspondences
    print("\n=== STAGE 3: COMPUTING TREE CORRESPONDENCES ===")
    correspondences = compute_tree_correspondences(trees_frame1, trees_frame2, similarity_threshold=0.6)

    if not correspondences:
        print("No valid correspondences found!")
        return

    print(f"\nFound {len(correspondences)} correspondences:")
    for i, (tree1_idx, tree2_idx, similarity) in enumerate(correspondences):
        tree1 = trees_frame1[tree1_idx]
        tree2 = trees_frame2[tree2_idx]
        print(f"  Match {i + 1}: Tree {tree1_idx} (frame {frame1_idx}) <-> Tree {tree2_idx} (frame {frame2_idx})")
        print(f"    Similarity: {similarity:.3f}")
        print(f"    Patches: {len(tree1.patch_indices)} <-> {len(tree2.patch_indices)}")

    print("\n=== TRACKING ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    data_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0")
    cache_path = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/tracking_cache.pkl")
    main(data_folder, cache_path)

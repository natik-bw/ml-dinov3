#!/usr/bin/env python3
"""
Feature extraction utilities for tree tracking using DINOv3 features.

This module contains functions for:
- Tree object extraction from frames
- Unified PCA computation across multiple frames
- Feature processing and analysis
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from sklearn.decomposition import PCA


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


def extract_tree_objects_from_frame(
    frame_data: Dict[str, Any],
    tree_class_id: int,
    patch_size: int,
    min_tree_patches: int = 5
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
        if 'box' not in detection:
            continue

        bbox = detection['box']
        confidence = detection.get('confidence', 1.0)

        # Convert bounding box to patch coordinates
        x1_patch = max(0, int(bbox['x1'] / patch_size))
        y1_patch = max(0, int(bbox['y1'] / patch_size))
        x2_patch = min(w_patches, int(bbox['x2'] / patch_size) + 1)
        y2_patch = min(h_patches, int(bbox['y2'] / patch_size) + 1)

        # Extract tree patches within bounding box
        bbox_tree_mask = tree_mask[y1_patch:y2_patch, x1_patch:x2_patch]

        if not torch.any(torch.from_numpy(bbox_tree_mask)):
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


def compute_unified_pca(*tree_lists: List[TreeObject]) -> PCA:
    """
    Compute a unified 3-channel PCA from multiple lists of tree objects.
    
    This function collects all tree feature vectors from multiple frames/lists,
    fits a single PCA model, and returns it for consistent color representation
    across all tree objects.
    
    Args:
        *tree_lists: Variable number of lists containing TreeObject instances
        
    Returns:
        Fitted PCA model with 3 components for RGB color mapping
        
    Raises:
        ValueError: If no tree objects are provided or if trees have no features
    """
    # Collect all feature vectors from all tree lists
    all_features = []

    for tree_list in tree_lists:
        for tree_obj in tree_list:
            if len(tree_obj.feature_vectors) > 0:
                # Add all patch features from this tree
                all_features.append(tree_obj.feature_vectors)

    if not all_features:
        raise ValueError("No tree objects with features found")

    # Concatenate all features into a single tensor
    combined_features = torch.cat(all_features, dim=0)

    # Convert to numpy for sklearn PCA
    features_numpy = combined_features.numpy()

    print(f"Computing unified PCA from {len(features_numpy)} feature vectors")
    print(f"Feature dimension: {features_numpy.shape[1]}")

    # Fit PCA with 3 components for RGB mapping
    pca = PCA(n_components=3, whiten=True)
    pca.fit(features_numpy)

    # Print explained variance for information
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    return pca


def apply_pca_coloring(tree_objects: List[TreeObject], pca: PCA) -> np.ndarray:
    """
    Apply PCA transformation to tree objects to get RGB color representations.
    
    Args:
        tree_objects: List of TreeObject instances
        pca: Fitted PCA model with 3 components
        
    Returns:
        RGB colors array of shape (n_trees, 3) with values in [0, 1]
    """
    if not tree_objects or pca is None:
        return np.array([])

    # Get averaged features for each tree
    tree_features = []
    for tree_obj in tree_objects:
        tree_features.append(tree_obj.averaged_feature.numpy())

    tree_features_array = np.array(tree_features)

    # Apply PCA transformation
    pca_transformed = pca.transform(tree_features_array)

    # Convert to RGB colors using sigmoid for vibrant colors with green dominance
    # Reorder components to make green (component 1) most dominant
    rgb_colors = torch.sigmoid(torch.from_numpy(pca_transformed[:, [1, 0, 2]]) * 2.0).numpy()

    return rgb_colors


def apply_pca_coloring_to_patches(tree_objects: List[TreeObject], pca: PCA,
                                  patch_size: int, target_image_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Apply PCA transformation to individual patches within tree objects and create spatial color maps.
    
    Args:
        tree_objects: List of TreeObject instances
        pca: Fitted PCA model with 3 components
        patch_size: Size of each patch in pixels
        target_image_size: (width, height) of the target image for patch grid
        
    Returns:
        Dictionary containing:
        - 'patch_colors': RGB color map of shape (h_patches, w_patches, 3)
        - 'patch_mask': Boolean mask of shape (h_patches, w_patches) indicating tree patches
        - 'tree_id_map': Map of shape (h_patches, w_patches) with tree IDs (-1 for non-tree)
    """
    if not tree_objects or pca is None:
        return {}

    # Calculate patch grid dimensions
    h_patches = target_image_size[1] // patch_size
    w_patches = target_image_size[0] // patch_size

    # Initialize output arrays
    patch_colors = np.zeros((h_patches, w_patches, 3))
    patch_mask = np.zeros((h_patches, w_patches), dtype=bool)
    tree_id_map = np.full((h_patches, w_patches), -1, dtype=int)

    print(f"Creating patch color map: {h_patches}x{w_patches} patches")

    # Process each tree object
    for tree_obj in tree_objects:
        if len(tree_obj.feature_vectors) == 0:
            continue

        # Apply PCA to all patch features in this tree
        patch_features = tree_obj.feature_vectors.numpy()
        pca_transformed = pca.transform(patch_features)

        # Convert to RGB colors with green dominance (reorder components)
        patch_rgb = torch.sigmoid(torch.from_numpy(pca_transformed[:, [1, 0, 2]]) * 2.0).numpy()

        # Map colors to their spatial locations
        for i, (patch_row, patch_col) in enumerate(tree_obj.patch_indices):
            if 0 <= patch_row < h_patches and 0 <= patch_col < w_patches:
                patch_colors[patch_row, patch_col] = patch_rgb[i]
                patch_mask[patch_row, patch_col] = True
                tree_id_map[patch_row, patch_col] = tree_obj.object_id

    return {
        'patch_colors': patch_colors,
        'patch_mask': patch_mask,
        'tree_id_map': tree_id_map
    }


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


def select_frame_indices(cache_data: Dict[str, Any], first_frame_index: Optional[int] = None, frame_gap: int = 10) -> \
        Tuple[int, int]:
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

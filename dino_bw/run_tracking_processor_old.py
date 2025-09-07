#!/usr/bin/env python3
"""
Enhanced tracking processor for tree object correspondence using DINOv3 features.

This script implements multi-frame tree object tracking by:
1. Processing multiple frame indices from the tracking dataset
2. Extracting tree objects from each frame using bounding boxes and segmentation masks
3. Computing unified PCA across all frames for consistent coloring
4. Creating comprehensive visualizations for individual frames and correspondences
5. Finding correspondences between the first two frames using Hungarian matching
"""

import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

from tracking_data_processor import load_tracking_cache
from dino_bw.dino_embeddings_utils import get_class_idx_from_name
from bw_ml_common.datasets.data_accessor_factory import create_dataset_accessor

from feature_extraction_utils import (
    TreeObject, extract_tree_objects_from_frame, compute_unified_pca,
    apply_pca_coloring_to_patches, compute_tree_correspondences
)
from tracking_vis_utils import (
    visualize_tree_objects, visualize_patch_colors, visualize_tree_correspondences,
    visualize_united_pca_overlay
)


def main(data_folder, cache_path, debugging_folder=None, frame_indices: List[int] = None):
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


def visualize_patch_colors(
        frame_data: Dict[str, Any],
        patch_color_data: Dict[str, np.ndarray],
        data_folder: Path,
        patch_size: int,
        alpha: float = 0.5,
        save_path: Optional[Path] = None
) -> None:
    """
    Visualize patch-level PCA colors overlaid on the original image.
    
    Args:
        frame_data: Frame data containing image filename
        patch_color_data: Output from apply_pca_coloring_to_patches
        data_folder: Path to data folder containing images
        patch_size: Size of each patch in pixels
        alpha: Transparency of color overlay (0=transparent, 1=opaque)
        save_path: Optional save path for the visualization
    """
    if not patch_color_data:
        print("No patch color data provided")
        return

    # Load original image
    image_path = data_folder / "images" / frame_data['filename']
    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return

    original_image = Image.open(image_path).convert('RGB')
    target_image_size = frame_data['target_image_size']  # (width, height)

    # Resize original image to match target size
    resized_image = original_image.resize(target_image_size)

    # Get patch data
    patch_colors = patch_color_data['patch_colors']
    patch_mask = patch_color_data['patch_mask']
    h_patches, w_patches = patch_colors.shape[:2]

    # Create color overlay image
    overlay = np.zeros((target_image_size[1], target_image_size[0], 3))
    overlay_mask = np.zeros((target_image_size[1], target_image_size[0]), dtype=bool)

    # Fill overlay with patch colors
    for i in range(h_patches):
        for j in range(w_patches):
            if patch_mask[i, j]:  # Only show tree patches
                # Calculate pixel coordinates for this patch
                y_start = i * patch_size
                y_end = min((i + 1) * patch_size, target_image_size[1])
                x_start = j * patch_size
                x_end = min((j + 1) * patch_size, target_image_size[0])

                # Set color for this patch
                overlay[y_start:y_end, x_start:x_end] = patch_colors[i, j]
                overlay_mask[y_start:y_end, x_start:x_end] = True

    # Create matplotlib figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    ax1.imshow(resized_image)
    ax1.set_title(f"Original Image - Frame {frame_data['frame_id']}")
    ax1.axis('off')

    # PCA color overlay only
    ax2.imshow(overlay)
    ax2.set_title("PCA Colors (Tree Patches Only)")
    ax2.axis('off')

    # Combined overlay
    combined = np.array(resized_image, dtype=float) / 255.0
    # Apply overlay where tree patches exist
    combined[overlay_mask] = (1 - alpha) * combined[overlay_mask] + alpha * overlay[overlay_mask]

    ax3.imshow(combined)
    ax3.set_title(f"Combined (α={alpha})")
    ax3.axis('off')

    # Add patch grid visualization for reference
    for ax in [ax2, ax3]:
        # Draw patch grid lines (light gray, subtle)
        for i in range(0, target_image_size[1], patch_size):
            ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.3)
        for j in range(0, target_image_size[0], patch_size):
            ax.axvline(x=j, color='lightgray', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Patch color visualization saved to {save_path}")

    plt.show()

    # Print statistics
    tree_patches_count = np.sum(patch_mask)


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
    Visualize tree objects on the original image with concise labels.
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

        # Add concise label in top-right corner of bounding box
        ax.text(
            bbox['x2'] - 5, bbox['y1'] + 15,  # Top-right corner with small offset
            f"{tree_obj.object_id}",
            color=colors[i],
            fontsize=8,  # Smaller font
            weight='normal',  # Not bold
            ha='right',  # Right-aligned to stay within bbox
            va='top',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=colors[i])
        )

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def visualize_tree_correspondences(
        frame1_data: Dict[str, Any],
        frame2_data: Dict[str, Any],
        trees_frame1: List[TreeObject],
        trees_frame2: List[TreeObject],
        correspondences: List[Tuple[int, int, float]],
        data_folder: Path,
        save_path: Optional[Path] = None
) -> None:
    """
    Visualize tree correspondences between two frames with color-coded legend.
    
    Args:
        frame1_data: First frame data
        frame2_data: Second frame data
        trees_frame1: Tree objects from first frame
        trees_frame2: Tree objects from second frame
        correspondences: List of (tree1_idx, tree2_idx, similarity_score) tuples
        data_folder: Path to data folder containing images
        save_path: Optional save path for the visualization
    """
    # Load both images
    image1_path = data_folder / "images" / frame1_data['filename']
    image2_path = data_folder / "images" / frame2_data['filename']

    if not image1_path.exists() or not image2_path.exists():
        print(f"Warning: Images not found at {image1_path} or {image2_path}")
        return

    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    # Make images the same width for vertical stacking
    width = max(image1.width, image2.width)
    image1_resized = image1.resize((width, int(image1.height * width / image1.width)))
    image2_resized = image2.resize((width, int(image2.height * width / image2.width)))

    # Create vertically stacked image using PIL
    total_height = image1_resized.height + image2_resized.height
    combined_image = Image.new('RGB', (width, total_height))
    combined_image.paste(image1_resized, (0, 0))
    combined_image.paste(image2_resized, (0, image1_resized.height))

    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.imshow(combined_image)
    ax.set_title(f"Tree Correspondences: Frame {frame1_data['frame_id']} → Frame {frame2_data['frame_id']}")

    # Color map for correspondences
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Calculate scaling factors for bounding box coordinates
    scale1_x = image1_resized.width / image1.width
    scale1_y = image1_resized.height / image1.height
    scale2_x = image2_resized.width / image2.width
    scale2_y = image2_resized.height / image2.height

    # Create correspondence color mapping
    correspondence_colors: Dict[Tuple[int, int], Tuple[str, float]] = {}
    for tree1_idx, tree2_idx, similarity in correspondences:
        color = color_list[len(correspondence_colors) % len(color_list)]
        correspondence_colors[(tree1_idx, tree2_idx)] = (color, similarity)

    # Draw tree objects with correspondence colors only
    for tree1_idx, tree2_idx, similarity in correspondences:
        color, _ = correspondence_colors[(tree1_idx, tree2_idx)]
        line_width = max(2, int(similarity * 6))  # Thicker lines for better visibility

        # Draw first frame tree
        tree1 = trees_frame1[tree1_idx]
        bbox1 = tree1.original_bbox
        scaled_bbox1 = {
            'x1': bbox1['x1'] * scale1_x,
            'y1': bbox1['y1'] * scale1_y,
            'x2': bbox1['x2'] * scale1_x,
            'y2': bbox1['y2'] * scale1_y
        }

        rect1 = patches.Rectangle(
            (scaled_bbox1['x1'], scaled_bbox1['y1']),
            scaled_bbox1['x2'] - scaled_bbox1['x1'],
            scaled_bbox1['y2'] - scaled_bbox1['y1'],
            linewidth=line_width,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect1)

        # Draw second frame tree
        tree2 = trees_frame2[tree2_idx]
        bbox2 = tree2.original_bbox
        scaled_bbox2 = {
            'x1': bbox2['x1'] * scale2_x,
            'y1': bbox2['y1'] * scale2_y + image1_resized.height,
            'x2': bbox2['x2'] * scale2_x,
            'y2': bbox2['y2'] * scale2_y + image1_resized.height
        }

        rect2 = patches.Rectangle(
            (scaled_bbox2['x1'], scaled_bbox2['y1']),
            scaled_bbox2['x2'] - scaled_bbox2['x1'],
            scaled_bbox2['y2'] - scaled_bbox2['y1'],
            linewidth=line_width,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect2)

    # Create legend in top-right corner
    legend_x = width - 200  # Position from right edge
    legend_y = 30  # Position from top
    legend_line_height = 25

    # Legend title
    ax.text(legend_x, legend_y, "Correspondences:", fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

    # Legend entries
    for i, (tree1_idx, tree2_idx, similarity) in enumerate(correspondences):
        color, _ = correspondence_colors[(tree1_idx, tree2_idx)]
        y_pos = legend_y + (i + 1) * legend_line_height

        # Draw colored square indicator
        legend_rect = patches.Rectangle(
            (legend_x, y_pos - 8), 16, 16,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
        )
        ax.add_patch(legend_rect)

        # Add text
        ax.text(legend_x + 25, y_pos, f"{tree1_idx} → {tree2_idx}: {similarity:.2f}",
                fontsize=10, va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Correspondence visualization saved to {save_path}")

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


def main(data_folder, cache_path, debugging_folder=None, patch_size=16, image_size=1024):
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
    print(f"Spatial resolution improvement: {(16 / patch_size) ** 2:.1f}x better than default")
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

    if debugging_folder is not None:
        debugging_folder.mkdir(exist_ok=True, parents=True)

    # Get tree class ID
    dataset_path = Path("/home/nati/source/datasets/bw2508")
    accessor = create_dataset_accessor(dataset_name=dataset_path.stem, data_root=str(dataset_path.parent),
                                       split_name="train")
    tree_class_id = get_class_idx_from_name(accessor, "tree")
    print(f"Tree class ID: {tree_class_id}")

    # Stage 1: Select two frame indices
    print("\n=== STAGE 1: SELECTING FRAMES ===")
    frame1_idx, frame2_idx = select_frame_indices(cache_data, first_frame_index=200, frame_gap=5)

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

    # Compute unified PCA from all tree objects
    print("\n=== COMPUTING UNIFIED PCA ===")
    try:
        unified_pca = compute_unified_pca(trees_frame1, trees_frame2)
        print("✓ Unified PCA computed successfully")
    except ValueError as e:
        print(f"Warning: Could not compute unified PCA: {e}")
        unified_pca = None

    # Display tree object information
    print(f"\nFrame {frame1_idx} tree objects:")
    for i, tree in enumerate(trees_frame1):
        print(f"  Tree {i}: {len(tree.patch_indices)} patches, confidence: {tree.confidence:.3f}")

    print(f"\nFrame {frame2_idx} tree objects:")
    for i, tree in enumerate(trees_frame2):
        print(f"  Tree {i}: {len(tree.patch_indices)} patches, confidence: {tree.confidence:.3f}")

    # Visualize tree objects
    print("\n=== VISUALIZING TREE OBJECTS ===")
    save_path_im1 = debugging_folder / "tree_objects_frame1.png" if debugging_folder is not None else None
    visualize_tree_objects(frame1_data, trees_frame1, data_folder,
                           save_path=save_path_im1)

    save_path_im2 = debugging_folder / "tree_objects_frame2.png" if debugging_folder is not None else None
    visualize_tree_objects(frame2_data, trees_frame2, data_folder,
                           save_path=save_path_im2)

    # Create and visualize patch-level PCA colors
    if unified_pca is not None:
        print("\n=== CREATING PATCH-LEVEL PCA VISUALIZATIONS ===")

        # Apply PCA coloring to patches for each frame
        patch_colors_frame1 = apply_pca_coloring_to_patches(
            trees_frame1, unified_pca, patch_size, frame1_data['target_image_size']
        )
        patch_colors_frame2 = apply_pca_coloring_to_patches(
            trees_frame2, unified_pca, patch_size, frame2_data['target_image_size']
        )

        # Visualize patch colors
        save_path_patches1 = debugging_folder / "patch_colors_frame1.png" if debugging_folder is not None else None
        visualize_patch_colors(frame1_data, patch_colors_frame1, data_folder, patch_size,
                               alpha=0.7, save_path=save_path_patches1)

        save_path_patches2 = debugging_folder / "patch_colors_frame2.png" if debugging_folder is not None else None
        visualize_patch_colors(frame2_data, patch_colors_frame2, data_folder, patch_size,
                               alpha=0.7, save_path=save_path_patches2)

    # Approval checkpoint
    print("\n" + "=" * 50)
    print("APPROVAL CHECKPOINT")
    print("=" * 50)
    print(f"Successfully extracted tree objects:")
    print(f"  Frame {frame1_idx}: {len(trees_frame1)} tree objects")
    print(f"  Frame {frame2_idx}: {len(trees_frame2)} tree objects")
    print(f"  Feature dimension: {trees_frame1[0].averaged_feature.shape[0] if trees_frame1 else 'N/A'}")
    if save_path_im1 is not None:
        print(f"\nVisualizations have been saved:")
        print(f"  - Tree objects: {save_path_im1.name} and {save_path_im2.name}")
        if unified_pca is not None:
            print(f"  - Patch colors: {save_path_patches1.name} and {save_path_patches2.name}")
        print("\nPlease review the extracted tree objects and patch visualizations before proceeding.")

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

    # Visualize correspondences
    print("\n=== VISUALIZING CORRESPONDENCES ===")
    save_path_corr = debugging_folder / "tree_correspondences.png" if debugging_folder is not None else None
    visualize_tree_correspondences(
        frame1_data, frame2_data, trees_frame1, trees_frame2, correspondences, data_folder, save_path_corr
    )

    print("\n=== TRACKING ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    data_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0")
    cache_path = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/tracking_cache_img_1024_patch_16.pkl")
    debugging_folder = Path("/home/nati/source/data/greeting_dev/bags_parsed/turn8_0/tracking_debug/")
    main(data_folder, cache_path, debugging_folder, patch_size=16, image_size=1024)

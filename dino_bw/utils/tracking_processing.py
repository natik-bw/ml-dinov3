import functools
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


class VideoFrameData:
    """Container for video frame data extracted from cache."""

    def __init__(self, frame_data: Dict[str, Any], frame_idx: int):
        self.frame_idx = frame_idx
        # self.frame_data = frame_data
        self.filename = frame_data['filename']
        self.frame_number = frame_data['frame_number']
        self.segmentation = frame_data.get('segmentation')  # [H_patches, W_patches] or None

        # Reshape features to [H, W, D] format
        if self.segmentation is not None:
            seg_array = np.array(self.segmentation)
            h_patches, w_patches = seg_array.shape
            self.features = torch.from_numpy(np.reshape(frame_data['features'], (h_patches, w_patches, -1))).cuda()
        else:
            # Fallback: assume square patch grid or use original feature shape
            features_flat = frame_data['features']
            total_patches = len(features_flat)
            patch_dim = int(np.sqrt(total_patches))  # Assume square grid
            feature_dim = features_flat.shape[-1] if len(features_flat.shape) > 1 else features_flat.shape[
                                                                                           0] // total_patches
            self.features = torch.from_numpy(np.reshape(features_flat, (patch_dim, patch_dim, feature_dim))).cuda()
        self.original_image_size = frame_data['original_image_size']  # (width, height)
        self.target_image_size = frame_data['target_image_size']  # (width, height)
        self.detections = frame_data.get('detections', [])
        self.detections_scaled = frame_data.get('detections_scaled', [])

    @property
    def has_segmentation(self) -> bool:
        return self.segmentation is not None

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0

    @property
    def feature_height(self) -> int:
        return self.features.shape[0]

    @property
    def feature_width(self) -> int:
        return self.features.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[2]


class TreesTracker:
    """
    Manages tree instance allocation and semantic-to-instance conversion.

    Handles:
    - Assignment of unique tree instance IDs
    - Conversion from semantic segmentation to instance segmentation
    - Tracking tree instances across frames using detection boxes
    """

    def __init__(self, tree_class_id: int = 2, background_class_id: int = 0):
        self.tree_class_id = tree_class_id
        self.background_class_id = background_class_id  # Use 0 for background
        self.next_instance_id = 1  # Start from 1, background is 0
        self.active_instances: Dict[int, Dict[str, Any]] = {}  # instance_id -> metadata
        self.frame_count = 0

    def get_next_instance_id(self) -> int:
        """Get the next available instance ID."""
        instance_id = self.next_instance_id
        self.next_instance_id += 1
        return instance_id

    def create_initial_instances(
            self,
            segmentation_mask: torch.Tensor,
            detections: List[Dict],
            patch_size: int,
            target_image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Create initial tree instances from semantic segmentation and detections.

        Args:
            segmentation_mask: [H_patches, W_patches] semantic segmentation
            detections: List of detection boxes
            patch_size: Size of patches
            target_image_size: (width, height) of target image

        Returns:
            Tuple of (instance_mask, class_mapping)
        """
        print(f"=== CREATING INITIAL TREE INSTANCES (Frame {self.frame_count}) ===")

        h_patches, w_patches = segmentation_mask.shape

        # Initialize instance mask with all background (0)
        instance_mask = torch.full_like(segmentation_mask, self.background_class_id)

        # Find tree patches
        tree_patches = (segmentation_mask == self.tree_class_id)
        print(f"Found {tree_patches.sum().item()} tree patches")

        # Debug: Print original segmentation unique classes
        orig_classes = segmentation_mask.unique()
        print(f"Original segmentation classes: {orig_classes.cpu().numpy()}")

        # Process detections to create instances
        instance_count = 0
        for detection in detections:
            if detection.get('name') == 'trunk':  # Filter for tree detections
                box = detection['box']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                # Convert to patch coordinates
                patch_x1 = max(0, min(int(x1 / patch_size), w_patches - 1))
                patch_y1 = max(0, min(int(y1 / patch_size), h_patches - 1))
                patch_x2 = max(0, min(int(x2 / patch_size), w_patches - 1))
                patch_y2 = max(0, min(int(y2 / patch_size), h_patches - 1))

                # Find tree patches in this box
                box_tree_patches = tree_patches[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

                if box_tree_patches.sum() > 0:
                    instance_id = self.get_next_instance_id()
                    instance_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1][box_tree_patches] = instance_id

                    # Store instance metadata
                    self.active_instances[instance_id] = {
                        'bbox': (patch_x1, patch_y1, patch_x2, patch_y2),
                        'patch_count': box_tree_patches.sum().item(),
                        'first_seen': self.frame_count
                    }

                    print(
                        f"  Created tree instance {instance_id}: bbox=({patch_x1},{patch_y1})->({patch_x2},{patch_y2}), "
                        f"{box_tree_patches.sum().item()} patches")
                    instance_count += 1

        # Tree patches not assigned to any instance remain as background (already 0)

        # Since we start with background=0 and instances=1,2,3..., classes should be contiguous
        unique_classes = instance_mask.unique()

        # Create identity mapping since classes should already be contiguous
        class_mapping = {k.item(): k.item() for k in unique_classes}

        print(f"Created {instance_count} tree instances")
        print(f"Active instances: {list(self.active_instances.keys())}")
        print(f"Final unique classes: {unique_classes.cpu().numpy()}")
        print(f"Class mapping: {class_mapping}")

        # Debug: Check if classes are contiguous starting from 0
        sorted_classes = sorted(unique_classes.cpu().numpy())
        expected_classes = list(range(len(sorted_classes)))
        print(f"Checking contiguity: expected {expected_classes}, got {sorted_classes}")

        if sorted_classes != expected_classes:
            print(f"ERROR: Classes are not contiguous! This will cause F.one_hot to fail.")
            print(f"Expected: {expected_classes}")
            print(f"Got: {sorted_classes}")

            # Create proper contiguous mapping
            class_mapping = {k.item(): i for i, k in enumerate(unique_classes)}

            # Apply remapping to ensure contiguous classes 0, 1, 2, ...
            remapped_mask = torch.zeros_like(instance_mask)
            for orig_id, new_id in class_mapping.items():
                remapped_mask[instance_mask == orig_id] = new_id

            # Update active_instances keys to use remapped IDs
            new_active_instances = {}
            for orig_id, metadata in self.active_instances.items():
                if orig_id in class_mapping:
                    new_id = class_mapping[orig_id]
                    new_active_instances[new_id] = metadata
                    print(f"  Remapped active instance {orig_id} -> {new_id}")
            self.active_instances = new_active_instances

            print(f"Applied remapping: {class_mapping}")
            print(f"Updated active instances: {list(self.active_instances.keys())}")
            return remapped_mask, class_mapping

        self.frame_count += 1
        return instance_mask, class_mapping

    def assign_instances_from_predictions(
            self,
            semantic_mask: torch.Tensor,
            predicted_probs: torch.Tensor,
            detections: List[Dict],
            patch_size: int,
            target_image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Convert semantic segmentation to instance segmentation using predicted probabilities.

        Process:
        1. For each detection box, find tree patches
        2. Check most common predicted instance in those patches
        3. If background -> create new instance
        4. If existing instance -> assign that instance

        Args:
            semantic_mask: [H_patches, W_patches] semantic segmentation
            predicted_probs: [H_patches, W_patches, M] predicted probabilities
            detections: List of detection boxes
            patch_size: Size of patches
            target_image_size: (width, height) of target image

        Returns:
            Tuple of (instance_mask, class_mapping)
        """
        print(f"=== ASSIGNING TREE INSTANCES (Frame {self.frame_count}) ===")

        h_patches, w_patches = semantic_mask.shape
        instance_mask = torch.full_like(semantic_mask, self.background_class_id)

        # Find tree patches in semantic segmentation
        tree_patches = (semantic_mask == self.tree_class_id)
        print(f"Found {tree_patches.sum().item()} tree patches in semantic mask")

        # Get predicted instance labels (argmax of probabilities)
        predicted_instances = predicted_probs.argmax(dim=-1)  # [H_patches, W_patches]

        # Process each detection box
        assigned_count = 0
        new_instances = []

        for detection in detections:
            if detection.get('name') == 'trunk':  # Filter for tree detections
                box = detection['box']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                # Convert to patch coordinates
                patch_x1 = max(0, min(int(x1 / patch_size), w_patches - 1))
                patch_y1 = max(0, min(int(y1 / patch_size), h_patches - 1))
                patch_x2 = max(0, min(int(x2 / patch_size), w_patches - 1))
                patch_y2 = max(0, min(int(y2 / patch_size), h_patches - 1))

                # Find tree patches in this box
                box_tree_patches = tree_patches[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

                if box_tree_patches.sum() > 0:
                    # Get predicted instances for tree patches in this box
                    box_predictions = predicted_instances[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]
                    tree_predictions = box_predictions[box_tree_patches]

                    # Find unique predictions and their counts
                    unique_preds, counts = torch.unique(tree_predictions, return_counts=True)

                    # Filter out background predictions
                    non_bg_mask = unique_preds != self.background_class_id
                    non_bg_preds = unique_preds[non_bg_mask]
                    non_bg_counts = counts[non_bg_mask]

                    total_non_bg_votes = non_bg_counts.sum().item()

                    print(
                        f"  Voting analysis: {total_non_bg_votes} non-background votes out of {len(tree_predictions)} tree patches")
                    for pred, count in zip(non_bg_preds, non_bg_counts):
                        percentage = (count.item() / total_non_bg_votes) * 100 if total_non_bg_votes > 0 else 0
                        print(f"    Class {pred.item()}: {count.item()} votes ({percentage:.1f}%)")

                    # Check if any non-background prediction has >50% of non-background votes
                    assigned_instance = None
                    if total_non_bg_votes > 0:
                        for pred, count in zip(non_bg_preds, non_bg_counts):
                            percentage = count.item() / total_non_bg_votes
                            if percentage > 0.5 and pred.item() in self.active_instances:
                                assigned_instance = pred.item()
                                print(
                                    f"  Assigning to existing instance {assigned_instance} ({percentage:.1%} of non-bg votes)")
                                break

                    if assigned_instance is None:
                        # Create new instance (no strong majority or non-existent instance)
                        assigned_instance = self.get_next_instance_id()
                        self.active_instances[assigned_instance] = {
                            'bbox': (patch_x1, patch_y1, patch_x2, patch_y2),
                            'patch_count': box_tree_patches.sum().item(),
                            'first_seen': self.frame_count
                        }
                        new_instances.append(assigned_instance)
                        if total_non_bg_votes == 0:
                            print(f"  Created new tree instance {assigned_instance} (no non-background votes)")
                        else:
                            print(
                                f"  Created new tree instance {assigned_instance} (no class >50% or not in active instances)")

                    # Assign instance to tree patches in this box
                    instance_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1][box_tree_patches] = assigned_instance
                    assigned_count += 1

        # No need for complex remapping - instance IDs are already clean
        # Just create a simple identity mapping for compatibility with existing code
        unique_classes = instance_mask.unique()
        class_mapping = {k.item(): k.item() for k in unique_classes}

        print(f"Assigned {assigned_count} trees to instances")
        print(f"New instances created: {new_instances}")
        print(f"Active instances: {list(self.active_instances.keys())}")
        print(f"Final unique classes: {unique_classes.cpu().numpy()}")

        self.frame_count += 1
        return instance_mask, class_mapping


@torch.compile(disable=True)
def propogate_context_masked_probs(
        current_features: Tensor,  # [h", w", D], where h=h", w=w", and " stands for current
        context_features: Tensor,  # [t, h, w, D]
        context_probs: Tensor,  # [t, h, w, M]
        neighborhood_mask: Tensor,  # [h", w", h, w]
        topk: int,
        temperature: float,
) -> Tensor:
    """
    Core label propagation function adapted from segmentation_tracking.ipynb.

    For each patch of the current frame:
    - Compute cosine similarity with all context patches
    - Restrict focus to local neighborhood and select top-k most similar patches
    - Compute weighted average of mask probabilities to get prediction
    """
    t, h, w, M = context_probs.shape

    # Compute similarity current -> context
    dot = torch.einsum(
        "ijd, tuvd -> ijtuv",
        current_features,  # [h", w", D]
        context_features,  # [t, h, w, D]
    )  # [h", w", t, h, w]

    # Restrict focus to local neighborhood
    dot = torch.where(
        neighborhood_mask[:, :, None, :, :],  # [h", w", 1, h, w]
        dot,  # [h", w", t, h, w]
        -torch.inf,
    )

    # Select top-k patches inside the neighborhood
    dot = dot.flatten(2, -1).flatten(0, 1)  # [h"w", thw]
    k_th_largest = torch.topk(dot, dim=1, k=topk).values  # [h"w", k]
    dot = torch.where(
        dot >= k_th_largest[:, -1:],  # [h"w", thw]
        dot,  # [h"w", thw]
        -torch.inf,
    )

    # Propagate probabilities from context to current frame
    weights = F.softmax(dot / temperature, dim=1)  # [h"w", thw]
    current_probs = torch.mm(
        weights,  # [h"w", thw]
        context_probs.flatten(0, 2),  # [thw, M]
    )  # [h"w", M]

    # Propagated probs should already sum to 1, but just in case
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True)  # [h"w", M]

    return current_probs.unflatten(0, (h, w))  # [h", w", M]


@functools.lru_cache()
def make_neighborhood_mask(h: int, w: int, size: float, shape: str) -> Tensor:
    """
    Create neighborhood mask for spatial locality constraints.
    Adapted from segmentation_tracking.ipynb.
    """
    ij = torch.stack(
        torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device="cuda"),
            torch.arange(w, dtype=torch.float32, device="cuda"),
            indexing="ij",
        ),
        dim=-1,
    )  # [h, w, 2]
    if shape == "circle":
        norm = torch.linalg.vector_norm(
            ij[:, :, None, None, :] - ij[None, None, :, :, :],  # [h", w", h, w, 2]
            ord=2,
            dim=-1,
        )  # [h", w", h, w]
    elif shape == "square":
        norm = torch.linalg.vector_norm(
            ij[:, :, None, None, :] - ij[None, None, :, :, :],  # [h", w", h, w, 2]
            ord=torch.inf,
            dim=-1,
        )  # [h", w", h, w]
    else:
        raise ValueError(f"Invalid {shape=}")
    mask = norm <= size  # [h", w", h, w] bool, True inside, False outside
    return mask


def normalize_probs(
        probs: Tensor,  # [B, M, H', W']
) -> Tensor:
    """
    Normalize and clean probability distributions.
    Adapted from segmentation_tracking.ipynb.
    """
    vmin = probs.flatten(2, 3).min(dim=2).values  # [B, M]
    vmax = probs.flatten(2, 3).max(dim=2).values  # [B, M]
    probs = (probs - vmin[:, :, None, None]) / (vmax[:, :, None, None] - vmin[:, :, None, None])
    probs = torch.nan_to_num(probs, nan=0)
    return probs  # [B, M, H', W']


def extract_frames_from_cache(
        cache_data: Dict[str, Any],
        frame_indices: Optional[Union[List[int], np.ndarray]] = None,
        require_segmentation: bool = True,
        require_detections: bool = False
) -> List[VideoFrameData]:
    """
    Extract frame data from cache similar to run_tracking_processor.py.

    Args:
        cache_data: Loaded cache data from tracking_data_processor.py
        frame_indices: Specific frame indices to extract (default: auto-select)
        require_segmentation: Whether frames must have segmentation data
        require_detections: Whether frames must have detection data

    Returns:
        List of VideoFrameData objects
    """
    print("=== EXTRACTING FRAMES FROM CACHE ===")

    # Select frames to process
    if frame_indices is None:
        # Auto-select frames with valid data (similar to run_tracking_processor.py)
        valid_frames = []
        for i, frame in enumerate(cache_data['frames']):
            has_detections = len(frame.get('detections', [])) > 0
            has_segmentation = frame.get('segmentation') is not None

            # Apply requirements
            if require_segmentation and not has_segmentation:
                continue
            if require_detections and not has_detections:
                continue

            valid_frames.append(i)

        if len(valid_frames) < 2:
            raise ValueError(f"Need at least 2 frames with valid data, found {len(valid_frames)}")

        # For video tracking, we want a contiguous sequence
        # Take a reasonable subset if too many frames
        if len(valid_frames) > 50:
            # Take first 50 frames for manageable processing
            frame_indices = valid_frames[:50]
        else:
            frame_indices = valid_frames

    print(f"Selected {len(frame_indices)} frames: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")

    # Extract frame data
    video_frames = []
    for i, frame_idx in enumerate(frame_indices):
        frame_data = cache_data['frames'][int(frame_idx)]
        video_frame = VideoFrameData(frame_data, frame_idx)
        video_frames.append(video_frame)

        # if i < 5:  # Print details for first few frames
        #     print(f"  Frame {frame_idx}: {video_frame.filename}, "
        #           f"features: {video_frame.features.shape}, "
        #           f"segmentation: {video_frame.has_segmentation}, "
        #           f"detections: {video_frame.has_detections}")

    return video_frames

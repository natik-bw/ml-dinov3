import functools
from typing import Dict, Any, List, Tuple, Optional, Union, Set

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


class TrunkFrameInformation:
    """Stores information for a single frame in the context window."""

    def __init__(self, trunk_instance_mask: torch.Tensor, feature_map: torch.Tensor, frame_idx: int,
                 context_based_mask: Optional[torch.Tensor] = None,
                 bbox_tracking_info: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            trunk_instance_mask: [H_patches, W_patches] final trunk instance segmentation mask
            feature_map: [H_patches, W_patches, feature_dim] feature map
            frame_idx: Frame index in the video sequence
            context_based_mask: [H_patches, W_patches] raw context-based inference result (optional)
            bbox_tracking_info: List of dicts with bbox_index, tracking_index, and bbox coordinates (optional)
        """
        self.trunk_instance_mask = trunk_instance_mask
        self.feature_map = feature_map
        self.frame_idx = frame_idx
        self.context_based_mask = context_based_mask
        self.bbox_tracking_info = bbox_tracking_info or []


class TrunkVideoTracker:
    """
    Modular video tracker for trunk instances across frames.

    Maintains context frames and trunk indexes for temporal tracking.
    """

    def __init__(self, tree_class_id: int = 2, trunk_od_class_name: str = 'trunk', context_window_size: int = 5,
                 most_likely_id_min_ratio: float = 0.5, create_united_visualization: bool = False):
        """
        Args:
            tree_class_id: Class ID for trunks in semantic segmentation
            trunk_od_class_name: Object detection class name for trunks
            context_window_size: Maximum number of frames to keep in context
            most_likely_id_min_ratio: Minimum ratio for majority trunk ID assignment (default: 0.5)
            create_united_visualization: If True, create united horizontal visualization; if False, save separately
        """
        self.segmentation_class_id = tree_class_id
        self.od_class_name = trunk_od_class_name

        self.context_window_size = context_window_size
        self.most_likely_id_min_ratio = most_likely_id_min_ratio
        self.create_united_visualization = create_united_visualization

        # History storage
        self.context_frames: List[TrunkFrameInformation] = []
        self.trunk_indexes: Set[int] = set()  # All trunk indexes ever assigned

        # Internal state
        self.next_trunk_id = 1  # Start from 1, background is 0
        self.frame_count = 0
        self.background_id = 0

    def get_next_trunk_id(self) -> int:
        """Get the next available trunk ID."""
        trunk_id = self.next_trunk_id
        self.next_trunk_id += 1
        self.trunk_indexes.add(trunk_id)
        return trunk_id

    def create_initial_trunk_instances(
            self,
            segmentation_mask: torch.Tensor,
            detections: List[Dict],
            patch_size: int,
            target_image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Create initial trunk instances from semantic segmentation and detections.
        Similar to TreesTracker::create_initial_instances but adapted for trunk terminology.

        Args:
            segmentation_mask: [H_patches, W_patches] semantic segmentation
            detections: List of detection boxes
            patch_size: Size of patches
            target_image_size: (width, height) of target image

        Returns:
            instance_mask: [H_patches, W_patches] trunk instance segmentation
        """
        print(f"=== CREATING INITIAL TRUNK INSTANCES (Frame {self.frame_count}) ===")

        h_patches, w_patches = segmentation_mask.shape

        # Initialize instance mask with all background (0)
        instance_mask = torch.zeros_like(segmentation_mask)

        # Find trunk patches
        tree_patches = (segmentation_mask == 1)
        trunk_bboxes = [detection['box'] for detection in detections if
                        'box' in detection and detection['name'] == self.od_class_name]

        if not tree_patches.any():
            print("No tree patches found in segmentation")
            return instance_mask

        print(f"Found {tree_patches.sum().item()} tree patches")
        print(f"Processing {len(detections)} trunk detections")

        instance_count = 0

        # Process each detection box
        for bbox in trunk_bboxes:
            patch_x1, patch_y1, patch_x2, patch_y2 = self.bbox_to_patch_coords(
                bbox, patch_size, w_patches, h_patches
            )
            box_trunk_patches = tree_patches[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

            if box_trunk_patches.sum() > 0:
                instance_id = self.get_next_trunk_id()
                instance_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1][box_trunk_patches] = instance_id

                print(f"  Created trunk instance {instance_id}: bbox=({patch_x1},{patch_y1})->({patch_x2},{patch_y2}), "
                      f"{box_trunk_patches.sum().item()} patches")
                instance_count += 1

        print(f"Created {instance_count} trunk instances")
        print(f"Active trunk indexes: {sorted(self.trunk_indexes)}")

        return instance_mask

    def create_temp_index_mapping(self) -> Dict[int, int]:
        """
        Create mapping from trunk indexes to continuous temporary indexes.

        Returns:
            Dict mapping original trunk_id -> temp_id (0=background, 1,2,3... for trunks)
        """
        # Collect all trunk indexes from context frames
        active_trunk_ids = set()
        for frame_info in self.context_frames:
            unique_ids = frame_info.trunk_instance_mask.unique()
            # Filter out background (0)
            active_trunk_ids.update([id.item() for id in unique_ids if id.item() != 0])

        # Create continuous mapping: 0=background, 1,2,3... for active trunks
        temp_mapping = {0: 0}  # Background stays 0
        for i, trunk_id in enumerate(sorted(active_trunk_ids), start=1):
            temp_mapping[trunk_id] = i

        print(f"Created temp index mapping: {temp_mapping}")
        return temp_mapping

    def convert_instance_masks_to_one_hot(
            self,
            continuous_mapping_dict: Dict[int, int],
            h_patches: int,
            w_patches: int
    ) -> torch.Tensor:
        """
        Convert instance masks from context window to one-hot probability matrices.

        Args:
            continuous_mapping_dict: Mapping from original trunk_id to continuous temp_id
        Returns:
            context_probs: [num_context_frames, h_patches, w_patches, M] one-hot matrices
                          where M is the number of active trunk instances (including background)
        """
        num_classes = len(continuous_mapping_dict)  # M = number of active trunk instances + background
        num_context_frames = len(self.context_frames)

        print(f"Converting {num_context_frames} context frames to one-hot with {num_classes} classes")

        # Initialize one-hot tensor
        context_probs = torch.zeros(num_context_frames, h_patches, w_patches, num_classes)

        # Convert each frame's instance mask to one-hot
        for frame_idx, frame_info in enumerate(self.context_frames):
            instance_mask = frame_info.trunk_instance_mask  # [h_patches, w_patches]
            print(f"  Processing frame {frame_idx}: instance_mask shape = {instance_mask.shape}")

            # Create one-hot encoding for this frame
            frame_one_hot = torch.zeros(h_patches, w_patches, num_classes)
            print(f"  Created frame_one_hot with shape: {frame_one_hot.shape}")

            # For each original trunk ID, assign to corresponding temp class
            for orig_id, temp_id in continuous_mapping_dict.items():
                mask = (instance_mask == orig_id)  # Boolean mask for this trunk ID
                num_pixels = mask.sum().item()
                if num_pixels > 0:
                    print(f"    Assigning {num_pixels} pixels from orig_id={orig_id} to temp_id={temp_id}")
                frame_one_hot[:, :, temp_id][mask] = 1.0

            # Ensure each pixel has exactly one class (should be guaranteed by construction)
            pixel_sums = frame_one_hot.sum(dim=-1)  # [h_patches, w_patches]
            if not torch.allclose(pixel_sums, torch.ones_like(pixel_sums)):
                raise ValueError(f"Frame {frame_idx} one-hot encoding not properly normalized")

            context_probs[frame_idx] = frame_one_hot

            # Debug info
            unique_orig_ids = instance_mask.unique().tolist()
            unique_temp_ids = torch.argmax(frame_one_hot, dim=-1).unique().tolist()
            print(f"  Frame {frame_idx}: orig_ids={unique_orig_ids} -> temp_ids={unique_temp_ids}")

        return context_probs

    def process_frame(
            self,
            video_frame: 'VideoFrameData',
            patch_size: int,
            target_image_size: Tuple[int, int]
    ) -> TrunkFrameInformation:
        """
        Process a single frame and return complete frame information.

        Args:
            video_frame: VideoFrameData object containing frame information
            patch_size: Size of patches
            target_image_size: (width, height) of target image

        Returns:
            TrunkFrameInformation object containing:
                - trunk_instance_mask: [H_patches, W_patches] final trunk instance segmentation
                - context_based_mask: [H_patches, W_patches] raw context-based inference result
                - bbox_tracking_info: List of dicts with bbox_index, tracking_index, and bbox coordinates
                - feature_map: [H_patches, W_patches, feature_dim] features for this frame
                - frame_idx: Frame index in sequence
        """
        print(f"\n=== PROCESSING FRAME {self.frame_count} ===")

        # Extract features and segmentation
        features = video_frame.features.float()  # [H_patches, W_patches, feature_dim]

        if video_frame.segmentation is None:
            raise ValueError("Video frame segmentation is None")

        segmentation_array = np.array(video_frame.segmentation)
        binary_semantic_segmentation = torch.from_numpy(
            self.create_binary_semantic_segmentation(segmentation_array, self.segmentation_class_id)).long()

        # Determine processing approach based on context availability
        is_first_frame = len(self.context_frames) == 0
        
        if is_first_frame:
            print("Empty context window - creating initial trunk instances")
            # For first frame: use simple instance creation
            trunk_instance_mask = self.create_initial_trunk_instances(
                binary_semantic_segmentation,
                video_frame.detections_scaled,
                patch_size,
                target_image_size
            )
            # For first frame, context_based_mask equals final mask
            context_based_mask = trunk_instance_mask.clone()
            bbox_tracking_info = self._create_bbox_tracking_info_first_frame(
                video_frame.detections_scaled
            )
        else:
            print(f"Context window has {len(self.context_frames)} frames - performing tracking")
            # For subsequent frames: use context-based tracking + aggregation
            temp_mapping = self.create_temp_index_mapping()
            context_based_mask = self.compute_context_based_instances(
                features,
                binary_semantic_segmentation,
                temp_mapping
            )
            # Get both trunk_instance_mask and bbox_tracking_info from aggregation
            trunk_instance_mask, bbox_tracking_info = self.aggregate_frame_instances(
                context_based_mask,
                video_frame.detections_scaled,
                binary_semantic_segmentation,
                patch_size,
                target_image_size
            )

        # Add frame to context (unified for both cases)
        frame_info = TrunkFrameInformation(
            trunk_instance_mask,
            features,
            self.frame_count,
            context_based_mask=context_based_mask,
            bbox_tracking_info=bbox_tracking_info
        )
        self.context_frames.append(frame_info)

        # Maintain context window size
        if len(self.context_frames) > self.context_window_size:
            self.context_frames.pop(0)

        self.frame_count += 1
        return frame_info

    def _create_bbox_tracking_info_first_frame(
            self,
            detections_scaled: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Create bbox tracking info for first frame with simple sequential assignment.
        
        Args:
            detections_scaled: List of detection dictionaries
            
        Returns:
            List of bbox tracking info dictionaries
        """
        bbox_tracking_info = []
        
        # First frame: simple sequential assignment
        for bbox_idx, detection in enumerate(detections_scaled):
            if 'box' in detection:
                bbox_info = {
                    'bbox_index': bbox_idx,
                    'tracking_index': bbox_idx + 1,  # Simple assignment for first frame
                    'bbox': detection['box']
                }
                bbox_tracking_info.append(bbox_info)
        
        return bbox_tracking_info

    def create_binary_semantic_segmentation(self, full_semantic_segmentation: np.ndarray, class_id: int):
        binary_semantic_segmentation = np.zeros_like(full_semantic_segmentation)
        binary_semantic_segmentation[full_semantic_segmentation == class_id] = 1
        return binary_semantic_segmentation

    def bbox_to_patch_coords(
            self,
            bbox: Dict[str, float],
            patch_size: int,
            w_patches: int,
            h_patches: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert bounding box coordinates to patch coordinates.

        Args:
            bbox: Bounding box with 'x1', 'y1', 'x2', 'y2' keys
            patch_size: Size of patches in pixels
            w_patches: Number of patches in width
            h_patches: Number of patches in height

        Returns:
            Tuple of (patch_x1, patch_y1, patch_x2, patch_y2) in patch coordinates
        """
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        patch_x1 = max(0, min(int(x1 / patch_size), w_patches - 1))
        patch_y1 = max(0, min(int(y1 / patch_size), h_patches - 1))
        patch_x2 = max(0, min(int(x2 / patch_size), w_patches - 1))
        patch_y2 = max(0, min(int(y2 / patch_size), h_patches - 1))

        return patch_x1, patch_y1, patch_x2, patch_y2

    def compute_context_based_instances(
            self,
            current_features: torch.Tensor,  # [H_patches, W_patches, feature_dim]
            current_segmentation: torch.Tensor,  # [H_patches, W_patches] binary segmentation
            temp_mapping: Dict[int, int],
    ) -> torch.Tensor:
        """
        Compute trunk instance segmentation using context-based inference.

        Steps:
        1. Create context features and probabilities from context frames
        2. Use neighborhood mask for spatial constraints
        3. Compute similarity between current frame and context frames
        4. Assign labels based on most similar patches

        Args:
            current_features: Features of current frame
            current_segmentation: Binary segmentation of current frame (1=tree, 0=background)
            temp_mapping: Mapping from trunk_id to continuous temp_id
            patch_size: Size of patches
            target_image_size: Target image size

        Returns:
            trunk_instance_mask: [H_patches, W_patches] with trunk instance assignments
        """
        print(f"  Computing context-based inference with {len(self.context_frames)} context frames")

        h_patches, w_patches, feature_dim = current_features.shape
        # Initialize result mask
        trunk_instance_mask = torch.zeros_like(current_segmentation)

        if len(self.context_frames) == 0:
            raise ValueError("This function shouldn't be called without context frames for now")

        # Step 1: Prepare context features and probabilities
        context_features = self.prepare_context_features(h_patches, w_patches, feature_dim)
        context_probs = self.convert_instance_masks_to_one_hot(temp_mapping, h_patches, w_patches)

        if context_features is None:
            raise ValueError("No valid context features. This shouldn't happen")

        print(f"  Context features shape: {context_features.shape}")
        print(f"  Context probabilities shape: {context_probs.shape}")

        # Step 2: Create neighborhood mask for spatial constraints
        neighborhood_size = 12.0  # Adjust as needed
        neighborhood_mask = make_neighborhood_mask(h_patches, w_patches, neighborhood_size, "circle")

        # Move to GPU if available
        device = current_features.device
        context_features = context_features.to(device)
        context_probs = context_probs.to(device)
        neighborhood_mask = neighborhood_mask.to(device)

        # Step 3: Compute similarity and propagate labels
        topk = min(10, context_features.shape[0] * h_patches * w_patches // 4)  # Adaptive topk
        temperature = 0.1

        print(f"  Using topk={topk}, temperature={temperature}")

        # Use the existing propagation function
        predicted_probs = propogate_context_masked_probs(
            current_features=current_features,
            context_features=context_features,
            context_probs=context_probs,
            neighborhood_mask=neighborhood_mask,
            topk=topk,
            temperature=temperature
        )  # [h_patches, w_patches, num_classes]

        # Find most likely class for each patch
        predicted_classes = torch.argmax(predicted_probs, dim=-1)  # [h_patches, w_patches]

        print(f"  Predicted temp classes: {predicted_classes.unique().tolist()}")

        # Only assign to patches that are trees in the current segmentation
        tree_mask = (current_segmentation == 1)

        reverse_temp_mapping = {v: k for k, v in temp_mapping.items()}
        print(f"  Reverse temp mapping: {reverse_temp_mapping}")

        for orig_id, temp_id in temp_mapping.items():
            mask = (predicted_classes == temp_id)
            trunk_instance_mask[mask] = orig_id

            num_assigned_this_id = mask.sum().item()
            if num_assigned_this_id > 0:
                print(
                    f"    Assigned {num_assigned_this_id} patches to original trunk ID {orig_id} (from temp ID {temp_id})")

        # Count assignments
        num_assigned = (trunk_instance_mask > 0).sum().item()
        num_tree_patches = tree_mask.sum().item()
        unique_trunk_ids = trunk_instance_mask.unique().tolist()
        if self.background_id in unique_trunk_ids:
            unique_trunk_ids.remove(self.background_id)

        print(f"  Final assignment: {num_assigned}/{num_tree_patches} tree patches assigned to trunk instances")
        print(f"  Unique trunk IDs in result: {unique_trunk_ids}")

        return trunk_instance_mask

    def aggregate_frame_instances(
            self,
            context_based_mask: torch.Tensor,  # [H_patches, W_patches] result from context-based inference
            trunk_detections: List[Dict],  # List of trunk bounding boxes
            semantic_segmentation: torch.Tensor,  # [H_patches, W_patches] binary semantic mask
            patch_size: int,
            target_image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Aggregate trunk instances using bounding box analysis and conflict resolution.

        Steps:
        1. For each bounding box, find majority trunk index (>50% threshold)
        2. Resolve conflicts when multiple boxes have same trunk index
        3. Create final segmentation mask

        Args:
            context_based_mask: Raw result from context-based inference
            trunk_detections: List of trunk detection bounding boxes
            semantic_segmentation: Binary semantic segmentation (1=tree, 0=background)
            patch_size: Size of patches
            target_image_size: Target image size

        Returns:
            final_trunk_mask: [H_patches, W_patches] final trunk instance assignments
        """
        print(f"=== AGGREGATING FRAME INSTANCES ===")
        print(f"Processing {len(trunk_detections)} trunk bounding boxes")

        h_patches, w_patches = context_based_mask.shape

        # Step 1: Analyze each bounding box for majority trunk index
        bbox_trunk_assignments: Dict[int, int] = {}  # bbox_idx -> assigned_trunk_id

        for detection_index, detection in enumerate(trunk_detections):
            # Convert to patch coordinates
            patch_x1, patch_y1, patch_x2, patch_y2 = self.bbox_to_patch_coords(
                detection['box'], patch_size, w_patches, h_patches
            )

            # Extract patches in this bounding box
            bbox_mask = context_based_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]
            bbox_semantic = semantic_segmentation[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

            # Get non-background tree patches
            non_bg_tree_mask = (bbox_mask != self.background_id) & (bbox_semantic == 1)
            trunk_ids_in_box = bbox_mask[non_bg_tree_mask]

            # Find majority (handles empty case naturally)
            if len(trunk_ids_in_box) > 0:
                unique_ids, counts = torch.unique(trunk_ids_in_box, return_counts=True)
                max_count_idx = torch.argmax(counts)
                most_frequent_id = unique_ids[max_count_idx].item()
                max_count = counts[max_count_idx].item()
                percentage = max_count / len(trunk_ids_in_box)
            else:
                percentage = 0.0

            if percentage >= self.most_likely_id_min_ratio and percentage > 0:
                tracking_index = most_frequent_id
                print(
                    f"  BBox {detection_index}: Majority trunk ID {most_frequent_id} ({percentage:.2%} of {len(trunk_ids_in_box)} patches)")
            else:
                tracking_index = self.get_next_trunk_id()
                reason = "No valid tree patches" if percentage == 0 else f"No majority found (best: {percentage:.2%})"
                print(
                    f"  BBox {detection_index}: {reason}, assigning new trunk ID {tracking_index}")
            bbox_trunk_assignments[detection_index] = tracking_index

        # Step 2: Resolve conflicts (multiple boxes with same trunk ID)
        trunk_id_to_bboxes = {}  # type: Dict[int, List[int]]
        for detection_index, trunk_id in bbox_trunk_assignments.items():
            if trunk_id not in trunk_id_to_bboxes:
                trunk_id_to_bboxes[trunk_id] = []
            trunk_id_to_bboxes[trunk_id].append(detection_index)

        # Resolve conflicts by choosing most probable box and reassigning others
        for trunk_id, bbox_indices in trunk_id_to_bboxes.items():  # TODO: Duplicate and could be simplified
            if len(bbox_indices) > 1:
                print(f"  Conflict: Trunk ID {trunk_id} assigned to {len(bbox_indices)} boxes: {bbox_indices}")

                # Calculate confidence scores for each conflicting box
                bbox_scores = []
                for detection_index in bbox_indices:
                    detection = trunk_detections[detection_index]

                    # Convert to patch coordinates
                    patch_x1, patch_y1, patch_x2, patch_y2 = self.bbox_to_patch_coords(
                        detection['box'], patch_size, w_patches, h_patches
                    )

                    # Count patches assigned to this trunk ID
                    bbox_mask = context_based_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]
                    bbox_semantic = semantic_segmentation[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1]

                    # Score based on number of patches with this trunk ID
                    target_patches = (bbox_mask == trunk_id) & (bbox_semantic == 1)
                    score = target_patches.sum().item()
                    bbox_scores.append((detection_index, score))

                # Sort by score (highest first) and keep the best one
                bbox_scores.sort(key=lambda x: x[1], reverse=True)
                best_bbox_idx = bbox_scores[0][0]

                print(f"    Keeping trunk ID {trunk_id} for bbox {best_bbox_idx} (score: {bbox_scores[0][1]})")

                # Reassign new trunk IDs to the rest
                for detection_index, score in bbox_scores[1:]:
                    new_tracking_id = self.get_next_trunk_id()
                    bbox_trunk_assignments[detection_index] = new_tracking_id
                    print(
                        f"    Reassigning bbox {detection_index} to new trunk ID {new_tracking_id} (score was: {score})")

        # Step 3: Create final segmentation mask
        final_trunk_mask = torch.zeros_like(context_based_mask)

        for detection_index, assigned_trunk_id in bbox_trunk_assignments.items():
            detection = trunk_detections[detection_index]

            # Convert to patch coordinates
            patch_x1, patch_y1, patch_x2, patch_y2 = self.bbox_to_patch_coords(
                detection['box'], patch_size, w_patches, h_patches
            )

            # Assign trunk ID to tree patches in this bounding box
            tree_mask_in_box = (semantic_segmentation[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1] == 1)
            final_trunk_mask[patch_y1:patch_y2 + 1, patch_x1:patch_x2 + 1][tree_mask_in_box] = assigned_trunk_id

        # Summary statistics
        num_assigned = (final_trunk_mask > 0).sum().item()
        num_tree_patches = (semantic_segmentation == 1).sum().item()
        unique_trunk_ids = final_trunk_mask.unique().tolist()
        if self.background_id in unique_trunk_ids:
            unique_trunk_ids.remove(self.background_id)

        print(f"Final aggregation: {num_assigned}/{num_tree_patches} tree patches assigned")
        print(f"Unique trunk IDs: {unique_trunk_ids}")
        print(f"Total trunk instances: {len(unique_trunk_ids)}")

        # Prepare bounding box tracking information
        bbox_tracking_info = []
        for detection_index, tracking_index in bbox_trunk_assignments.items():
            bbox_info = {
                'bbox_index': detection_index,
                'tracking_index': tracking_index,
                'bbox': trunk_detections[detection_index]['box']
            }
            bbox_tracking_info.append(bbox_info)

        return final_trunk_mask, bbox_tracking_info

    def prepare_context_features(
            self,
            h_patches: int,
            w_patches: int,
            feature_dim: int
    ) -> Optional[torch.Tensor]:
        """
        Prepare context features from context frames.

        Returns:
            context_features: [num_context_frames, h_patches, w_patches, feature_dim]
        """
        if len(self.context_frames) == 0:
            raise ValueError("This function shouldn't be called without context frames for now")

        context_features = torch.zeros(len(self.context_frames), h_patches, w_patches, feature_dim)
        for i, frame_info in enumerate(self.context_frames):
            context_features[i] = frame_info.feature_map

        print(f"  Prepared context features: {len(self.context_frames)} frames, feature_dim={feature_dim}")
        return context_features

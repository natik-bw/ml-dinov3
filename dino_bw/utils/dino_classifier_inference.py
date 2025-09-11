from pathlib import Path
from dataclasses import dataclass

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA

from dino_bw.utils.dino_embeddings import extract_patch_features


@dataclass
class DinoLRResult:
    features: np.ndarray
    model_img_size: tuple
    h_patches: int
    w_patches: int
    fg_score: np.ndarray
    fg_score_mf: torch.Tensor
    foreground_selection: torch.Tensor
    fg_patches: np.ndarray
    pca: PCA
    projected_image: torch.Tensor
    projected_img_resized: np.ndarray


def regress_dino_embeddings(model, classifier, image, n_layers, image_size, patch_size) -> DinoLRResult:
    """
    Process a single image to extract features, calculate foreground score, and apply PCA.
    """
    features, model_img_size = extract_patch_features(image, model, n_layers, image_size, patch_size)
    h_patches, w_patches = [int(d / patch_size) for d in model_img_size]

    fg_score = classifier.predict_proba(features)[:, 1].reshape(h_patches, w_patches)

    # PCA Calculation
    fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))
    foreground_selection = fg_score_mf.view(-1) > 0.5
    fg_patches = features[foreground_selection]

    pca = PCA(n_components=3, whiten=True)
    pca.fit(fg_patches)

    # Apply PCA projection to image
    projected_image = torch.from_numpy(pca.transform(features.numpy())).view(h_patches, w_patches, 3)
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
    projected_image *= (fg_score_mf.unsqueeze(0) > 0.5)

    projected_img_resized = cv2.resize(projected_image.permute(1, 2, 0).numpy(), dsize=image.size,
                                       interpolation=cv2.INTER_LINEAR)

    return DinoLRResult(
        features=features,
        model_img_size=model_img_size,
        h_patches=h_patches,
        w_patches=w_patches,
        fg_score=fg_score,
        fg_score_mf=fg_score_mf,
        foreground_selection=foreground_selection,
        fg_patches=fg_patches,
        pca=pca,
        projected_image=projected_image,
        projected_img_resized=projected_img_resized
    )


def visualize_results(image, result: DinoLRResult, output_dir: Path | None = None, image_name: str | None = None):
    """
    Visualize the results and optionally save the figures.
    """
    plt.rcParams.update({
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "axes.labelsize": 5,
        "axes.titlesize": 4,
    })

    # Plot main results
    fig1, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=300)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Image, Size {image.size}")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result.fg_score_mf)
    axes[0, 1].set_title(f"Foreground Score, Size {tuple(result.fg_score_mf.shape)}")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(result.projected_img_resized)
    axes[1, 0].set_title(f"PCA Results, Size {tuple(result.projected_img_resized.shape)}")
    axes[1, 0].axis('off')

    axes[1, 1].axis('off')  # Empty subplot

    fig1.subplots_adjust(wspace=0.05, hspace=0.05)

    if output_dir and image_name:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(output_dir / f"{image_name}_results.png")
        plt.close(fig1)
    else:
        plt.show()

    # Create and plot overlay
    fg_score_resized = cv2.resize(result.fg_score, dsize=image.size, interpolation=cv2.INTER_LINEAR)
    fg_score_resized_mf = signal.medfilt2d(fg_score_resized, kernel_size=5)

    overlay_rgba = np.zeros((fg_score_resized_mf.shape[0], fg_score_resized_mf.shape[1], 4), dtype=np.float32)
    overlay_rgba[fg_score_resized_mf >= 0.5] = [0, 1, 0, 0.15]  # Green
    overlay_rgba[fg_score_resized_mf < 0.5] = [1, 0, 0, 0.15]  # Red

    fig2 = plt.figure(figsize=(8, 8), dpi=200)
    plt.imshow(image)
    plt.imshow(overlay_rgba)
    plt.axis('off')
    plt.title("Overlay")

    if output_dir and image_name:
        fig2.savefig(output_dir / f"{image_name}_overlay.png")
        plt.close(fig2)
    else:
        plt.show()

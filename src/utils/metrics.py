import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from scipy.linalg import sqrtm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# Image loader — reads a folder of images from disk ------------------------------------------

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),   # HWC uint8 [0,255] -> CHW float [0,1]
])


def load_images_from_folder(
    folder    : Union[str, Path],
    transform : Optional[object] = None,
    max_images: Optional[int]    = None,
    verbose   : bool              = True,
) -> List[torch.Tensor]:
    
    from PIL import Image  # lazy import — only needed for loading

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not paths:
        raise ValueError(
            f"No supported images found in '{folder}'. "
            f"Supported extensions: {_SUPPORTED_EXTENSIONS}"
        )

    if max_images is not None:
        paths = paths[:max_images]

    tf = transform if transform is not None else _DEFAULT_TRANSFORM

    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        images.append(tf(img))

    if verbose:
        print(f"Loaded {len(images)} images from '{folder}'")

    return images


# InceptionV3 ---------------------------------------------------------------------------

class _InceptionV3(nn.Module):
    """
    Thin wrapper around torchvision InceptionV3 that exposes:
      - pool3 features (2048-d)          used by FID
      - softmax class probabilities      used by IS
    Images must be (B, 3, H, W) float tensors in [0, 1].
    ImageNet normalisation is applied internally.
    """

    _LAYER_NAMES = [
        "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
        "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3", "maxpool2",
        "Mixed_5b", "Mixed_5c", "Mixed_5d",
        "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
        "Mixed_7a", "Mixed_7b", "Mixed_7c",
        "avgpool", "dropout", "fc",
    ]

    def __init__(self):
        super().__init__()
        base = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        base.eval()
        for name in self._LAYER_NAMES:
            setattr(self, name, getattr(base, name))

    @staticmethod
    def _normalise(x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _pool3(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self._normalise(x)
        x = self.Conv2d_1a_3x3(x);  x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x);  x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x);  x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x);  x = self.Mixed_5c(x);  x = self.Mixed_5d(x)
        x = self.Mixed_6a(x);  x = self.Mixed_6b(x);  x = self.Mixed_6c(x)
        x = self.Mixed_6d(x);  x = self.Mixed_6e(x)
        x = self.Mixed_7a(x);  x = self.Mixed_7b(x);  x = self.Mixed_7c(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (N, 2048)

    def features(self, x):
        """Returns 2048-d pool3 features."""
        return self._pool3(x)

    def probs(self, x):
        """Returns softmax probabilities over 1000 ImageNet classes."""
        logits = self.fc(self.dropout(self._pool3(x)))
        return F.softmax(logits, dim=1)


# Module-level singleton so InceptionV3 is loaded only once
_INCEPTION_CACHE: dict = {}


def _get_inception(device):
    key = str(device)
    if key not in _INCEPTION_CACHE:
        m = _InceptionV3().to(device)
        m.eval()
        _INCEPTION_CACHE[key] = m
    return _INCEPTION_CACHE[key]


def _extract(images, inception, mode, batch_size, device):
    """
    Run Inception on a list of (C, H, W) tensors in [0, 1].
    mode = 'features'  ->  pool3 activations  (N, 2048)
    mode = 'probs'     ->  softmax output     (N, 1000)
    Returns numpy array (N, D).
    """
    imgs    = torch.stack(images, 0)
    collect = []
    for start in range(0, len(imgs), batch_size):
        batch = imgs[start: start + batch_size].to(device)
        with torch.no_grad():
            out = inception.features(batch) if mode == "features" else inception.probs(batch)
        collect.append(out.cpu().numpy())
    return np.concatenate(collect, axis=0)



# Inception Score (IS) ------------------------------------------------------

def inception_score(images, batch_size=32, splits=10, device="cpu"):

    inception = _get_inception(device)
    probs     = _extract(images, inception, "probs", batch_size, device)  # (N, 1000)

    N     = len(probs)
    chunk = N // splits
    scores = []

    for k in range(splits):
        part = probs[k * chunk: (k + 1) * chunk]       # (chunk, 1000)
        p_y  = part.mean(axis=0, keepdims=True)         # marginal  (1, 1000)
        # KL( p(y|x) || p(y) ) = sum_y p(y|x) * [ log p(y|x) - log p(y) ]
        kl   = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))

    return float(np.mean(scores)), float(np.std(scores))


# Frechet Inception Distance (FID) ------------------------------------------

def frechet_inception_distance(real_images, fake_images,
                                batch_size=32, device="cpu"):

    inception = _get_inception(device)

    feats_r = _extract(real_images, inception, "features", batch_size, device)
    feats_f = _extract(fake_images, inception, "features", batch_size, device)

    mu_r,    sigma_r = feats_r.mean(axis=0), np.cov(feats_r, rowvar=False)
    mu_f,    sigma_f = feats_f.mean(axis=0), np.cov(feats_f, rowvar=False)

    diff      = mu_r - mu_f

    
    sqrt_prod, _ = sqrtm(sigma_r @ sigma_f, disp=False)

    # Discard negligible imaginary parts that arise from numerical noise
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * sqrt_prod))
    return fid


# Structural Similarity Index (SSIM) ----------------------------------------

def _gaussian_kernel(size=11, sigma=1.5):
    """
    2-D Gaussian kernel as a (1, 1, size, size) torch tensor.
    Built as an outer product of two 1-D Gaussian kernels (separable),
    so no external library is needed.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    k1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    k1d    = k1d / k1d.sum()
    k2d    = k1d.unsqueeze(1) * k1d.unsqueeze(0)   # outer product
    return k2d.unsqueeze(0).unsqueeze(0)            # (1, 1, size, size)


def ssim(img1, img2, window_size=11, sigma=1.5, K1=0.01, K2=0.03, L=1.0):

   
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    _, C, _, _ = img1.shape
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Depthwise convolution: one kernel per channel, channels do not mix
    kernel = _gaussian_kernel(window_size, sigma).to(img1.device)
    kernel = kernel.expand(C, 1, window_size, window_size)  # (C, 1, size, size)
    pad    = window_size // 2

    def lconv(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1    = lconv(img1)
    mu2    = lconv(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    # Var(X)   = E[X^2] - E[X]^2   (local, inside Gaussian window)
    # Cov(X,Y) = E[XY]  - E[X]E[Y]
    sigma1_sq = lconv(img1 * img1) - mu1_sq
    sigma2_sq = lconv(img2 * img2) - mu2_sq
    sigma12   = lconv(img1 * img2) - mu12

    numerator   = (2.0 * mu12   + C1) * (2.0 * sigma12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return float((numerator / denominator).mean())


def mean_ssim(real_images, fake_images, **kwargs):
    if len(real_images) != len(fake_images):
        raise ValueError("real_images and fake_images must have the same length")
    scores = [ssim(r, f, **kwargs) for r, f in zip(real_images, fake_images)]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# EvaluationResult — structured container for metric outputs
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    model_name : str
    is_mean    : float
    is_std     : float
    fid        : float
    mean_ssim  : float
    n_samples  : int
    elapsed_sec: float
    notes      : str = ""

    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"[{self.model_name}]  "
            f"IS={self.is_mean:.3f}±{self.is_std:.3f}  "
            f"FID={self.fid:.2f}  "
            f"SSIM={self.mean_ssim:.4f}  "
            f"(n={self.n_samples}, {self.elapsed_sec:.1f}s)"
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict (useful for logging / JSON export)."""
        return {
            "model_name" : self.model_name,
            "IS_mean"    : round(self.is_mean,  4),
            "IS_std"     : round(self.is_std,   4),
            "FID"        : round(self.fid,      4),
            "SSIM"       : round(self.mean_ssim,4),
            "n_samples"  : self.n_samples,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "notes"      : self.notes,
        }


# ---------------------------------------------------------------------------
# evaluate_model — single entry point: accepts folder paths
# ---------------------------------------------------------------------------

def evaluate_model(
    real_dir   : Union[str, Path],
    fake_dir   : Union[str, Path],
    model_name : str  = "model",
    batch_size : int  = 32,
    is_splits  : int  = 10,
    max_images : Optional[int] = None,
    transform  : Optional[object] = None,
    device     : str  = "cpu",
    verbose    : bool = True,
    notes      : str  = "",
) -> EvaluationResult:

    t0 = time.perf_counter()

    if verbose:
        print(f"[{model_name}] Loading images …", flush=True)

    real_images = load_images_from_folder(
        real_dir, transform=transform, max_images=max_images, verbose=verbose
    )
    fake_images = load_images_from_folder(
        fake_dir, transform=transform, max_images=max_images, verbose=verbose
    )

    # Align lengths for paired SSIM
    n = min(len(real_images), len(fake_images))
    if len(real_images) != len(fake_images):
        warnings.warn(
            f"[{model_name}] Folder sizes differ "
            f"(real={len(real_images)}, fake={len(fake_images)}). "
            f"SSIM will use the first {n} pairs.",
            UserWarning,
            stacklevel=2,
        )

    if len(fake_images) < 2048:
        warnings.warn(
            f"[{model_name}] Only {len(fake_images)} samples available. "
            "FID is unreliable below ~2 048 samples.",
            UserWarning,
            stacklevel=2,
        )

    # ---- IS ---------------------------------------------------------------
    if verbose:
        print(f"[{model_name}] Computing Inception Score …", flush=True)
    is_mean, is_std = inception_score(
        fake_images, batch_size=batch_size, splits=is_splits, device=device
    )

    # ---- FID --------------------------------------------------------------
    if verbose:
        print(f"[{model_name}] Computing FID …", flush=True)
    fid = frechet_inception_distance(
        real_images, fake_images, batch_size=batch_size, device=device
    )

    # ---- SSIM -------------------------------------------------------------
    if verbose:
        print(f"[{model_name}] Computing SSIM …", flush=True)
    ssim_value = mean_ssim(real_images[:n], fake_images[:n])

    elapsed = time.perf_counter() - t0

    result = EvaluationResult(
        model_name  = model_name,
        is_mean     = is_mean,
        is_std      = is_std,
        fid         = fid,
        mean_ssim   = ssim_value,
        n_samples   = len(fake_images),
        elapsed_sec = elapsed,
        notes       = notes,
    )

    if verbose:
        print(result.summary())

    return result


# ---------------------------------------------------------------------------
# ModelEvaluator — compare multiple generative models side-by-side
# ---------------------------------------------------------------------------

class ModelEvaluator:
    

    def __init__(
        self,
        real_dir   : Union[str, Path],
        device     : str  = "cpu",
        batch_size : int  = 32,
        is_splits  : int  = 10,
        max_images : Optional[int] = None,
        transform  : Optional[object] = None,
        verbose    : bool = True,
    ):
        self.real_dir   = Path(real_dir)
        self.device     = device
        self.batch_size = batch_size
        self.is_splits  = is_splits
        self.max_images = max_images
        self.transform  = transform
        self.verbose    = verbose
        self.results: Dict[str, EvaluationResult] = {}

    # ------------------------------------------------------------------

    def add_model(
        self,
        model_name : str,
        fake_dir   : Union[str, Path],
        notes      : str = "",
    ) -> EvaluationResult:
        
        if model_name in self.results:
            warnings.warn(
                f"Model '{model_name}' already evaluated. Overwriting.",
                UserWarning,
                stacklevel=2,
            )

        result = evaluate_model(
            real_dir   = self.real_dir,
            fake_dir   = fake_dir,
            model_name = model_name,
            batch_size = self.batch_size,
            is_splits  = self.is_splits,
            max_images = self.max_images,
            transform  = self.transform,
            device     = self.device,
            verbose    = self.verbose,
            notes      = notes,
        )
        self.results[model_name] = result
        return result

    # ------------------------------------------------------------------

    def report(self, sort_by: str = "FID") -> str:
        
        if not self.results:
            return "(No models evaluated yet.)"

        rows = list(self.results.values())

        _sort_cfg = {
            "FID" : (lambda r: r.fid,       False),
            "IS"  : (lambda r: r.is_mean,   True ),
            "SSIM": (lambda r: r.mean_ssim, True ),
        }
        key_fn, reverse = _sort_cfg.get(sort_by.upper(), (lambda r: r.fid, False))
        rows.sort(key=key_fn, reverse=reverse)

        header = (
            f"{'Model':<20}  {'IS (↑)':>14}  {'FID (↓)':>10}  "
            f"{'SSIM (↑)':>10}  {'N':>6}  {'Time':>7}"
        )
        sep   = "-" * len(header)
        lines = [sep, header, sep]

        for r in rows:
            lines.append(
                f"{r.model_name:<20}  "
                f"{r.is_mean:>7.3f}±{r.is_std:<6.3f}  "
                f"{r.fid:>10.2f}  "
                f"{r.mean_ssim:>10.4f}  "
                f"{r.n_samples:>6}  "
                f"{r.elapsed_sec:>6.1f}s"
            )

        lines.append(sep)
        lines.append(
            f"Sorted by {sort_by.upper()}  |  higher is better  |   lower is better"
        )

        table = "\n".join(lines)
        print(table)
        return table

    # ------------------------------------------------------------------

    def best(self, metric: str = "FID") -> Optional[EvaluationResult]:
       
        if not self.results:
            return None

        m = metric.upper()
        if m == "FID":
            return min(self.results.values(), key=lambda r: r.fid)
        elif m == "IS":
            return max(self.results.values(), key=lambda r: r.is_mean)
        elif m == "SSIM":
            return max(self.results.values(), key=lambda r: r.mean_ssim)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose FID, IS, or SSIM.")

    # ------------------------------------------------------------------

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            ) from e

        rows = [r.to_dict() for r in self.results.values()]
        return pd.DataFrame(rows).set_index("model_name")
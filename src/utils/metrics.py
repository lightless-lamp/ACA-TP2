import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



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


# ---------------------------------------------------------------------------
# Matrix square root via eigendecomposition  (avoids scipy)
# ---------------------------------------------------------------------------

def _mat_sqrt(A):
    """
    Square root of a symmetric positive semi-definite matrix.

    Decomposition:  A = V D V^T  =>  sqrt(A) = V sqrt(D) V^T

    numpy.linalg.eigh is used because it guarantees real eigenvalues for
    symmetric matrices.  Small negative eigenvalues from floating-point
    noise are clamped to zero before taking the square root.
    """
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0.0)
    return (vecs * np.sqrt(vals)) @ vecs.T


# Inception Score (IS) ------------------------------------------------------

def inception_score(images, batch_size=32, splits=10, device="cpu"):
    """
    Inception Score (IS) -- Salimans et al., 2016.

    Measures both quality (sharpness) and diversity of generated images.

        IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )

    where p(y|x) is the class-conditional distribution from InceptionV3 and
    p(y) = E_x[p(y|x)] is the marginal.  Higher IS is better.

    Parameters
    ----------
    images     : list of (C, H, W) float tensors, values in [0, 1]
    batch_size : InceptionV3 mini-batch size
    splits     : number of equal chunks used to estimate mean and std of IS
    device     : 'cpu' or 'cuda'

    Returns
    -------
    (mean_is, std_is) : (float, float)
    """
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


# 2. Frechet Inception Distance (FID)

def frechet_inception_distance(real_images, fake_images,
                                batch_size=32, device="cpu"):
    """
    Frechet Inception Distance (FID) -- Heusel et al., 2017.
 
    Compares the distribution of pool3 InceptionV3 features between real and
    generated images by treating both as multivariate Gaussians:
 
        FID = ||mu_r - mu_f||^2
              + Tr( Sigma_r + Sigma_f - 2 * sqrt(Sigma_r @ Sigma_f) )
 
    Lower FID is better.
    Note: statistical reliability improves with more images (ideally >= 2048),
    but the function runs with any number of samples.
 
    Parameters
    ----------
    real_images : list of (C, H, W) float tensors, values in [0, 1]
    fake_images : list of (C, H, W) float tensors, values in [0, 1]
    batch_size  : InceptionV3 mini-batch size
    device      : 'cpu' or 'cuda'
 
    Returns
    -------
    fid : float
    """
    inception = _get_inception(device)
 
    feats_r = _extract(real_images, inception, "features", batch_size, device)
    feats_f = _extract(fake_images, inception, "features", batch_size, device)
 
    mu_r,    sigma_r = feats_r.mean(axis=0), np.cov(feats_r, rowvar=False)
    mu_f,    sigma_f = feats_f.mean(axis=0), np.cov(feats_f, rowvar=False)
 
    diff      = mu_r - mu_f
    sqrt_prod = _mat_sqrt(sigma_r @ sigma_f)
 
    # Discard negligible imaginary parts that arise from numerical noise
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real
 
    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * sqrt_prod))
    return fid



# 3. Structural Similarity Index (SSIM)---------------------------------------

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
    """
    Structural Similarity Index (SSIM) -- Wang et al., 2004.

    Quantifies perceptual similarity between two images by comparing local
    luminance, contrast, and structure with a Gaussian sliding window:

        SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
                     ------------------------------------------
                     (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)

    where mu and sigma are local Gaussian-weighted means / standard deviations
    and sigma_xy is the local cross-covariance.
    C1 = (K1*L)^2, C2 = (K2*L)^2 are stability constants.

    Returns values in [-1, 1]; 1 means identical images.  Higher is better.

    Parameters
    ----------
    img1, img2  : torch tensors (C, H, W) or (B, C, H, W), values in [0, 1]
    window_size : size of the Gaussian kernel  (paper default: 11)
    sigma       : std of the Gaussian kernel   (paper default: 1.5)
    K1, K2      : stability constants          (paper defaults: 0.01, 0.03)
    L           : dynamic range (1.0 for [0, 1] images)

    Returns
    -------
    ssim_value : float
    """
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
    """
    Average SSIM over paired (real, fake) images.

    Parameters
    ----------
    real_images, fake_images : lists of (C, H, W) tensors, values in [0, 1]
    **kwargs                 : forwarded to ssim()

    Returns
    -------
    mean_ssim_value : float
    """
    if len(real_images) != len(fake_images):
        raise ValueError("real_images and fake_images must have the same length")
    scores = [ssim(r, f, **kwargs) for r, f in zip(real_images, fake_images)]
    return float(np.mean(scores))

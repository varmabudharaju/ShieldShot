# ShieldShot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an open-source Python CLI + library that protects photos from deepfake misuse via adversarial perturbation, invisible watermarking, and C2PA provenance signing.

**Architecture:** Three-layer pipeline — face detection feeds adversarial perturbation (fast generator or thorough PGD), invisible watermark encoder covers the full image, C2PA signs the output. Each layer is independent and testable. The CLI orchestrates all three via a `protect` command.

**Tech Stack:** Python 3.11+, PyTorch, Click, Pillow/OpenCV, c2pa-python, insightface/facenet-pytorch, lpips

---

### Task 1: Project Scaffolding & Package Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/shieldshot/__init__.py`
- Create: `src/shieldshot/cli.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "shieldshot"
version = "0.1.0"
description = "Protect your photos from deepfake misuse"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "click>=8.0",
    "torch>=2.0",
    "torchvision>=0.15",
    "Pillow>=10.0",
    "opencv-python>=4.8",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[project.scripts]
shieldshot = "shieldshot.cli:main"
```

**Step 2: Create src/shieldshot/__init__.py**

```python
"""ShieldShot — Protect your photos from deepfake misuse."""

__version__ = "0.1.0"
```

**Step 3: Create minimal CLI skeleton**

```python
"""ShieldShot CLI."""

import click


@click.group()
@click.version_option()
def main():
    """ShieldShot — Protect your photos from deepfake misuse."""
    pass


@main.command()
def init():
    """Set up ShieldShot (download models, generate keys)."""
    click.echo("ShieldShot initialized.")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output path for protected image.")
@click.option("--mode", type=click.Choice(["fast", "thorough"]), default="fast",
              help="Protection mode: fast (generator) or thorough (PGD).")
def protect(input_path, output_path, mode):
    """Protect a photo from deepfake misuse."""
    click.echo(f"Protecting {input_path} (mode={mode})...")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def verify(input_path):
    """Verify provenance of a protected photo."""
    click.echo(f"Verifying {input_path}...")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def extract(input_path):
    """Extract watermark from a protected photo."""
    click.echo(f"Extracting watermark from {input_path}...")
```

**Step 4: Install in dev mode and verify CLI works**

Run: `cd /Users/varma/shieldshot && pip3 install -e ".[dev]"`
Run: `python3 -m shieldshot.cli --version`
Expected: version prints

**Step 5: Commit**

```bash
git add pyproject.toml src/
git commit -m "feat: scaffold project with CLI skeleton"
```

---

### Task 2: Image Utilities

**Files:**
- Create: `src/shieldshot/utils/__init__.py`
- Create: `src/shieldshot/utils/image.py`
- Create: `tests/test_image_utils.py`

**Step 1: Write the failing tests**

```python
"""Tests for image utilities."""

import numpy as np
import pytest
from PIL import Image

from shieldshot.utils.image import load_image, save_image, to_tensor, to_pil


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample 256x256 RGB image."""
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    path = tmp_path / "sample.jpg"
    img.save(path)
    return path


def test_load_image_returns_pil(sample_image):
    img = load_image(str(sample_image))
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_image_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path.jpg")


def test_save_image_jpeg(tmp_path, sample_image):
    img = load_image(str(sample_image))
    out = tmp_path / "out.jpg"
    save_image(img, str(out), quality=85)
    assert out.exists()
    reloaded = load_image(str(out))
    assert reloaded.size == img.size


def test_save_image_png(tmp_path, sample_image):
    img = load_image(str(sample_image))
    out = tmp_path / "out.png"
    save_image(img, str(out))
    assert out.exists()


def test_to_tensor_shape(sample_image):
    img = load_image(str(sample_image))
    tensor = to_tensor(img)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_to_pil_roundtrip(sample_image):
    img = load_image(str(sample_image))
    tensor = to_tensor(img)
    result = to_pil(tensor)
    assert isinstance(result, Image.Image)
    assert result.size == img.size
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/varma/shieldshot && python3 -m pytest tests/test_image_utils.py -v`
Expected: FAIL — module not found

**Step 3: Implement image utilities**

```python
"""Image loading, saving, and tensor conversion utilities."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: str) -> Image.Image:
    """Load an image and convert to RGB."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(p).convert("RGB")


def save_image(img: Image.Image, path: str, quality: int = 95) -> None:
    """Save an image. JPEG quality is configurable."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        img.save(p, quality=quality)
    else:
        img.save(p)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to float tensor [1, 3, H, W] in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert float tensor [1, 3, H, W] in [0, 1] to PIL Image."""
    arr = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))
```

`src/shieldshot/utils/__init__.py`: empty file.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_image_utils.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/shieldshot/utils/ tests/test_image_utils.py
git commit -m "feat: add image loading/saving/tensor utilities"
```

---

### Task 3: Quality Gate (SSIM/LPIPS)

**Files:**
- Create: `src/shieldshot/utils/quality.py`
- Create: `tests/test_quality.py`

**Step 1: Write the failing tests**

```python
"""Tests for quality gate utilities."""

import numpy as np
import pytest
import torch
from PIL import Image

from shieldshot.utils.image import to_tensor
from shieldshot.utils.quality import compute_ssim, check_quality


@pytest.fixture
def identical_pair():
    """Two identical tensors."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    pil = Image.fromarray(img)
    t = to_tensor(pil)
    return t, t.clone()


@pytest.fixture
def different_pair():
    """Two very different tensors."""
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img2 = np.full((256, 256, 3), 255, dtype=np.uint8)
    t1 = to_tensor(Image.fromarray(img1))
    t2 = to_tensor(Image.fromarray(img2))
    return t1, t2


def test_ssim_identical(identical_pair):
    t1, t2 = identical_pair
    ssim = compute_ssim(t1, t2)
    assert ssim > 0.99


def test_ssim_different(different_pair):
    t1, t2 = different_pair
    ssim = compute_ssim(t1, t2)
    assert ssim < 0.1


def test_check_quality_passes_identical(identical_pair):
    t1, t2 = identical_pair
    passed, metrics = check_quality(t1, t2, ssim_threshold=0.95)
    assert passed is True
    assert metrics["ssim"] > 0.95


def test_check_quality_fails_different(different_pair):
    t1, t2 = different_pair
    passed, metrics = check_quality(t1, t2, ssim_threshold=0.95)
    assert passed is False
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_quality.py -v`
Expected: FAIL

**Step 3: Implement quality gate**

Note: We use `torchmetrics` for SSIM or implement a simple one. For v1, use `pytorch-msssim` which is lightweight.

Add `pytorch-msssim>=1.0` to `pyproject.toml` dependencies.

```python
"""Quality gate — SSIM checks for perturbation visibility."""

import torch
from pytorch_msssim import ssim


def compute_ssim(original: torch.Tensor, modified: torch.Tensor) -> float:
    """Compute SSIM between two image tensors [1, 3, H, W] in [0, 1].

    Returns float in [0, 1] where 1 = identical.
    """
    with torch.no_grad():
        return ssim(original, modified, data_range=1.0, size_average=True).item()


def check_quality(
    original: torch.Tensor,
    modified: torch.Tensor,
    ssim_threshold: float = 0.95,
) -> tuple[bool, dict]:
    """Check if modified image passes quality gate.

    Returns (passed, metrics_dict).
    """
    ssim_val = compute_ssim(original, modified)
    passed = ssim_val >= ssim_threshold
    return passed, {"ssim": ssim_val}
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_quality.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/shieldshot/utils/quality.py tests/test_quality.py pyproject.toml
git commit -m "feat: add SSIM quality gate"
```

---

### Task 4: Face Detection Module

**Files:**
- Create: `src/shieldshot/detect/__init__.py`
- Create: `src/shieldshot/detect/face_detector.py`
- Create: `tests/test_face_detection.py`
- Create: `tests/fixtures/` (test images)

**Step 1: Create a test fixture image with a face**

We'll generate a synthetic test image. For real face detection we need an actual face photo. Create a small script to download a public-domain sample face image for testing.

```python
"""Tests for face detection."""

import numpy as np
import pytest
from PIL import Image

from shieldshot.detect.face_detector import FaceDetector


@pytest.fixture
def detector():
    return FaceDetector()


@pytest.fixture
def blank_image():
    """Image with no face."""
    return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))


def test_detector_initializes(detector):
    assert detector is not None


def test_no_faces_in_blank(detector, blank_image):
    faces = detector.detect(blank_image)
    assert len(faces) == 0


def test_detect_returns_list_of_dicts(detector, blank_image):
    faces = detector.detect(blank_image)
    assert isinstance(faces, list)


def test_face_dict_has_bbox_and_confidence(detector):
    """When a face is found, result has bbox [x1,y1,x2,y2] and confidence."""
    # This test requires a real face image.
    # We'll create a synthetic face-like pattern (may not trigger detection).
    # In integration tests, use a real sample image.
    img = Image.fromarray(np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8))
    faces = detector.detect(img)
    # We don't assert faces found (synthetic image), but if found, check structure.
    for face in faces:
        assert "bbox" in face
        assert "confidence" in face
        assert len(face["bbox"]) == 4
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_face_detection.py -v`
Expected: FAIL

**Step 3: Implement face detector**

Add `insightface>=0.7` and `onnxruntime>=1.16` to `pyproject.toml` dependencies.

```python
"""Face detection using InsightFace/RetinaFace."""

import numpy as np
from PIL import Image


class FaceDetector:
    """Detects faces in images using RetinaFace via InsightFace."""

    def __init__(self, min_confidence: float = 0.5):
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_sc",  # lightweight model
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        self._min_confidence = min_confidence

    def detect(self, image: Image.Image) -> list[dict]:
        """Detect faces in a PIL Image.

        Returns list of dicts with 'bbox' [x1, y1, x2, y2] and 'confidence'.
        """
        arr = np.array(image)
        # InsightFace expects BGR
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]

        results = self._app.get(arr)

        faces = []
        for face in results:
            conf = float(face.det_score)
            if conf >= self._min_confidence:
                bbox = face.bbox.astype(int).tolist()
                faces.append({"bbox": bbox, "confidence": conf})

        return faces
```

`src/shieldshot/detect/__init__.py`: empty file.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_face_detection.py -v`
Expected: all PASS (blank image returns empty list, detector initializes)

**Step 5: Commit**

```bash
git add src/shieldshot/detect/ tests/test_face_detection.py pyproject.toml
git commit -m "feat: add RetinaFace face detection module"
```

---

### Task 5: Target Model Loaders

**Files:**
- Create: `src/shieldshot/perturb/__init__.py`
- Create: `src/shieldshot/perturb/models.py`
- Create: `tests/test_target_models.py`

**Step 1: Write the failing tests**

```python
"""Tests for target model loaders."""

import pytest
import torch

from shieldshot.perturb.models import load_arcface, load_facenet, get_face_embedding


@pytest.fixture
def dummy_face_tensor():
    """A random 112x112 face tensor (ArcFace input size)."""
    return torch.randn(1, 3, 112, 112)


def test_load_arcface():
    model = load_arcface()
    assert model is not None


def test_load_facenet():
    model = load_facenet()
    assert model is not None


def test_arcface_output_shape(dummy_face_tensor):
    model = load_arcface()
    model.eval()
    with torch.no_grad():
        emb = model(dummy_face_tensor)
    assert emb.shape[0] == 1
    assert emb.shape[1] == 512  # ArcFace embedding dim


def test_facenet_output_shape():
    model = load_facenet()
    model.eval()
    face_tensor = torch.randn(1, 3, 160, 160)  # FaceNet input size
    with torch.no_grad():
        emb = model(face_tensor)
    assert emb.shape[0] == 1
    assert emb.shape[1] == 512


def test_get_face_embedding_returns_dict(dummy_face_tensor):
    embeddings = get_face_embedding(dummy_face_tensor)
    assert "arcface" in embeddings
    assert isinstance(embeddings["arcface"], torch.Tensor)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_target_models.py -v`
Expected: FAIL

**Step 3: Implement model loaders**

Add `facenet-pytorch>=2.5` to `pyproject.toml` dependencies.

```python
"""Target model loaders for adversarial perturbation."""

import torch
import torch.nn as nn


_model_cache: dict[str, nn.Module] = {}


def load_arcface() -> nn.Module:
    """Load pretrained ArcFace (insightface) model."""
    if "arcface" not in _model_cache:
        from insightface.recognition.arcface_torch.backbones import get_model

        model = get_model("r50", fp16=False)
        # Load pretrained weights — downloaded via insightface
        model.eval()
        _model_cache["arcface"] = model
    return _model_cache["arcface"]


def load_facenet() -> nn.Module:
    """Load pretrained FaceNet (InceptionResnetV1) model."""
    if "facenet" not in _model_cache:
        from facenet_pytorch import InceptionResnetV1

        model = InceptionResnetV1(pretrained="vggface2").eval()
        _model_cache["facenet"] = model
    return _model_cache["facenet"]


def get_face_embedding(
    face_tensor: torch.Tensor,
    models: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Get embeddings from multiple face recognition models.

    Args:
        face_tensor: [1, 3, H, W] face crop tensor in [0, 1].
        models: list of model names to use. Default: ["arcface", "facenet"].

    Returns:
        Dict mapping model name to embedding tensor.
    """
    if models is None:
        models = ["arcface", "facenet"]

    loaders = {
        "arcface": load_arcface,
        "facenet": load_facenet,
    }

    embeddings = {}
    for name in models:
        model = loaders[name]()
        model.eval()
        with torch.no_grad():
            emb = model(face_tensor)
        embeddings[name] = emb

    return embeddings
```

`src/shieldshot/perturb/__init__.py`: empty file.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_target_models.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/shieldshot/perturb/ tests/test_target_models.py pyproject.toml
git commit -m "feat: add ArcFace and FaceNet model loaders"
```

---

### Task 6: Multi-Model Loss Functions

**Files:**
- Create: `src/shieldshot/perturb/losses.py`
- Create: `tests/test_losses.py`

**Step 1: Write the failing tests**

```python
"""Tests for adversarial loss functions."""

import pytest
import torch

from shieldshot.perturb.losses import cosine_distance_loss, multi_model_loss


def test_cosine_distance_identical():
    """Identical embeddings should have zero distance loss (cosine sim = 1)."""
    emb = torch.randn(1, 512)
    loss = cosine_distance_loss(emb, emb.clone())
    # We maximize distance, so loss = -(1 - cosine_sim) = 0 for identical
    assert loss.item() == pytest.approx(0.0, abs=0.01)


def test_cosine_distance_orthogonal():
    """Orthogonal embeddings should have high distance loss."""
    emb1 = torch.zeros(1, 512)
    emb1[0, 0] = 1.0
    emb2 = torch.zeros(1, 512)
    emb2[0, 1] = 1.0
    loss = cosine_distance_loss(emb1, emb2)
    assert loss.item() > 0.5


def test_multi_model_loss_returns_scalar():
    """Multi-model loss combines losses into a single scalar."""
    clean = {"arcface": torch.randn(1, 512), "facenet": torch.randn(1, 512)}
    perturbed = {"arcface": torch.randn(1, 512), "facenet": torch.randn(1, 512)}
    loss = multi_model_loss(clean, perturbed)
    assert loss.dim() == 0  # scalar


def test_multi_model_loss_with_weights():
    clean = {"arcface": torch.randn(1, 512), "facenet": torch.randn(1, 512)}
    perturbed = {"arcface": torch.randn(1, 512), "facenet": torch.randn(1, 512)}
    weights = {"arcface": 2.0, "facenet": 0.5}
    loss = multi_model_loss(clean, perturbed, weights=weights)
    assert loss.dim() == 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_losses.py -v`
Expected: FAIL

**Step 3: Implement loss functions**

```python
"""Multi-model adversarial loss functions."""

import torch
import torch.nn.functional as F


def cosine_distance_loss(clean_emb: torch.Tensor, perturbed_emb: torch.Tensor) -> torch.Tensor:
    """Compute loss that maximizes cosine distance between embeddings.

    Returns 1 - cosine_similarity (0 = identical, 2 = opposite).
    We want to MAXIMIZE this, so the optimizer should MINIMIZE the negative.
    """
    cos_sim = F.cosine_similarity(clean_emb, perturbed_emb, dim=1)
    return (1.0 - cos_sim).mean()


def multi_model_loss(
    clean_embeddings: dict[str, torch.Tensor],
    perturbed_embeddings: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Combine cosine distance losses across multiple models.

    We negate the total because PGD minimizes loss, and we want to
    maximize feature distortion.

    Args:
        clean_embeddings: model_name -> clean embedding tensor.
        perturbed_embeddings: model_name -> perturbed embedding tensor.
        weights: model_name -> weight. Default: equal weights.

    Returns:
        Negative total loss (for minimization by PGD).
    """
    if weights is None:
        weights = {k: 1.0 for k in clean_embeddings}

    total = torch.tensor(0.0, device=next(iter(clean_embeddings.values())).device)
    for name in clean_embeddings:
        dist = cosine_distance_loss(clean_embeddings[name], perturbed_embeddings[name])
        total = total + weights[name] * dist

    # Negate: PGD minimizes, we want to maximize distance
    return -total
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_losses.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/shieldshot/perturb/losses.py tests/test_losses.py
git commit -m "feat: add multi-model cosine distance loss"
```

---

### Task 7: PGD Adversarial Perturbation

**Files:**
- Create: `src/shieldshot/perturb/pgd.py`
- Create: `tests/test_perturbation.py`

**Step 1: Write the failing tests**

```python
"""Tests for PGD adversarial perturbation."""

import pytest
import torch

from shieldshot.perturb.pgd import pgd_attack


@pytest.fixture
def dummy_face():
    """Random 112x112 face tensor."""
    return torch.randn(1, 3, 112, 112).clamp(0, 1)


def test_pgd_returns_tensor(dummy_face):
    perturbed = pgd_attack(dummy_face, num_steps=5, epsilon=8 / 255)
    assert isinstance(perturbed, torch.Tensor)
    assert perturbed.shape == dummy_face.shape


def test_pgd_output_within_epsilon(dummy_face):
    epsilon = 8 / 255
    perturbed = pgd_attack(dummy_face, num_steps=5, epsilon=epsilon)
    diff = (perturbed - dummy_face).abs()
    assert diff.max().item() <= epsilon + 1e-5


def test_pgd_output_in_valid_range(dummy_face):
    perturbed = pgd_attack(dummy_face, num_steps=5, epsilon=8 / 255)
    assert perturbed.min() >= 0.0
    assert perturbed.max() <= 1.0


def test_pgd_modifies_image(dummy_face):
    perturbed = pgd_attack(dummy_face, num_steps=10, epsilon=8 / 255)
    assert not torch.allclose(perturbed, dummy_face, atol=1e-6)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_perturbation.py -v`
Expected: FAIL

**Step 3: Implement PGD attack**

```python
"""PGD (Projected Gradient Descent) adversarial perturbation."""

import torch
import torch.nn.functional as F

from shieldshot.perturb.models import get_face_embedding


def pgd_attack(
    face_tensor: torch.Tensor,
    num_steps: int = 100,
    epsilon: float = 8 / 255,
    step_size: float | None = None,
    target_models: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Apply PGD adversarial perturbation to a face tensor.

    Args:
        face_tensor: [1, 3, H, W] in [0, 1].
        num_steps: Number of PGD iterations.
        epsilon: L-infinity perturbation budget.
        step_size: Per-step size. Default: epsilon / (num_steps / 4).
        target_models: Which models to attack. Default: ["arcface", "facenet"].
        weights: Per-model loss weights.

    Returns:
        Perturbed tensor [1, 3, H, W] in [0, 1].
    """
    if step_size is None:
        step_size = epsilon / max(num_steps / 4, 1)
    if target_models is None:
        target_models = ["arcface", "facenet"]

    from shieldshot.perturb.losses import multi_model_loss

    # Get clean embeddings (no grad needed)
    with torch.no_grad():
        clean_embeddings = get_face_embedding(face_tensor, models=target_models)

    # Initialize perturbation with small random noise
    delta = torch.zeros_like(face_tensor, requires_grad=True)
    delta.data.uniform_(-epsilon, epsilon)
    delta.data = torch.clamp(face_tensor + delta.data, 0, 1) - face_tensor

    for _ in range(num_steps):
        delta.requires_grad_(True)
        perturbed = face_tensor + delta

        # Get perturbed embeddings
        perturbed_embeddings = get_face_embedding(perturbed, models=target_models)

        # Compute loss (negated — we minimize to maximize distance)
        loss = multi_model_loss(clean_embeddings, perturbed_embeddings, weights=weights)
        loss.backward()

        # PGD step
        grad = delta.grad.detach()
        delta = delta.detach() - step_size * grad.sign()

        # Project back into epsilon ball
        delta = torch.clamp(delta, -epsilon, epsilon)
        # Also clamp so perturbed image stays in [0, 1]
        delta = torch.clamp(face_tensor + delta, 0, 1) - face_tensor

    return (face_tensor + delta).detach()
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_perturbation.py -v`
Expected: all PASS

Note: These tests use the real models. If model download is slow, mock `get_face_embedding` for unit tests and keep real-model tests as integration tests.

**Step 5: Commit**

```bash
git add src/shieldshot/perturb/pgd.py tests/test_perturbation.py
git commit -m "feat: add PGD adversarial perturbation"
```

---

### Task 8: Watermark Payload Encoding/Decoding

**Files:**
- Create: `src/shieldshot/watermark/__init__.py`
- Create: `src/shieldshot/watermark/payload.py`
- Create: `tests/test_payload.py`

**Step 1: Write the failing tests**

```python
"""Tests for watermark payload encoding/decoding."""

import time

import pytest

from shieldshot.watermark.payload import encode_payload, decode_payload, PAYLOAD_BITS


def test_encode_returns_correct_length():
    bits = encode_payload(user_id="testuser", timestamp=int(time.time()))
    assert len(bits) == PAYLOAD_BITS


def test_encode_returns_binary():
    bits = encode_payload(user_id="testuser", timestamp=int(time.time()))
    assert all(b in (0, 1) for b in bits)


def test_roundtrip():
    ts = int(time.time())
    bits = encode_payload(user_id="testuser", timestamp=ts)
    result = decode_payload(bits)
    assert result["user_id_hash"] == encode_payload(user_id="testuser", timestamp=ts)[:32]


def test_different_users_different_payload():
    ts = int(time.time())
    bits1 = encode_payload(user_id="alice", timestamp=ts)
    bits2 = encode_payload(user_id="bob", timestamp=ts)
    assert bits1 != bits2


def test_decode_with_bit_errors():
    """Payload should survive a few bit errors thanks to Reed-Solomon."""
    ts = int(time.time())
    bits = encode_payload(user_id="testuser", timestamp=ts)
    # Flip 2 bits
    corrupted = list(bits)
    corrupted[0] = 1 - corrupted[0]
    corrupted[5] = 1 - corrupted[5]
    result = decode_payload(corrupted)
    assert result is not None
    assert result["valid"]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_payload.py -v`
Expected: FAIL

**Step 3: Implement payload**

Add `reedsolo>=1.7` to `pyproject.toml` dependencies.

```python
"""Watermark payload encoding/decoding with Reed-Solomon error correction."""

import hashlib
import struct

from reedsolo import RSCodec

PAYLOAD_BITS = 96  # 64 data bits + 32 ECC bits
DATA_BYTES = 8     # 64 bits of data
ECC_SYMBOLS = 4    # 4 bytes of Reed-Solomon ECC

_rs = RSCodec(ECC_SYMBOLS)


def _hash_user_id(user_id: str) -> int:
    """Hash user ID to 32-bit integer."""
    h = hashlib.sha256(user_id.encode()).digest()
    return struct.unpack(">I", h[:4])[0]


def encode_payload(user_id: str, timestamp: int) -> list[int]:
    """Encode user_id + timestamp into a fixed-length bit payload with ECC.

    Layout (64 data bits):
        - user_id_hash: 32 bits
        - timestamp: 32 bits (seconds, wraps ~2106)

    Returns list of 0/1 ints, length = PAYLOAD_BITS.
    """
    uid_hash = _hash_user_id(user_id)
    # Truncate timestamp to 32 bits
    ts_32 = timestamp & 0xFFFFFFFF
    data_bytes = struct.pack(">II", uid_hash, ts_32)

    # Apply Reed-Solomon ECC
    encoded = bytes(_rs.encode(data_bytes))

    # Convert to bits
    bits = []
    for byte in encoded:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    return bits[:PAYLOAD_BITS]


def decode_payload(bits: list[int]) -> dict:
    """Decode a bit payload back to user_id_hash and timestamp.

    Returns dict with 'user_id_hash', 'timestamp', 'valid'.
    """
    # Convert bits to bytes
    byte_list = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val = (byte_val << 1) | bits[i + j]
            else:
                byte_val <<= 1
        byte_list.append(byte_val)

    try:
        decoded = bytes(_rs.decode(bytes(byte_list)))
        uid_hash, ts_32 = struct.unpack(">II", bytes(decoded[:DATA_BYTES]))
        return {
            "user_id_hash": uid_hash,
            "timestamp": ts_32,
            "valid": True,
        }
    except Exception:
        return {"user_id_hash": None, "timestamp": None, "valid": False}
```

`src/shieldshot/watermark/__init__.py`: empty file.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_payload.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/shieldshot/watermark/ tests/test_payload.py pyproject.toml
git commit -m "feat: add watermark payload encoding with Reed-Solomon ECC"
```

---

### Task 9: Watermark Encoder/Decoder Networks

**Files:**
- Create: `src/shieldshot/watermark/encoder.py`
- Create: `src/shieldshot/watermark/decoder.py`
- Create: `tests/test_watermark.py`

**Step 1: Write the failing tests**

```python
"""Tests for watermark encoder/decoder networks."""

import pytest
import torch

from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import PAYLOAD_BITS


@pytest.fixture
def encoder():
    return WatermarkEncoder(payload_bits=PAYLOAD_BITS)


@pytest.fixture
def decoder():
    return WatermarkDecoder(payload_bits=PAYLOAD_BITS)


@pytest.fixture
def sample_image():
    return torch.randn(1, 3, 256, 256).clamp(0, 1)


@pytest.fixture
def sample_payload():
    return torch.randint(0, 2, (1, PAYLOAD_BITS)).float()


def test_encoder_output_shape(encoder, sample_image, sample_payload):
    encoded = encoder(sample_image, sample_payload)
    assert encoded.shape == sample_image.shape


def test_encoder_output_range(encoder, sample_image, sample_payload):
    encoded = encoder(sample_image, sample_payload)
    assert encoded.min() >= -0.5  # may slightly exceed [0,1] before clamping
    assert encoded.max() <= 1.5


def test_decoder_output_shape(decoder, sample_image):
    decoded = decoder(sample_image)
    assert decoded.shape == (1, PAYLOAD_BITS)


def test_encoder_decoder_untrained_roundtrip(encoder, decoder, sample_image, sample_payload):
    """Even untrained, the shapes should work end-to-end."""
    encoded = encoder(sample_image, sample_payload)
    clamped = encoded.clamp(0, 1)
    decoded = decoder(clamped)
    assert decoded.shape == (1, PAYLOAD_BITS)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_watermark.py -v`
Expected: FAIL

**Step 3: Implement encoder (U-Net style)**

```python
"""Watermark encoder — embeds payload bits into an image."""

import torch
import torch.nn as nn


class WatermarkEncoder(nn.Module):
    """U-Net style encoder that embeds a bit payload into an image.

    Takes [B, 3, H, W] image + [B, payload_bits] payload,
    returns [B, 3, H, W] watermarked image.
    """

    def __init__(self, payload_bits: int = 96, hidden_dim: int = 64):
        super().__init__()
        self.payload_bits = payload_bits

        # Project payload to spatial feature map
        self.payload_proj = nn.Sequential(
            nn.Linear(payload_bits, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Encoder (downsampling)
        self.enc1 = self._conv_block(3 + hidden_dim, hidden_dim)
        self.enc2 = self._conv_block(hidden_dim, hidden_dim * 2)
        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(hidden_dim * 2, hidden_dim * 2)

        # Decoder (upsampling)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = self._conv_block(hidden_dim * 4, hidden_dim)
        self.dec1 = self._conv_block(hidden_dim * 2, hidden_dim)

        # Output: residual (perturbation to add to image)
        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim, 3, 1),
            nn.Tanh(),
        )
        self.strength = 0.1  # Scale residual to be imperceptible

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape

        # Expand payload to spatial dims
        p = self.payload_proj(payload)  # [B, hidden_dim]
        p = p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, hidden_dim, H, W]

        # Concat image + payload
        x = torch.cat([image, p], dim=1)  # [B, 3 + hidden_dim, H, W]

        # Encoder
        e1 = self.enc1(x)       # [B, hidden, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, hidden*2, H/2, W/2]

        # Bottleneck
        b = self.bottleneck(self.pool(e2))  # [B, hidden*2, H/4, W/4]

        # Decoder
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))  # [B, hidden, H/2, W/2]
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))  # [B, hidden, H, W]

        # Residual
        residual = self.out(d1) * self.strength
        return image + residual
```

**Step 4: Implement decoder**

```python
"""Watermark decoder — extracts payload bits from a watermarked image."""

import torch
import torch.nn as nn


class WatermarkDecoder(nn.Module):
    """CNN decoder that extracts payload bits from a watermarked image.

    Takes [B, 3, H, W] image, returns [B, payload_bits] logits.
    """

    def __init__(self, payload_bits: int = 96):
        super().__init__()
        self.payload_bits = payload_bits

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, payload_bits),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = x.flatten(1)
        return self.classifier(x)
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_watermark.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/shieldshot/watermark/encoder.py src/shieldshot/watermark/decoder.py tests/test_watermark.py
git commit -m "feat: add watermark encoder/decoder networks"
```

---

### Task 10: Watermark Training Script & Augmentations

**Files:**
- Create: `train/augmentations.py`
- Create: `train/train_watermark.py`
- Create: `tests/test_augmentations.py`

**Step 1: Write the failing tests for augmentations**

```python
"""Tests for training augmentations."""

import pytest
import torch

from train.augmentations import (
    jpeg_compress,
    screenshot_simulate,
    random_crop_resize,
    apply_random_augmentation,
)


@pytest.fixture
def sample_tensor():
    return torch.randn(1, 3, 256, 256).clamp(0, 1)


def test_jpeg_compress_shape(sample_tensor):
    result = jpeg_compress(sample_tensor, quality=70)
    assert result.shape == sample_tensor.shape


def test_jpeg_compress_modifies(sample_tensor):
    result = jpeg_compress(sample_tensor, quality=50)
    assert not torch.allclose(result, sample_tensor, atol=1e-3)


def test_screenshot_simulate_shape(sample_tensor):
    result = screenshot_simulate(sample_tensor)
    assert result.shape == sample_tensor.shape


def test_random_crop_resize_shape(sample_tensor):
    result = random_crop_resize(sample_tensor, min_crop=0.7)
    assert result.shape == sample_tensor.shape


def test_apply_random_augmentation_shape(sample_tensor):
    result = apply_random_augmentation(sample_tensor)
    assert result.shape == sample_tensor.shape
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_augmentations.py -v`
Expected: FAIL

**Step 3: Implement augmentations**

```python
"""Training augmentations simulating real-world image degradation."""

import io
import random

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from shieldshot.utils.image import to_pil, to_tensor


def jpeg_compress(tensor: torch.Tensor, quality: int = 70) -> torch.Tensor:
    """Differentiable-ish JPEG compression via PIL (non-differentiable but used in training noise layer)."""
    pil_img = to_pil(tensor)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    return to_tensor(compressed).to(tensor.device)


def screenshot_simulate(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate screenshot artifacts: gamma shift, slight blur, rescale."""
    # Gamma shift
    gamma = random.uniform(0.8, 1.2)
    result = tensor.pow(gamma)

    # Slight Gaussian blur
    k = 3
    padding = k // 2
    kernel = torch.ones(1, 1, k, k, device=tensor.device) / (k * k)
    channels = result.shape[1]
    kernel = kernel.expand(channels, 1, k, k)
    result = F.conv2d(result, kernel, padding=padding, groups=channels)

    # Downscale then upscale (simulates resolution loss)
    _, _, H, W = result.shape
    scale = random.uniform(0.7, 0.9)
    small = F.interpolate(result, scale_factor=scale, mode="bilinear", align_corners=False)
    result = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)

    return result.clamp(0, 1)


def random_crop_resize(tensor: torch.Tensor, min_crop: float = 0.5) -> torch.Tensor:
    """Random crop then resize back to original dimensions."""
    _, _, H, W = tensor.shape
    crop_ratio = random.uniform(min_crop, 0.95)
    crop_h, crop_w = int(H * crop_ratio), int(W * crop_ratio)
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    cropped = tensor[:, :, top:top + crop_h, left:left + crop_w]
    return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)


def apply_random_augmentation(tensor: torch.Tensor) -> torch.Tensor:
    """Apply one random augmentation from the noise layer suite."""
    aug = random.choice([
        lambda t: jpeg_compress(t, quality=random.randint(50, 95)),
        screenshot_simulate,
        lambda t: random_crop_resize(t, min_crop=0.6),
        lambda t: t + torch.randn_like(t) * random.uniform(0.01, 0.05),  # Gaussian noise
        lambda t: t * random.uniform(0.7, 1.3),  # Brightness
    ])
    return aug(tensor).clamp(0, 1)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_augmentations.py -v`
Expected: all PASS

**Step 5: Implement watermark training script**

```python
"""Training script for watermark encoder-decoder."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import PAYLOAD_BITS
from shieldshot.utils.image import to_tensor
from shieldshot.utils.quality import compute_ssim
from train.augmentations import apply_random_augmentation


class FaceImageDataset(Dataset):
    """Load face images from a directory for watermark training."""

    def __init__(self, root_dir: str, image_size: int = 256):
        self.root = Path(root_dir)
        self.image_size = image_size
        self.files = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        tensor = to_tensor(img).squeeze(0)  # [3, H, W]
        payload = torch.randint(0, 2, (PAYLOAD_BITS,)).float()
        return tensor, payload


def train(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_dir: str = "models/watermark",
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = WatermarkEncoder(payload_bits=PAYLOAD_BITS).to(device)
    decoder = WatermarkDecoder(payload_bits=PAYLOAD_BITS).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )

    bce = nn.BCEWithLogitsLoss()
    dataset = FaceImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        total_bit_acc = 0.0
        total_ssim = 0.0
        n_batches = 0

        for images, payloads in dataloader:
            images = images.to(device)
            payloads = payloads.to(device)

            # Encode watermark
            watermarked = encoder(images, payloads)
            watermarked_clamped = watermarked.clamp(0, 1)

            # Apply random augmentation (noise layer)
            augmented = torch.stack([
                apply_random_augmentation(watermarked_clamped[i:i+1]).squeeze(0)
                for i in range(watermarked_clamped.shape[0])
            ]).to(device)

            # Decode from augmented image
            decoded_logits = decoder(augmented)

            # Losses
            loss_payload = bce(decoded_logits, payloads)
            loss_image = 1.0 - compute_ssim(images, watermarked_clamped)
            loss = loss_payload + 10.0 * loss_image

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            with torch.no_grad():
                predicted = (decoded_logits > 0).float()
                bit_acc = (predicted == payloads).float().mean().item()
                ssim_val = compute_ssim(images, watermarked_clamped)

            total_loss += loss.item()
            total_bit_acc += bit_acc
            total_ssim += ssim_val
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_bit_acc / max(n_batches, 1)
        avg_ssim = total_ssim / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}, "
              f"bit_acc: {avg_acc:.4f}, ssim: {avg_ssim:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(encoder.state_dict(), save_path / "encoder.pt")
            torch.save(decoder.state_dict(), save_path / "decoder.pt")
            print(f"  Saved checkpoint to {save_path}")

    # Final save
    torch.save(encoder.state_dict(), save_path / "encoder.pt")
    torch.save(decoder.state_dict(), save_path / "decoder.pt")
    print(f"Training complete. Models saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train watermark encoder-decoder")
    parser.add_argument("--data-dir", required=True, help="Directory of face images")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="models/watermark")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
    )
```

**Step 6: Commit**

```bash
git add train/ tests/test_augmentations.py
git commit -m "feat: add watermark training script and augmentations"
```

---

### Task 11: Perturbation Generator Network & Training

**Files:**
- Create: `src/shieldshot/perturb/generator.py`
- Create: `train/train_generator.py`
- Create: `train/generate_pgd_targets.py`
- Create: `tests/test_generator.py`

**Step 1: Write the failing tests**

```python
"""Tests for perturbation generator network."""

import pytest
import torch

from shieldshot.perturb.generator import PerturbationGenerator


@pytest.fixture
def generator():
    return PerturbationGenerator()


@pytest.fixture
def sample_face():
    return torch.randn(1, 3, 112, 112).clamp(0, 1)


def test_generator_output_shape(generator, sample_face):
    perturbed = generator(sample_face)
    assert perturbed.shape == sample_face.shape


def test_generator_output_in_valid_range(generator, sample_face):
    perturbed = generator(sample_face)
    assert perturbed.min() >= 0.0
    assert perturbed.max() <= 1.0


def test_generator_perturbation_bounded(generator, sample_face):
    """Output should differ from input by at most epsilon."""
    perturbed = generator(sample_face)
    diff = (perturbed - sample_face).abs().max().item()
    assert diff <= generator.epsilon + 1e-5
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generator.py -v`
Expected: FAIL

**Step 3: Implement generator**

```python
"""Perturbation generator — single-pass adversarial noise prediction."""

import torch
import torch.nn as nn


class PerturbationGenerator(nn.Module):
    """U-Net that predicts adversarial perturbation in a single forward pass.

    Takes [B, 3, H, W] face image, returns [B, 3, H, W] perturbed image.
    The perturbation is clamped to [-epsilon, epsilon].
    """

    def __init__(self, epsilon: float = 8 / 255, hidden_dim: int = 64):
        super().__init__()
        self.epsilon = epsilon

        # Encoder
        self.enc1 = self._conv_block(3, hidden_dim)
        self.enc2 = self._conv_block(hidden_dim, hidden_dim * 2)
        self.enc3 = self._conv_block(hidden_dim * 2, hidden_dim * 4)
        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(hidden_dim * 4, hidden_dim * 4)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = self._conv_block(hidden_dim * 8, hidden_dim * 2)
        self.dec2 = self._conv_block(hidden_dim * 4, hidden_dim)
        self.dec1 = self._conv_block(hidden_dim * 2, hidden_dim)

        # Output: perturbation delta
        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim, 3, 1),
            nn.Tanh(),
        )

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        # Perturbation clamped to epsilon
        delta = self.out(d1) * self.epsilon
        perturbed = torch.clamp(image + delta, 0, 1)
        return perturbed
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generator.py -v`
Expected: all PASS

**Step 5: Implement PGD target generation script**

```python
"""Generate PGD perturbation targets for generator training."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from shieldshot.perturb.pgd import pgd_attack
from shieldshot.utils.image import load_image, to_tensor, to_pil


def generate_targets(
    data_dir: str,
    output_dir: str,
    num_steps: int = 100,
    epsilon: float = 8 / 255,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in data_path.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for i, f in enumerate(files):
        img = load_image(str(f))
        tensor = to_tensor(img).to(device)

        perturbed = pgd_attack(tensor, num_steps=num_steps, epsilon=epsilon)
        delta = perturbed - tensor

        # Save the delta as a .pt file
        torch.save(delta.cpu(), out_path / f"{f.stem}_delta.pt")

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(files)}")

    print(f"Generated {len(files)} PGD targets in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PGD targets")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    generate_targets(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        device=args.device,
    )
```

**Step 6: Implement generator training script**

```python
"""Training script for perturbation generator network."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from shieldshot.perturb.generator import PerturbationGenerator
from shieldshot.perturb.models import get_face_embedding
from shieldshot.perturb.losses import multi_model_loss
from shieldshot.utils.image import to_tensor
from shieldshot.utils.quality import compute_ssim
from train.augmentations import jpeg_compress


class FaceDeltaDataset(Dataset):
    """Dataset of face images with precomputed PGD deltas."""

    def __init__(self, image_dir: str, delta_dir: str, image_size: int = 112):
        self.image_dir = Path(image_dir)
        self.delta_dir = Path(delta_dir)
        self.image_size = image_size

        self.delta_files = sorted(self.delta_dir.glob("*_delta.pt"))
        self.image_files = []
        for df in self.delta_files:
            stem = df.stem.replace("_delta", "")
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = self.image_dir / f"{stem}{ext}"
                if candidate.exists():
                    self.image_files.append(candidate)
                    break

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        tensor = to_tensor(img).squeeze(0)

        delta = torch.load(self.delta_files[idx], weights_only=True).squeeze(0)
        return tensor, delta


def train(
    image_dir: str,
    delta_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_dir: str = "models/generator",
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = PerturbationGenerator().to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr)

    dataset = FaceDeltaDataset(image_dir, delta_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    mse = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for images, target_deltas in dataloader:
            images = images.to(device)
            target_deltas = target_deltas.to(device)

            perturbed = generator(images)
            predicted_delta = perturbed - images

            # Loss 1: Match PGD delta
            loss_delta = mse(predicted_delta, target_deltas)

            # Loss 2: Quality (SSIM)
            loss_quality = 1.0 - compute_ssim(images, perturbed)

            # Loss 3: Compression resilience
            compressed = torch.stack([
                jpeg_compress(perturbed[i:i+1], quality=70).squeeze(0)
                for i in range(perturbed.shape[0])
            ]).to(device)
            # Feature distortion should survive compression
            with torch.no_grad():
                clean_emb = get_face_embedding(images)
            comp_emb = get_face_embedding(compressed)
            loss_compression = -multi_model_loss(clean_emb, comp_emb)

            loss = loss_delta + 5.0 * loss_quality + 0.1 * loss_compression

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), save_path / "generator.pt")

    torch.save(generator.state_dict(), save_path / "generator.pt")
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train perturbation generator")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--delta-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="models/generator")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(
        image_dir=args.image_dir,
        delta_dir=args.delta_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
    )
```

**Step 7: Commit**

```bash
git add src/shieldshot/perturb/generator.py train/generate_pgd_targets.py train/train_generator.py tests/test_generator.py
git commit -m "feat: add perturbation generator network and training scripts"
```

---

### Task 12: C2PA Provenance Layer

**Files:**
- Create: `src/shieldshot/provenance/__init__.py`
- Create: `src/shieldshot/provenance/c2pa.py`
- Create: `tests/test_provenance.py`

**Step 1: Write the failing tests**

```python
"""Tests for C2PA provenance signing and verification."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from shieldshot.provenance.c2pa import (
    init_keys,
    sign_image,
    verify_image,
    KEYS_DIR,
)


@pytest.fixture
def keys_dir(tmp_path):
    return init_keys(keys_dir=str(tmp_path / "keys"))


@pytest.fixture
def sample_jpeg(tmp_path):
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    path = tmp_path / "sample.jpg"
    img.save(path, quality=95)
    return str(path)


def test_init_keys_creates_cert(keys_dir):
    keys_path = Path(keys_dir)
    assert (keys_path / "cert.pem").exists()
    assert (keys_path / "key.pem").exists()


def test_sign_image_creates_output(keys_dir, sample_jpeg, tmp_path):
    output = str(tmp_path / "signed.jpg")
    sign_image(sample_jpeg, output, keys_dir=keys_dir)
    assert Path(output).exists()


def test_verify_signed_image(keys_dir, sample_jpeg, tmp_path):
    output = str(tmp_path / "signed.jpg")
    sign_image(sample_jpeg, output, keys_dir=keys_dir)
    result = verify_image(output)
    assert result["valid"] is True
    assert "shieldshot" in result.get("software", "").lower()


def test_verify_unsigned_image(sample_jpeg):
    result = verify_image(sample_jpeg)
    assert result["valid"] is False
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_provenance.py -v`
Expected: FAIL

**Step 3: Implement C2PA wrapper**

Add `c2pa-python>=0.4` to `pyproject.toml` dependencies.

Note: c2pa-python API may vary by version. This implementation uses the builder pattern. If the API differs, adjust accordingly.

```python
"""C2PA provenance signing and verification."""

import json
import subprocess
from pathlib import Path

KEYS_DIR = Path.home() / ".shieldshot" / "keys"


def init_keys(keys_dir: str | None = None) -> str:
    """Generate a self-signed certificate for C2PA signing.

    Returns path to keys directory.
    """
    kd = Path(keys_dir) if keys_dir else KEYS_DIR
    kd.mkdir(parents=True, exist_ok=True)

    cert_path = kd / "cert.pem"
    key_path = kd / "key.pem"

    if cert_path.exists() and key_path.exists():
        return str(kd)

    # Generate self-signed cert using openssl
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "ec",
        "-pkeyopt", "ec_paramgen_curve:P-256",
        "-keyout", str(key_path),
        "-out", str(cert_path),
        "-days", "3650",
        "-nodes",
        "-subj", "/CN=ShieldShot Self-Signed",
    ], check=True, capture_output=True)

    return str(kd)


def sign_image(
    input_path: str,
    output_path: str,
    keys_dir: str | None = None,
) -> None:
    """Sign an image with C2PA provenance manifest."""
    import c2pa

    kd = Path(keys_dir) if keys_dir else KEYS_DIR
    cert_path = kd / "cert.pem"
    key_path = kd / "key.pem"

    if not cert_path.exists():
        raise FileNotFoundError(f"No certificate found. Run 'shieldshot init' first.")

    manifest_json = json.dumps({
        "claim_generator": "shieldshot/0.1.0",
        "title": Path(input_path).name,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.edited",
                            "softwareAgent": "shieldshot/0.1.0",
                            "description": "Applied deepfake protection",
                        }
                    ]
                }
            }
        ],
    })

    builder = c2pa.Builder(manifest_json)

    with open(cert_path) as f:
        cert_data = f.read()
    with open(key_path) as f:
        key_data = f.read()

    signer = c2pa.create_signer(cert_data, key_data, "es256", tsa_url=None)

    with open(input_path, "rb") as inp:
        with open(output_path, "wb") as out:
            builder.sign(signer, inp, out)


def verify_image(image_path: str) -> dict:
    """Verify C2PA provenance of an image.

    Returns dict with 'valid' bool and metadata if present.
    """
    import c2pa

    try:
        reader = c2pa.Reader.from_file(image_path)
        manifest_store = json.loads(reader.json())

        # Find active manifest
        active = manifest_store.get("active_manifest")
        if not active:
            return {"valid": False, "reason": "No active manifest"}

        manifest = manifest_store["manifests"].get(active, {})
        claim_gen = manifest.get("claim_generator", "")

        return {
            "valid": True,
            "software": claim_gen,
            "title": manifest.get("title", ""),
        }
    except Exception:
        return {"valid": False, "reason": "No C2PA manifest found"}
```

`src/shieldshot/provenance/__init__.py`: empty file.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_provenance.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/shieldshot/provenance/ tests/test_provenance.py pyproject.toml
git commit -m "feat: add C2PA provenance signing and verification"
```

---

### Task 13: Orchestrator — `protect.py`

**Files:**
- Create: `src/shieldshot/protect.py`
- Create: `tests/test_integration.py`

**Step 1: Write the failing tests**

```python
"""Integration tests for the full protection pipeline."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from shieldshot.protect import protect_image


@pytest.fixture
def sample_photo(tmp_path):
    """Create a sample photo (no real face — integration test for pipeline flow)."""
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    path = tmp_path / "photo.jpg"
    img.save(path, quality=95)
    return str(path)


def test_protect_creates_output(sample_photo, tmp_path):
    output = str(tmp_path / "protected.jpg")
    result = protect_image(sample_photo, output, mode="fast", skip_no_face=True)
    assert Path(output).exists()
    assert result["success"] is True


def test_protect_thorough_mode(sample_photo, tmp_path):
    output = str(tmp_path / "protected.jpg")
    result = protect_image(sample_photo, output, mode="thorough", skip_no_face=True)
    assert Path(output).exists()


def test_protect_returns_metrics(sample_photo, tmp_path):
    output = str(tmp_path / "protected.jpg")
    result = protect_image(sample_photo, output, mode="fast", skip_no_face=True)
    assert "faces_found" in result
    assert "watermark_embedded" in result


def test_protect_no_face_skips_perturbation(sample_photo, tmp_path):
    """When no face is detected, perturbation is skipped but watermark still applied."""
    output = str(tmp_path / "protected.jpg")
    result = protect_image(sample_photo, output, mode="fast", skip_no_face=True)
    assert result["faces_found"] == 0
    assert result["watermark_embedded"] is True
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_integration.py -v`
Expected: FAIL

**Step 3: Implement orchestrator**

```python
"""Main protection pipeline — orchestrates all three layers."""

import time
from pathlib import Path

import torch

from shieldshot.detect.face_detector import FaceDetector
from shieldshot.perturb.pgd import pgd_attack
from shieldshot.utils.image import load_image, save_image, to_tensor, to_pil
from shieldshot.utils.quality import check_quality
from shieldshot.watermark.encoder import WatermarkEncoder
from shieldshot.watermark.decoder import WatermarkDecoder
from shieldshot.watermark.payload import encode_payload, PAYLOAD_BITS


def protect_image(
    input_path: str,
    output_path: str,
    mode: str = "fast",
    user_id: str = "default",
    skip_no_face: bool = False,
    sign_c2pa: bool = False,
    keys_dir: str | None = None,
) -> dict:
    """Apply full protection pipeline to an image.

    Args:
        input_path: Path to input image.
        output_path: Path to save protected image.
        mode: "fast" (generator) or "thorough" (PGD).
        user_id: User identifier for watermark payload.
        skip_no_face: If True, continue even if no faces found.
        sign_c2pa: If True, apply C2PA signing.
        keys_dir: Path to C2PA keys directory.

    Returns:
        Dict with metrics: success, faces_found, watermark_embedded, ssim, time_seconds.
    """
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    pil_image = load_image(input_path)
    tensor = to_tensor(pil_image).to(device)

    # Step 1: Face detection
    detector = FaceDetector()
    faces = detector.detect(pil_image)

    result = {
        "success": True,
        "faces_found": len(faces),
        "perturbation_applied": False,
        "watermark_embedded": False,
        "c2pa_signed": False,
        "ssim": 1.0,
    }

    # Step 2: Adversarial perturbation (face regions)
    if len(faces) > 0:
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            # Clamp to image bounds
            _, _, H, W = tensor.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            face_crop = tensor[:, :, y1:y2, x1:x2]

            if mode == "thorough":
                perturbed_crop = pgd_attack(face_crop, num_steps=100)
            else:
                # Fast mode — use generator if available, fallback to light PGD
                try:
                    from shieldshot.perturb.generator import PerturbationGenerator
                    gen = PerturbationGenerator().to(device)
                    # Try loading trained weights
                    weights_path = Path.home() / ".shieldshot" / "models" / "generator.pt"
                    if weights_path.exists():
                        gen.load_state_dict(torch.load(weights_path, weights_only=True))
                    gen.eval()
                    with torch.no_grad():
                        perturbed_crop = gen(face_crop)
                except Exception:
                    # Fallback to light PGD
                    perturbed_crop = pgd_attack(face_crop, num_steps=20)

            tensor[:, :, y1:y2, x1:x2] = perturbed_crop
        result["perturbation_applied"] = True
    elif not skip_no_face:
        result["success"] = False
        result["reason"] = "No faces detected"
        return result

    # Step 3: Invisible watermark (full image)
    encoder = WatermarkEncoder(payload_bits=PAYLOAD_BITS).to(device)
    # Try loading trained weights
    wm_weights = Path.home() / ".shieldshot" / "models" / "watermark_encoder.pt"
    if wm_weights.exists():
        encoder.load_state_dict(torch.load(wm_weights, weights_only=True))
    encoder.eval()

    payload_bits = encode_payload(user_id=user_id, timestamp=int(time.time()))
    payload_tensor = torch.tensor(payload_bits, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        watermarked = encoder(tensor, payload_tensor).clamp(0, 1)
    result["watermark_embedded"] = True

    # Quality check
    original_tensor = to_tensor(pil_image).to(device)
    passed, metrics = check_quality(original_tensor, watermarked)
    result["ssim"] = metrics["ssim"]

    if not passed:
        result["success"] = False
        result["reason"] = f"Quality gate failed: SSIM={metrics['ssim']:.4f}"
        return result

    # Save
    output_pil = to_pil(watermarked)
    save_image(output_pil, output_path)

    # Step 4: C2PA signing (optional)
    if sign_c2pa:
        try:
            from shieldshot.provenance.c2pa import sign_image
            signed_output = output_path + ".tmp"
            sign_image(output_path, signed_output, keys_dir=keys_dir)
            Path(signed_output).rename(output_path)
            result["c2pa_signed"] = True
        except Exception as e:
            result["c2pa_warning"] = f"C2PA signing failed: {e}"

    result["time_seconds"] = round(time.time() - start, 2)
    return result
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_integration.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/shieldshot/protect.py tests/test_integration.py
git commit -m "feat: add protection pipeline orchestrator"
```

---

### Task 14: Wire Up the CLI

**Files:**
- Modify: `src/shieldshot/cli.py`

**Step 1: Write CLI integration test**

Add to `tests/test_integration.py`:

```python
from click.testing import CliRunner
from shieldshot.cli import main


def test_cli_protect(sample_photo, tmp_path):
    runner = CliRunner()
    output = str(tmp_path / "cli_protected.jpg")
    result = runner.invoke(main, ["protect", sample_photo, "-o", output])
    assert result.exit_code == 0
    assert Path(output).exists()


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_integration.py::test_cli_protect -v`
Expected: FAIL (CLI doesn't call protect_image yet)

**Step 3: Update CLI to call protect_image**

Replace `src/shieldshot/cli.py` with:

```python
"""ShieldShot CLI."""

from pathlib import Path

import click

from shieldshot import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """ShieldShot — Protect your photos from deepfake misuse."""
    pass


@main.command()
def init():
    """Set up ShieldShot (download models, generate keys)."""
    from shieldshot.provenance.c2pa import init_keys
    keys_dir = init_keys()
    click.echo(f"Keys generated at {keys_dir}")
    click.echo("ShieldShot initialized.")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output path for protected image.")
@click.option("--mode", type=click.Choice(["fast", "thorough"]), default="fast",
              help="Protection mode: fast (generator) or thorough (PGD).")
@click.option("--sign", is_flag=True, help="Apply C2PA provenance signing.")
@click.option("--user-id", default="default", help="User ID for watermark payload.")
def protect(input_path, output_path, mode, sign, user_id):
    """Protect a photo from deepfake misuse."""
    from shieldshot.protect import protect_image

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_protected{p.suffix}")

    click.echo(f"Protecting {input_path} (mode={mode})...")

    result = protect_image(
        input_path,
        output_path,
        mode=mode,
        user_id=user_id,
        skip_no_face=True,
        sign_c2pa=sign,
    )

    if result["success"]:
        click.echo(f"Protected image saved to {output_path}")
        click.echo(f"  Faces found: {result['faces_found']}")
        click.echo(f"  SSIM: {result['ssim']:.4f}")
        click.echo(f"  Time: {result['time_seconds']}s")
    else:
        click.echo(f"Protection failed: {result.get('reason', 'unknown')}", err=True)
        raise SystemExit(1)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def verify(input_path):
    """Verify provenance of a protected photo."""
    from shieldshot.provenance.c2pa import verify_image

    result = verify_image(input_path)
    if result["valid"]:
        click.echo(f"Valid C2PA manifest found")
        click.echo(f"  Software: {result.get('software', 'unknown')}")
    else:
        click.echo(f"No valid C2PA manifest: {result.get('reason', 'not found')}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def extract(input_path):
    """Extract watermark from a protected photo."""
    import torch
    from shieldshot.utils.image import load_image, to_tensor
    from shieldshot.watermark.decoder import WatermarkDecoder
    from shieldshot.watermark.payload import decode_payload, PAYLOAD_BITS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = WatermarkDecoder(payload_bits=PAYLOAD_BITS).to(device)

    # Try loading trained weights
    weights_path = Path.home() / ".shieldshot" / "models" / "watermark_decoder.pt"
    if weights_path.exists():
        decoder.load_state_dict(torch.load(weights_path, weights_only=True))

    decoder.eval()
    img = load_image(input_path)
    tensor = to_tensor(img).to(device)

    with torch.no_grad():
        logits = decoder(tensor)
    bits = (logits > 0).int().squeeze(0).tolist()

    result = decode_payload(bits)
    if result["valid"]:
        click.echo(f"Watermark extracted successfully")
        click.echo(f"  User ID hash: {result['user_id_hash']}")
        click.echo(f"  Timestamp: {result['timestamp']}")
    else:
        click.echo("No valid watermark found (model may need training)")
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_integration.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/shieldshot/cli.py tests/test_integration.py
git commit -m "feat: wire CLI to protection pipeline"
```

---

### Task 15: Model Download Script

**Files:**
- Create: `models/download_models.py`

**Step 1: Implement download script**

```python
"""Download pretrained model weights for ShieldShot."""

import argparse
from pathlib import Path
import subprocess
import sys


MODELS_DIR = Path.home() / ".shieldshot" / "models"


def download_insightface():
    """Ensure InsightFace models are downloaded."""
    print("Checking InsightFace models...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  InsightFace models ready.")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Run: pip install insightface onnxruntime")


def download_facenet():
    """Ensure FaceNet model is downloaded."""
    print("Checking FaceNet model...")
    try:
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained="vggface2")
        print("  FaceNet model ready.")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Run: pip install facenet-pytorch")


def setup_dirs():
    """Create model directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {MODELS_DIR}")


def main():
    setup_dirs()
    download_insightface()
    download_facenet()
    print("\nAll models ready.")


if __name__ == "__main__":
    main()
```

**Step 2: Test it runs**

Run: `cd /Users/varma/shieldshot && python3 models/download_models.py`
Expected: prints model status

**Step 3: Commit**

```bash
git add models/download_models.py
git commit -m "feat: add model download script"
```

---

### Task 16: Final Integration Test & README

**Files:**
- Create: `tests/test_full_pipeline.py`

**Step 1: Write end-to-end smoke test**

```python
"""End-to-end smoke test for the full ShieldShot pipeline."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from click.testing import CliRunner

from shieldshot.cli import main


@pytest.fixture
def photo_dir(tmp_path):
    """Create a directory with sample photos."""
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(tmp_path / f"photo_{i}.jpg", quality=95)
    return tmp_path


def test_cli_init():
    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0
    assert "initialized" in result.output.lower()


def test_protect_single_photo(photo_dir, tmp_path):
    runner = CliRunner()
    input_path = str(photo_dir / "photo_0.jpg")
    output_path = str(tmp_path / "protected.jpg")
    result = runner.invoke(main, ["protect", input_path, "-o", output_path])
    assert result.exit_code == 0
    assert Path(output_path).exists()


def test_protect_and_extract(photo_dir, tmp_path):
    runner = CliRunner()
    input_path = str(photo_dir / "photo_0.jpg")
    output_path = str(tmp_path / "protected.jpg")

    # Protect
    result = runner.invoke(main, ["protect", input_path, "-o", output_path])
    assert result.exit_code == 0

    # Extract (won't find valid watermark without trained model, but shouldn't crash)
    result = runner.invoke(main, ["extract", output_path])
    assert result.exit_code == 0
```

**Step 2: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS

**Step 3: Commit**

```bash
git add tests/test_full_pipeline.py
git commit -m "feat: add end-to-end smoke tests"
```

---

## Implementation Order Summary

| Task | Component | Depends On |
|------|-----------|------------|
| 1 | Project scaffolding + CLI skeleton | — |
| 2 | Image utilities | 1 |
| 3 | Quality gate (SSIM) | 2 |
| 4 | Face detection | 1 |
| 5 | Target model loaders | 1 |
| 6 | Multi-model loss functions | 5 |
| 7 | PGD adversarial perturbation | 5, 6 |
| 8 | Watermark payload encoding | 1 |
| 9 | Watermark encoder/decoder networks | 8 |
| 10 | Watermark training + augmentations | 2, 9 |
| 11 | Perturbation generator + training | 7 |
| 12 | C2PA provenance layer | 1 |
| 13 | Orchestrator (protect.py) | 2, 3, 4, 7, 9, 12 |
| 14 | Wire up CLI | 13 |
| 15 | Model download script | 4, 5 |
| 16 | End-to-end smoke tests | 14 |

## Parallel Tracks

Tasks can be parallelized into three tracks:

- **Track A (Perturbation):** 4 → 5 → 6 → 7 → 11
- **Track B (Watermark):** 8 → 9 → 10
- **Track C (Provenance):** 12

All tracks converge at Task 13 (orchestrator).

"""Integration tests for the full protection pipeline."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from click.testing import CliRunner
from shieldshot.cli import main
from shieldshot.protect import protect_image


@pytest.fixture
def sample_photo(tmp_path):
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
    output = str(tmp_path / "protected.jpg")
    result = protect_image(sample_photo, output, mode="fast", skip_no_face=True)
    assert result["faces_found"] == 0
    assert result["watermark_embedded"] is True


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

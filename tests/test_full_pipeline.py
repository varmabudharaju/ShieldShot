"""End-to-end smoke test for the full ShieldShot pipeline."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from click.testing import CliRunner
from shieldshot.cli import main


@pytest.fixture
def photo_dir(tmp_path):
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

    result = runner.invoke(main, ["protect", input_path, "-o", output_path])
    assert result.exit_code == 0

    result = runner.invoke(main, ["extract", output_path])
    assert result.exit_code == 0

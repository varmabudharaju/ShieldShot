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
    img = Image.fromarray(np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8))
    faces = detector.detect(img)
    for face in faces:
        assert "bbox" in face
        assert "confidence" in face
        assert len(face["bbox"]) == 4

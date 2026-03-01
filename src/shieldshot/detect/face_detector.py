"""Face detection using InsightFace/RetinaFace."""

import numpy as np
from PIL import Image


class FaceDetector:
    """Detects faces in images using RetinaFace via InsightFace."""

    def __init__(self, min_confidence: float = 0.5):
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        self._min_confidence = min_confidence

    def detect(self, image: Image.Image) -> list[dict]:
        """Detect faces in a PIL Image.

        Returns list of dicts with 'bbox' [x1, y1, x2, y2] and 'confidence'.
        """
        arr = np.array(image)
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

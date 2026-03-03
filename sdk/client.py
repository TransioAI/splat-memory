"""Splat Memory Python SDK.

Usage:
    from sdk.client import SplatMemory

    client = SplatMemory("172.31.39.38")
    scene = client.snap("IMG_1234.heic")
    print(scene.objects)

    answer = client.ask("How far is the chair from the wall?")
    print(answer)

    client.save_annotated(scene.scene_id, "annotated.jpg")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass
class SceneObject:
    """A detected object with 3D position and dimensions."""

    id: int
    label: str
    confidence: float
    bbox: list[float]
    position_m: list[float]
    dimensions_m: list[float]
    distance_m: float

    @property
    def x(self) -> float:
        return self.position_m[0]

    @property
    def y(self) -> float:
        return self.position_m[1]

    @property
    def z(self) -> float:
        return self.position_m[2]

    @property
    def width(self) -> float:
        return self.dimensions_m[0]

    @property
    def height(self) -> float:
        return self.dimensions_m[1]

    @property
    def depth(self) -> float:
        return self.dimensions_m[2]

    def __repr__(self) -> str:
        return (
            f"SceneObject(id={self.id}, label='{self.label}', "
            f"conf={self.confidence:.2f}, dist={self.distance_m:.2f}m)"
        )


@dataclass
class SceneRelation:
    """A spatial relation between two objects."""

    subject_id: int
    subject_label: str
    predicate: str
    object_id: int
    object_label: str
    distance_m: float

    def __repr__(self) -> str:
        return (
            f"{self.subject_label} {self.predicate} "
            f"{self.object_label} ({self.distance_m:.2f}m)"
        )


@dataclass
class Calibration:
    """Camera calibration info for the scene."""

    fov_degrees: float
    intrinsics_source: str
    scale_factor: float
    reference_object: str | None
    image_width: int
    image_height: int


@dataclass
class Scene:
    """Result of analyzing an image."""

    scene_id: str
    intrinsics_source: str
    objects: list[SceneObject]
    relations: list[SceneRelation]
    calibration: Calibration
    raw: dict = field(repr=False)

    def find(self, label: str) -> list[SceneObject]:
        """Find all objects matching a label (case-insensitive)."""
        label_lower = label.lower()
        return [o for o in self.objects if label_lower in o.label.lower()]

    def nearest(self) -> SceneObject | None:
        """Return the object closest to the camera."""
        if not self.objects:
            return None
        return min(self.objects, key=lambda o: o.distance_m)

    def farthest(self) -> SceneObject | None:
        """Return the object farthest from the camera."""
        if not self.objects:
            return None
        return max(self.objects, key=lambda o: o.distance_m)

    def __repr__(self) -> str:
        return (
            f"Scene(id='{self.scene_id[:8]}...', "
            f"objects={len(self.objects)}, "
            f"relations={len(self.relations)}, "
            f"fov={self.calibration.fov_degrees:.1f}° "
            f"[{self.intrinsics_source}])"
        )


class SplatMemoryError(Exception):
    """Error from the Splat Memory API."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}")


class SplatMemory:
    """Client for the Splat Memory API.

    Parameters
    ----------
    host:
        Server IP address or hostname.
    port:
        Server port (default: 8000).
    """

    def __init__(self, host: str, port: int = 8000) -> None:
        self.base_url = f"http://{host}:{port}"
        self._last_scene_id: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> bytes:
        """Make an HTTP request and return the response body."""
        url = f"{self.base_url}{path}"
        req = Request(url, method=method, **kwargs)
        try:
            with urlopen(req) as resp:
                return resp.read()
        except HTTPError as e:
            body = e.read().decode()
            try:
                detail = json.loads(body).get("detail", body)
            except json.JSONDecodeError:
                detail = body
            raise SplatMemoryError(e.code, detail) from None

    def _get(self, path: str) -> bytes:
        return self._request("GET", path)

    def _get_json(self, path: str) -> dict:
        return json.loads(self._get(path))

    def _post_json(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        return json.loads(
            self._request(
                "POST", path,
                data=body,
                headers={"Content-Type": "application/json"},
            )
        )

    def _post_multipart(self, path: str, filepath: str | Path) -> dict:
        """Upload a file using multipart/form-data."""
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")

        boundary = "----SplatMemoryBoundary"
        filename = filepath.name

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n"
            f"\r\n"
        ).encode()
        body += filepath.read_bytes()
        body += f"\r\n--{boundary}--\r\n".encode()

        return json.loads(
            self._request(
                "POST", path,
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
        )

    @staticmethod
    def _parse_scene(data: dict) -> Scene:
        """Parse an API response into a Scene object."""
        sg = data["scene_graph"]
        cal_data = sg["calibration"]

        objects = [
            SceneObject(
                id=o["id"],
                label=o["label"],
                confidence=o["confidence"],
                bbox=o["bbox"],
                position_m=o["position_m"],
                dimensions_m=o["dimensions_m"],
                distance_m=o["distance_m"],
            )
            for o in sg["objects"]
        ]

        relations = [
            SceneRelation(
                subject_id=r["subject_id"],
                subject_label=r["subject_label"],
                predicate=r["predicate"],
                object_id=r["object_id"],
                object_label=r["object_label"],
                distance_m=r["distance_m"],
            )
            for r in sg["relations"]
        ]

        calibration = Calibration(
            fov_degrees=cal_data["fov_degrees"],
            intrinsics_source=cal_data["intrinsics_source"],
            scale_factor=cal_data["scale_factor"],
            reference_object=cal_data.get("reference_object"),
            image_width=cal_data["image_width"],
            image_height=cal_data["image_height"],
        )

        return Scene(
            scene_id=data["scene_id"],
            intrinsics_source=data["intrinsics_source"],
            objects=objects,
            relations=relations,
            calibration=calibration,
            raw=data,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def snap(self, image_path: str | Path) -> Scene:
        """Upload an image and get a 3D scene graph.

        This is the simplest way to analyze an image. EXIF metadata
        is auto-extracted for accurate FOV.

        Parameters
        ----------
        image_path:
            Path to the image file (JPEG, PNG, HEIC/HEIF).

        Returns
        -------
        Scene
            Parsed scene with objects, relations, and calibration.
        """
        data = self._post_multipart("/snap", image_path)
        scene = self._parse_scene(data)
        self._last_scene_id = scene.scene_id
        return scene

    def analyze(
        self,
        image_path: str | Path,
        detect: list[str] | None = None,
        fov_degrees: float | None = None,
        focal_length_35mm: float | None = None,
        use_gemini_tagger: bool = False,
    ) -> Scene:
        """Upload an image with additional options.

        Parameters
        ----------
        image_path:
            Path to the image file.
        detect:
            Additional object categories to detect (merged with
            auto-discovered tags).
        fov_degrees:
            Horizontal FOV override in degrees.
        focal_length_35mm:
            35mm-equivalent focal length in mm (converted to FOV).
        use_gemini_tagger:
            When True, use Gemini 2.5 Flash for tagging instead of
            RAM++ + Claude filter.
        """
        filepath = Path(image_path)
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")

        boundary = "----SplatMemoryBoundary"
        parts = []

        # File part
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filepath.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n"
            f"\r\n"
        )

        # Optional form fields
        form_fields = {}
        if detect:
            form_fields["detect"] = ",".join(detect)
        if fov_degrees is not None:
            form_fields["fov_degrees"] = str(fov_degrees)
        if focal_length_35mm is not None:
            form_fields["focal_length_35mm"] = str(focal_length_35mm)
        if use_gemini_tagger:
            form_fields["use_gemini_tagger"] = "true"

        field_parts = ""
        for name, value in form_fields.items():
            field_parts += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n"
            )

        body = parts[0].encode()
        body += filepath.read_bytes()
        body += f"\r\n{field_parts}--{boundary}--\r\n".encode()

        data = json.loads(
            self._request(
                "POST", "/analyze",
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
        )
        scene = self._parse_scene(data)
        self._last_scene_id = scene.scene_id
        return scene

    def ask(self, question: str, scene_id: str | None = None) -> str:
        """Ask a spatial question about a scene.

        Parameters
        ----------
        question:
            Natural-language spatial question.
        scene_id:
            Scene UUID. If omitted, uses the last analyzed scene.

        Returns
        -------
        str
            The answer.
        """
        sid = scene_id or self._last_scene_id
        payload: dict = {"question": question}
        if sid:
            payload["scene_id"] = sid
        data = self._post_json("/ask", payload)
        return data["answer"]

    # ------------------------------------------------------------------
    # Scene outputs
    # ------------------------------------------------------------------

    def _scene_path(self, scene_id: str | None) -> str:
        sid = scene_id or self._last_scene_id
        if not sid:
            raise ValueError("No scene_id — call snap() or analyze() first.")
        return f"/scene/{sid}"

    def save_detections(self, path: str, scene_id: str | None = None) -> None:
        """Save detection boxes image to a file."""
        data = self._get(f"{self._scene_path(scene_id)}/detections")
        Path(path).write_bytes(data)

    def save_masks(self, path: str, scene_id: str | None = None) -> None:
        """Save segmentation masks image to a file."""
        data = self._get(f"{self._scene_path(scene_id)}/masks")
        Path(path).write_bytes(data)

    def save_depth(self, path: str, scene_id: str | None = None) -> None:
        """Save depth heatmap image to a file."""
        data = self._get(f"{self._scene_path(scene_id)}/depth")
        Path(path).write_bytes(data)

    def save_annotated(self, path: str, scene_id: str | None = None) -> None:
        """Save fully annotated image to a file."""
        data = self._get(f"{self._scene_path(scene_id)}/annotated")
        Path(path).write_bytes(data)

    def save_pointcloud(self, path: str, scene_id: str | None = None) -> None:
        """Save interactive 3D point cloud HTML to a file."""
        data = self._get(f"{self._scene_path(scene_id)}/pointcloud")
        Path(path).write_bytes(data)

    def get_tags(self, scene_id: str | None = None) -> dict:
        """Get raw and filtered tags used for detection."""
        return self._get_json(f"{self._scene_path(scene_id)}/tags")

    def get_objects(self, scene_id: str | None = None) -> list[dict]:
        """Get post-NMS detection data (label, confidence, bbox)."""
        return self._get_json(f"{self._scene_path(scene_id)}/objects")

    def get_graph(self, scene_id: str | None = None) -> dict:
        """Get the full scene graph as JSON."""
        return self._get_json(f"{self._scene_path(scene_id)}/graph")

    def get_graph_text(self, scene_id: str | None = None) -> str:
        """Get the scene graph as human-readable text."""
        return self._get(f"{self._scene_path(scene_id)}/graph/text").decode()

    def health(self) -> dict:
        """Check server health."""
        return self._get_json("/health")

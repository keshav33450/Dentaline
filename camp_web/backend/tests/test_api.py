from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

import backend.server as server


class Collection:
    def __init__(self, patient=None):
        self.patient = patient
        self.inserted = []

    async def find_one(self, *_args, **_kwargs):
        return self.patient

    async def insert_one(self, document):
        document["_id"] = ObjectId()
        self.inserted.append(document)
        return SimpleNamespace(inserted_id=document["_id"])


class FakeDatabase:
    def __init__(self, doctor_id):
        self.patients = Collection({"_id": ObjectId(), "patient_id": "DX-2026-00001", "doctor_id": doctor_id,
                                    "name": "Sample patient", "age": 32, "phone": "+10000000000", "email": "patient@example.com"})
        self.screenings = Collection()


@pytest.fixture
def api(monkeypatch):
    doctor_id = ObjectId()
    database = FakeDatabase(doctor_id)
    doctor = {"_id": doctor_id, "name": "Test Doctor", "email": "doctor@example.com"}
    server.app.dependency_overrides[server.current_doctor] = lambda: doctor
    server.app.dependency_overrides[server.get_db] = lambda: database
    monkeypatch.setenv("MONGODB_URI", "")
    with TestClient(server.app) as client:
        yield client, database
    server.app.dependency_overrides.clear()


def valid_jpeg():
    ok, image = cv2.imencode(".jpg", np.full((320, 420, 3), 190, dtype=np.uint8))
    assert ok
    return image.tobytes()


def test_health_reports_each_downstream_status(api, monkeypatch):
    async def health():
        return {"caries": {"status": "available"}, "tooth": {"status": "unavailable"}, "gingivitis": {"status": "available"}}

    monkeypatch.setattr(server, "check_model_health", health)
    client, _ = api
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["models"]["tooth"]["status"] == "unavailable"


def test_screen_continues_when_one_model_is_unavailable(api, monkeypatch):
    async def partial(_image, _filename):
        return {
            "caries": {"status": "available", "data": {"caries": [{"severity": "mild", "confidence": 0.7, "box": [10, 12, 80, 90]}]}},
            "tooth": {"status": "unavailable", "reason": "request_failed"},
            "gingivitis": {"status": "available", "data": {"gingivitis_detected": False, "max_confidence": 0.0, "regions": []}},
        }

    monkeypatch.setattr(server, "run_models", partial)
    client, db = api
    response = client.post("/api/screen", data={"patient_id": "DX-2026-00001"}, files={"file": ("photo.jpg", valid_jpeg(), "image/jpeg")})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model_status"] == {"caries": "available", "tooth": "unavailable", "gingivitis": "available"}
    assert result["tooth_count"] is None
    assert result["risk_level"] == "moderate"
    assert len(db.screenings.inserted) == 1
    assert "annotated_image_b64" in result


def test_screen_rejects_invalid_and_oversized_images(api):
    client, _ = api
    invalid = client.post("/api/screen", data={"patient_id": "DX-2026-00001"}, files={"file": ("photo.jpg", b"not an image", "image/jpeg")})
    assert invalid.status_code == 415
    oversized = client.post("/api/screen", data={"patient_id": "DX-2026-00001"}, files={"file": ("photo.jpg", b"x" * (server.MAX_IMAGE_BYTES + 1), "image/jpeg")})
    assert oversized.status_code == 413


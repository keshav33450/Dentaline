import asyncio

import httpx

from backend.server import _fetch_model, _hash_password, _verify_password, fuse_results


def model_results(caries=None, tooth=None, gingivitis=None):
    return {
        "caries": caries or {"status": "available", "data": {"caries": []}},
        "tooth": tooth or {"status": "available", "data": {"n_teeth": 3, "counts": {"molar": 3}, "teeth": []}},
        "gingivitis": gingivitis or {"status": "available", "data": {"gingivitis_detected": False, "max_confidence": 0.0, "regions": []}},
    }


def test_no_findings_are_low_only_when_all_relevant_models_are_available():
    assert fuse_results(model_results())["risk_level"] == "low"
    unavailable = model_results(gingivitis={"status": "unavailable"})
    result = fuse_results(unavailable)
    assert result["risk_level"] == "incomplete"
    assert result["gingivitis"]["found"] is None


def test_risk_rules_use_caries_severity_and_gingivitis_confidence():
    moderate = model_results(caries={"status": "available", "data": {"caries": [{"severity": "moderate", "confidence": 0.72}]}})
    assert fuse_results(moderate)["risk_level"] == "high"
    gum = model_results(gingivitis={"status": "available", "data": {"gingivitis_detected": True, "max_confidence": 0.61, "regions": [{"confidence": 0.61}]}})
    assert fuse_results(gum)["risk_level"] == "high"
    mild_gum = model_results(gingivitis={"status": "available", "data": {"gingivitis_detected": True, "max_confidence": 0.4, "regions": [{"confidence": 0.4}]}})
    assert fuse_results(mild_gum)["risk_level"] == "moderate"


def test_unavailable_and_malformed_model_responses_are_isolated():
    async def scenario():
        async def handler(request):
            return httpx.Response(200, json={"wrong": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await _fetch_model(client, "tooth", b"image", "photo.jpg")
        assert result["status"] == "unavailable"
        assert result["reason"] == "request_failed"

    asyncio.run(scenario())
    one_down = fuse_results(model_results(tooth={"status": "unavailable"}))
    assert one_down["tooth_count"] is None
    assert one_down["caries"]["status"] == "available"
    assert one_down["gingivitis"]["status"] == "available"


def test_passwords_are_salted_and_never_stored_as_plaintext():
    hashed = _hash_password("DentalX example password")
    assert hashed != "DentalX example password"
    assert _verify_password("DentalX example password", hashed)
    assert not _verify_password("wrong password", hashed)


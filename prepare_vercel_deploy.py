"""Build a self-contained Vercel Services deployment bundle for DentalX Camp."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    "caries": {"source": ROOT / "model1_occlusal", "prefix": "/models/caries", "weights": ["occlusal_caries_det.pt", "occlusal_severity.pt"]},
    "tooth": {"source": ROOT / "model2_tooth_type", "prefix": "/models/tooth", "weights": ["best.pt"]},
    "gingivitis": {"source": ROOT / "model3_gingival", "prefix": "/models/gingivitis", "weights": ["best.pt"]},
}
MODEL_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
torchvision==0.20.1+cpu
fastapi>=0.110
python-multipart>=0.0.9
ultralytics==8.4.173
opencv-python-headless==4.11.0.86
numpy==1.26.4
"""
BACKEND_REQUIREMENTS = """fastapi>=0.115,<1
uvicorn[standard]>=0.30,<1
httpx>=0.27,<1
pymongo[srv]>=4.10,<5
opencv-python-headless==4.11.0.86
numpy==1.26.4
python-multipart>=0.0.9
python-dotenv>=1,<2
reportlab>=4,<5
"""


def copy_model(name: str, spec: dict, target: Path) -> None:
    code = spec["source"] / "code"
    if not code.is_dir():
        raise FileNotFoundError(f"Model code folder not found: {code}")
    shutil.copytree(code, target, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "project_location.txt", ".venv", "venv",
        "outputs", "runs", "data", "datasets", "notebooks", "tests",
    ))
    weights_dir = target / "models"
    weights_dir.mkdir(exist_ok=True)
    for filename in spec["weights"]:
        candidates = [spec["source"] / "models" / filename, code / "models" / filename]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Required {name} model weights not found: {filename}")
        shutil.copy2(source, weights_dir / filename)
    (target / "requirements.txt").write_text(MODEL_REQUIREMENTS, encoding="utf-8")
    entry = ("from fastapi import FastAPI\nfrom api.main import app as model_api\n\n"
             f"app = FastAPI(title='DentalX {name} API')\napp.mount({spec['prefix']!r}, model_api)\n")
    (target / "main.py").write_text(entry, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Empty output folder (defaults to a new system-temp folder)")
    args = parser.parse_args()
    if args.out:
        output = args.out.expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"Output folder must be empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = Path(tempfile.mkdtemp(prefix="dentalx-vercel-"))

    web_source = ROOT / "camp_web" / "client"
    shutil.copytree(web_source, output / "client", ignore=shutil.ignore_patterns("node_modules", "dist", ".vite"))
    backend_source = ROOT / "camp_web" / "backend"
    shutil.copytree(backend_source, output / "backend", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", ".pytest_cache"))
    shutil.copy2(ROOT / "camp_web" / ".env.example", output / "backend" / ".env.example")
    (output / "backend" / "requirements.txt").write_text(BACKEND_REQUIREMENTS, encoding="utf-8")

    services = {
        "web": {"root": "client", "framework": "vite"},
        "backend": {"root": "backend", "framework": "fastapi", "entrypoint": "server:app"},
    }
    rewrites = [{"source": "/api/(.*)", "destination": {"service": "backend"}}]
    for name, spec in MODELS.items():
        copy_model(name, spec, output / name)
        services[name] = {"root": name, "framework": "fastapi", "entrypoint": "main:app"}
        rewrites.append({"source": spec["prefix"] + "/(.*)", "destination": {"service": name}})
    rewrites.append({"source": "/(.*)", "destination": {"service": "web"}})
    (output / "vercel.json").write_text(json.dumps({"services": services, "rewrites": rewrites}, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "DentalX Camp: React web, FastAPI orchestration backend, and three FastAPI model services.\n\n"
        "Enable Vercel Services, then from this folder run vercel login and vercel deploy --prod.\n"
        "Configure APP_ENV=production, MONGODB_URI, DATABASE_NAME, JWT_SECRET (32+ random characters),\n"
        "CARIES_API_URL, TOOTH_API_URL, GINGIVITIS_API_URL and optional email settings in Vercel.\n"
        "The URLs must point to /models/caries/analyze, /models/tooth/analyze, and\n"
        "/models/gingivitis/analyze on the deployed application domain. Configure MODEL_API_KEY\n"
        "on the deployment if model API authentication is enabled.\n\n"
        "Health: /api/health and /models/{caries,tooth,gingivitis}/health\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

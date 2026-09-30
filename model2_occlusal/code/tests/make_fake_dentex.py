#!/usr/bin/env python3
"""Build tiny fake DENTEX-shaped zips (same folder/JSON layout) to smoke-test the pipeline offline."""
import json
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 480
DIAG = [{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"},
        {"id": 2, "name": "Periapical Lesion"}, {"id": 3, "name": "Deep Caries"}]
C1 = [{"id": i, "name": str(i + 1)} for i in range(4)]
C2 = [{"id": i, "name": str(i + 1)} for i in range(8)]


def tooth_boxes():
    """16 upper + 16 lower teeth laid out like a panoramic (patient's right = image left)."""
    out = []
    tw = W // 18
    for row, (qs, y0) in enumerate((((1, 2), 60), ((4, 3), 250))):
        for k in range(16):
            q = qs[0] if k < 8 else qs[1]
            n = 8 - k if k < 8 else k - 7
            x0 = tw + k * tw
            out.append((q, n, [x0 + 3, y0, tw - 6, 170]))
    return out


def render(seed, diseased):
    rng = random.Random(seed)
    img = np.full((H, W), 40, np.uint8)
    img = cv2.GaussianBlur((np.random.default_rng(seed).random((H, W)) * 30 + 30).astype(np.uint8), (5, 5), 0)
    boxes = tooth_boxes()
    for q, n, (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), 150 + n * 10, -1)
        cv2.putText(img, str(n), (x + 8, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 250, 2)
    anns = []
    for idx in diseased:
        q, n, (x, y, w, h) = boxes[idx]
        d = rng.randrange(4)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(img, (cx, cy), 10 + d * 3, 20, -1)
        anns.append((q, n, [x, y, w, h], d))
    return img, boxes, anns


def coco(kind, n_imgs, prefix, seed0):
    images, anns, pix = [], [], {}
    aid = 0
    for i in range(n_imgs):
        fn = f"{prefix}_{i}.png"
        rng = random.Random(seed0 + i)
        diseased = rng.sample(range(32), 3)
        img, boxes, dis = render(seed0 + i, diseased)
        pix[fn] = img
        images.append({"id": i + 1, "file_name": fn, "width": W, "height": H})
        if kind == "enumeration":
            for q, n, b in boxes:
                aid += 1
                anns.append({"id": aid, "image_id": i + 1, "bbox": b, "area": b[2] * b[3], "iscrowd": 0,
                             "category_id_1": q - 1, "category_id_2": n - 1})
        else:
            for q, n, b, d in dis:
                aid += 1
                anns.append({"id": aid, "image_id": i + 1, "bbox": b, "area": b[2] * b[3], "iscrowd": 0,
                             "category_id_1": q - 1, "category_id_2": n - 1, "category_id_3": d})
    j = {"images": images, "annotations": anns, "categories_1": C1, "categories_2": C2}
    if kind == "diagnosis":
        j["categories_3"] = DIAG
    return j, pix


def add(zf, folder, jname, j, pix):
    zf.writestr(f"{folder}/{jname}", json.dumps(j))
    for fn, img in pix.items():
        zf.writestr(f"{folder}/xrays/{fn}", cv2.imencode(".png", img)[1].tobytes())


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    enum_j, enum_p = coco("enumeration", 30, "train", 0)
    dis_j, dis_p = coco("diagnosis", 40, "train", 1000)       # same file names, different images
    val_j, val_p = coco("diagnosis", 8, "val", 5000)
    test_j, test_p = coco("diagnosis", 10, "test", 7000)
    # leak one test image into the enumeration set to check hash de-duplication
    enum_p["train_0.png"] = test_p["test_0.png"]
    with zipfile.ZipFile(out / "training_data.zip", "w") as zf:
        add(zf, "training_data/quadrant_enumeration", "train_quadrant_enumeration.json", enum_j, enum_p)
        add(zf, "training_data/quadrant-enumeration-disease", "train_quadrant_enumeration_disease.json", dis_j, dis_p)
    with zipfile.ZipFile(out / "validation_data.zip", "w") as zf:
        for fn, img in val_p.items():
            zf.writestr(f"validation_data/quadrant_enumeration_disease/xrays/{fn}", cv2.imencode(".png", img)[1].tobytes())
    (out / "validation_triple.json").write_text(json.dumps(val_j))
    with zipfile.ZipFile(out / "test_data.zip", "w") as zf:
        add(zf, "test_data/disease", "test_quadrant_enumeration_disease.json", test_j, test_p)
    print("fake DENTEX written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_dentex")

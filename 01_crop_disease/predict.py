"""Diagnose a leaf photo, log it, and check for a nearby outbreak.

Run:  python predict.py --image leaf.jpg --lat 17.38 --lon 78.48
"""
import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

IMG = 224
LOG = Path("../outputs/crop/diagnosis_log.csv")

# Replace/extend with advice from your local agriculture extension office.
TREATMENT = {
    "default": "Isolate affected plants, remove infected leaves, and consult your "
               "agricultural extension officer before applying any chemical."
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def severity(conf: float, label: str) -> str:
    if "healthy" in label.lower():
        return "None"
    return "High" if conf > 0.9 else "Medium" if conf > 0.7 else "Low (uncertain - retake photo)"


def log_diagnosis(label, conf, lat, lon):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"time": datetime.now().isoformat(), "disease": label,
                         "confidence": round(conf, 3), "lat": lat, "lon": lon}])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def check_outbreak(label, lat, lon, radius_km=10, days=7, threshold=5):
    """Alert if >= threshold reports of the same disease nearby in the last N days."""
    if "healthy" in label.lower() or not LOG.exists():
        return False
    df = pd.read_csv(LOG, parse_dates=["time"])
    df = df[(df.disease == label) & (df.time > datetime.now() - timedelta(days=days))]
    near = df[df.apply(lambda r: haversine_km(lat, lon, r.lat, r.lon) <= radius_km, axis=1)]
    return len(near) >= threshold


def main():
    import tensorflow as tf
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model_dir", default="../outputs/crop")
    ap.add_argument("--lat", type=float, default=0.0)
    ap.add_argument("--lon", type=float, default=0.0)
    a = ap.parse_args()

    model = tf.keras.models.load_model(Path(a.model_dir) / "crop_model.keras")
    names = json.loads((Path(a.model_dir) / "class_names.json").read_text())

    img = tf.keras.utils.load_img(a.image, target_size=(IMG, IMG))
    x = np.expand_dims(tf.keras.utils.img_to_array(img), 0)
    probs = model.predict(x, verbose=0)[0]
    top = probs.argsort()[::-1][:3]

    label, conf = names[top[0]], float(probs[top[0]])
    print("Top predictions:")
    for i in top:
        print(f"  {names[i]:45s} {probs[i]:.1%}")
    print("Severity:", severity(conf, label))
    print("Advice  :", TREATMENT.get(label, TREATMENT["default"]))

    log_diagnosis(label, conf, a.lat, a.lon)
    if check_outbreak(label, a.lat, a.lon):
        print("*** OUTBREAK ALERT: multiple nearby reports of this disease this week ***")


if __name__ == "__main__":
    main()

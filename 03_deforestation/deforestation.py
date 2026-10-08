"""Satellite deforestation monitoring demo (synthetic 4-band imagery: R,G,B,NIR).

Pipeline: preprocess -> classify land cover -> compare two dates -> measure
changes -> rank severity -> alert CSV + map.

For real data: export Sentinel-2 (10 m) tiles from Google Earth Engine, load them
into arrays of shape (H, W, 4) and replace make_scene().
Run:  python deforestation.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

OUT = Path("../outputs/deforestation")
FOREST, CLEARED, WATER, BUILT = 0, 1, 2, 3
NAMES = ["forest", "cleared", "water", "built-up"]
SIGNATURE = np.array([[.04, .08, .04, .45],   # R,G,B,NIR reflectance per class
                      [.25, .20, .15, .30],
                      [.03, .05, .08, .02],
                      [.30, .30, .30, .25]])
PIXEL_M = 10                      # Sentinel-2 resolution
ORIGIN = (19.00, 80.00)           # lat, lon of the top-left corner (example)
DEG_PER_PIXEL = PIXEL_M / 111_000


def make_labels(size=128):
    lab = np.full((size, size), FOREST)
    lab[:, 60:68] = WATER                  # river
    lab[100:120, 5:25] = BUILT             # village
    return lab


def render(lab, rng, clouds=False):
    img = SIGNATURE[lab] + rng.normal(0, 0.02, (*lab.shape, 4))
    img = np.clip(img, 0, 1)
    cloud = np.zeros(lab.shape, bool)
    if clouds:
        cloud[10:30, 90:120] = True
        img[cloud] = np.clip(0.75 + rng.normal(0, 0.02, (cloud.sum(), 4)), 0, 1)
    return img, cloud


def features(img):
    r, nir = img[..., 0], img[..., 3]
    ndvi = (nir - r) / (nir + r + 1e-6)
    return np.dstack([img, ndvi]).reshape(-1, 5)


def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)

    # Date 1 (baseline) and Date 2 (new clearing + a cloud patch)
    lab1 = make_labels()
    lab2 = lab1.copy()
    lab2[20:45, 10:30] = CLEARED      # large illegal clearing
    lab2[80:86, 85:91] = CLEARED      # small clearing
    lab2[50:52, 100:102] = CLEARED    # tiny (below min size -> ignored)
    img1, _ = render(lab1, rng)
    img2, cloud2 = render(lab2, rng, clouds=True)

    # Train pixel classifier on an independent labelled scene
    tr_lab = make_labels()
    tr_lab[60:80, 30:50] = CLEARED
    tr_img, _ = render(tr_lab, np.random.default_rng(1))
    clf = RandomForestClassifier(100, n_jobs=-1, random_state=0)
    clf.fit(features(tr_img), tr_lab.ravel())

    pred1 = clf.predict(features(img1)).reshape(lab1.shape)
    pred2 = clf.predict(features(img2)).reshape(lab2.shape)
    mask = ~cloud2  # ignore cloudy pixels
    print(classification_report(lab2[mask].ravel(), pred2[mask].ravel(),
                                target_names=NAMES, zero_division=0))

    # Change detection: forest -> cleared
    change = (pred1 == FOREST) & (pred2 == CLEARED) & mask
    change = ndimage.binary_opening(change)            # remove speckle noise
    blobs, n = ndimage.label(change)
    rows = []
    for i in range(1, n + 1):
        ys, xs = np.where(blobs == i)
        if len(ys) < 4:                                  # min size filter
            continue
        ha = len(ys) * PIXEL_M ** 2 / 10_000
        lat = ORIGIN[0] - ys.mean() * DEG_PER_PIXEL
        lon = ORIGIN[1] + xs.mean() * DEG_PER_PIXEL
        sev = "HIGH" if ha >= 2 else "MEDIUM" if ha >= 0.3 else "LOW"
        rows.append(dict(alert_id=len(rows) + 1, area_ha=round(ha, 2), severity=sev,
                         lat=round(lat, 5), lon=round(lon, 5)))
    alerts = pd.DataFrame(rows).sort_values("area_ha", ascending=False)
    alerts.to_csv(OUT / "alerts.csv", index=False)
    print("\n=== DEFORESTATION ALERTS (send to forest department) ===")
    print(alerts.to_string(index=False))

    # Visual output
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    ax[0].imshow(np.clip(img1[..., [0, 1, 2]] * 4, 0, 1)); ax[0].set_title("Date 1 (RGB)")
    ax[1].imshow(np.clip(img2[..., [0, 1, 2]] * 4, 0, 1)); ax[1].set_title("Date 2 (RGB, with cloud)")
    ax[2].imshow(pred2, cmap="viridis"); ax[2].contour(change, colors="red", linewidths=1)
    ax[2].set_title("Land cover + detected clearing (red)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "deforestation_map.png", dpi=120)
    print("\nSaved:", OUT / "alerts.csv", "and", OUT / "deforestation_map.png")


if __name__ == "__main__":
    main()

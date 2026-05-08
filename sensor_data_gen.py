"""
Phase 1 - Synthetic sensor dataset generator.

PlantVillage has no real environmental sensor readings, so we synthesize
a sensor dataset using class-conditional climate ranges grounded in
plant pathology references. Each row is:

    temperature, humidity, soil_moisture, disease_class

The fusion model later learns to combine these synthetic-but-realistic
sensor signals with the image model's predictions.

Usage:
    venv/Scripts/python.exe -m src.sensor_data_gen
"""

import csv
import random
from pathlib import Path

import numpy as np

from . import config
from .utils import ensure_dir, list_class_names

# ---------------------------------------------------------------------------
# Class-conditional climate profiles.
#
# Design:
#   1. Each CROP has a distinct base climate (temperate fruits cool, tropical
#      fruits warm, etc). This gives the sensor model a strong crop-level
#      signal even when diseases overlap.
#   2. Each DISEASE adds a characteristic SHIFT to the crop base to encode
#      the environmental conditions that favor the pathogen.
#   3. Standard deviations are kept narrow (1.5-2.5) so the clusters are
#      learnable with only 300 samples per class.
#
# Tuple format: (temp_mean, temp_std, hum_mean, hum_std, soil_mean, soil_std)
# ---------------------------------------------------------------------------

# Each crop's "healthy growing" baseline
CROP_BASES = {
    "Apple":       (16.0, 70.0, 55.0),   # temperate, moderate humidity
    "Blueberry":   (19.0, 72.0, 65.0),   # temperate, likes moist acidic soil
    "Cherry":      (18.0, 68.0, 55.0),   # temperate
    "Corn":        (25.0, 65.0, 50.0),   # warm season
    "Grape":       (23.0, 60.0, 45.0),   # Mediterranean, drier
    "Orange":      (27.0, 70.0, 55.0),   # subtropical
    "Peach":       (22.0, 65.0, 55.0),   # warm temperate
    "Pepper":      (26.0, 65.0, 55.0),   # warm
    "Potato":      (20.0, 72.0, 65.0),   # cool, moist soil
    "Raspberry":   (19.0, 72.0, 60.0),   # temperate
    "Soybean":     (26.0, 68.0, 55.0),   # warm season
    "Squash":      (24.0, 65.0, 55.0),   # warm
    "Strawberry":  (20.0, 70.0, 60.0),   # temperate
    "Tomato":      (24.0, 68.0, 55.0),   # warm
}

# Small standard deviations so each class cluster is tight and learnable
T_STD, H_STD, S_STD = 1.8, 3.5, 4.5


def profile(crop: str, dT: float = 0.0, dH: float = 0.0, dS: float = 0.0):
    """Return (t_mean, t_std, h_mean, h_std, s_mean, s_std) for crop + shift."""
    t, h, s = CROP_BASES[crop]
    return (t + dT, T_STD, h + dH, H_STD, s + dS, S_STD)


# Disease-specific shifts from crop baseline. Small offsets, chosen so that
# diseases within a crop differ in at least one dimension.
CLIMATE_PROFILES = {
    # ---- Apple (base 16, 70, 55) ----
    "Apple___Apple_scab":              profile("Apple", dT=+1, dH=+15, dS=+8),
    "Apple___Black_rot":               profile("Apple", dT=+6, dH=+8,  dS=+3),
    "Apple___Cedar_apple_rust":        profile("Apple", dT=+3, dH=+12, dS=-2),
    "Apple___healthy":                 profile("Apple"),

    # ---- Blueberry ----
    "Blueberry___healthy":             profile("Blueberry"),

    # ---- Cherry ----
    "Cherry_(including_sour)___Powdery_mildew": profile("Cherry", dT=+4, dH=-3, dS=-8),
    "Cherry_(including_sour)___healthy":        profile("Cherry"),

    # ---- Corn ----
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": profile("Corn", dT=+2, dH=+18, dS=+12),
    "Corn_(maize)___Common_rust_":     profile("Corn", dT=-4, dH=+13, dS=+8),
    "Corn_(maize)___Northern_Leaf_Blight": profile("Corn", dT=-2, dH=+22, dS=+15),
    "Corn_(maize)___healthy":          profile("Corn"),

    # ---- Grape ----
    "Grape___Black_rot":               profile("Grape", dT=+3, dH=+20, dS=+10),
    "Grape___Esca_(Black_Measles)":    profile("Grape", dT=+5, dH=+8,  dS=+0),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": profile("Grape", dT=+2, dH=+25, dS=+15),
    "Grape___healthy":                 profile("Grape"),

    # ---- Orange ----
    "Orange___Haunglongbing_(Citrus_greening)": profile("Orange", dT=+2, dH=+4, dS=-2),

    # ---- Peach ----
    "Peach___Bacterial_spot":          profile("Peach", dT=+3, dH=+15, dS=+8),
    "Peach___healthy":                 profile("Peach"),

    # ---- Pepper ----
    "Pepper,_bell___Bacterial_spot":   profile("Pepper", dT=+2, dH=+18, dS=+10),
    "Pepper,_bell___healthy":          profile("Pepper"),

    # ---- Potato ----
    "Potato___Early_blight":           profile("Potato", dT=+7, dH=-5, dS=-15),
    "Potato___Late_blight":            profile("Potato", dT=-3, dH=+20, dS=+15),
    "Potato___healthy":                profile("Potato"),

    # ---- Raspberry ----
    "Raspberry___healthy":             profile("Raspberry"),

    # ---- Soybean ----
    "Soybean___healthy":               profile("Soybean"),

    # ---- Squash ----
    "Squash___Powdery_mildew":         profile("Squash", dT=+2, dH=-5, dS=-12),

    # ---- Strawberry ----
    "Strawberry___Leaf_scorch":        profile("Strawberry", dT=+5, dH=+8, dS=-5),
    "Strawberry___healthy":            profile("Strawberry"),

    # ---- Tomato ----
    "Tomato___Bacterial_spot":         profile("Tomato", dT=+3, dH=+18, dS=+10),
    "Tomato___Early_blight":           profile("Tomato", dT=+5, dH=-8, dS=-12),
    "Tomato___Late_blight":            profile("Tomato", dT=-5, dH=+22, dS=+18),
    "Tomato___Leaf_Mold":              profile("Tomato", dT=-1, dH=+24, dS=+8),
    "Tomato___Septoria_leaf_spot":     profile("Tomato", dT=-2, dH=+20, dS=+12),
    "Tomato___Spider_mites Two-spotted_spider_mite": profile("Tomato", dT=+8, dH=-22, dS=-18),
    "Tomato___Target_Spot":            profile("Tomato", dT=+2, dH=+14, dS=+6),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": profile("Tomato", dT=+6, dH=-10, dS=-8),
    "Tomato___Tomato_mosaic_virus":    profile("Tomato", dT=+1, dH=-5, dS=-3),
    "Tomato___healthy":                profile("Tomato"),
}

# Fallback for any class missing a profile
DEFAULT_HEALTHY = profile("Tomato")

ROWS_PER_CLASS = 300  # 38 * 300 = 11,400 rows total


def sample_row(profile):
    t_m, t_s, h_m, h_s, s_m, s_s = profile
    temp = np.random.normal(t_m, t_s)
    hum = np.random.normal(h_m, h_s)
    soil = np.random.normal(s_m, s_s)

    # Clip to physically valid sensor ranges
    t_lo, t_hi = config.SENSOR_RANGES["temperature"]
    h_lo, h_hi = config.SENSOR_RANGES["humidity"]
    s_lo, s_hi = config.SENSOR_RANGES["soil_moisture"]
    temp = float(np.clip(temp, t_lo, t_hi))
    hum = float(np.clip(hum, h_lo, h_hi))
    soil = float(np.clip(soil, s_lo, s_hi))
    return round(temp, 2), round(hum, 2), round(soil, 2)


def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Use raw data folder so this script also works before data_prep.py runs.
    class_names = list_class_names(config.RAW_DATA_DIR)
    missing = [c for c in class_names if c not in CLIMATE_PROFILES]
    if missing:
        print(f"WARNING: no climate profile for {len(missing)} classes "
              f"(will use default healthy profile): {missing}")

    ensure_dir(config.SENSOR_DIR)
    out_path: Path = config.SENSOR_CSV

    rows_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["temperature", "humidity", "soil_moisture", "disease_class"])
        for class_name in class_names:
            profile = CLIMATE_PROFILES.get(class_name, DEFAULT_HEALTHY)
            for _ in range(ROWS_PER_CLASS):
                temp, hum, soil = sample_row(profile)
                writer.writerow([temp, hum, soil, class_name])
                rows_written += 1

    print(f"Wrote {rows_written} rows across {len(class_names)} classes")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()

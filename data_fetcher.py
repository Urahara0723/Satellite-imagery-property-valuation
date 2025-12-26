# -*- coding: utf-8 -*-
"""enlarged_satellite_image_acquisition.ipynb

# Satellite Image Acquisition

In this stage, satellite imagery corresponding to each property location is programmatically retrieved using latitude and longitude coordinates from the tabular dataset. These images capture environmental and neighborhood context (such as vegetation, road networks, and nearby water bodies), which is not directly represented in structured tabular features.

The acquired satellite images will later be used as visual inputs in a multimodal regression framework.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = "/content/drive/MyDrive/property_valuation_project"

import os

os.makedirs(PROJECT_ROOT, exist_ok=True)
os.makedirs(f"{PROJECT_ROOT}/data", exist_ok=True)
os.makedirs(f"{PROJECT_ROOT}/data/images", exist_ok=True)
os.makedirs(f"{PROJECT_ROOT}/notebooks", exist_ok=True)

IMAGE_VERSION = "v2"
IMAGE_DIR = f"{PROJECT_ROOT}/data/images_{IMAGE_VERSION}"
os.makedirs(IMAGE_DIR, exist_ok=True)

train = pd.read_excel(f"{PROJECT_ROOT}/data/train.xlsx")

train = train[["id", "lat", "long", "price"]]
train.sample(5)

# Create price bins for stratification
train["price_bin"] = pd.qcut(train["price"], q=5, labels=False)

SAMPLES_PER_BIN = 1600

image_subset = (
    train
    .groupby("price_bin", group_keys=False)
    .apply(lambda x: x.sample(
        min(len(x), SAMPLES_PER_BIN),
        random_state=42
    ))
    .reset_index(drop=True)
)


len(image_subset)

# Save metadata for reproducibility
metadata_path = f"{PROJECT_ROOT}/data/image_metadata_{IMAGE_VERSION}.csv"
image_subset[["id", "lat", "long", "price"]].to_csv(
    metadata_path, index=False
)

print("Saved image metadata to:", metadata_path)

import math
from io import BytesIO

def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0 * n
    )
    return xtile, ytile

"""
Zoom level rationale:
- Zoom 18-20 captures neighborhood-scale context
- Roads, vegetation, water bodies, parcel layout
- Avoids excessive noise from micro-details (cars, shadows)
- Consistent spatial context across all properties

"""

def fetch_esri_image(lat, lon, zoom=19):
    x, y = latlon_to_tile(lat, lon, zoom)

    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    )

    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None

sample_row = image_subset.iloc[0]

test_img = fetch_esri_image(
    lat=sample_row["lat"],
    lon=sample_row["long"],
    zoom=19
)

plt.figure(figsize=(4,4))
plt.imshow(test_img)
plt.axis("off")
plt.title("ESRI World Imagery (Test)")
plt.show()

def save_esri_image(img, property_id):
    if img is not None and img.size[0] > 50 and img.size[1] > 50:
        img.save(f"{IMAGE_DIR}/{property_id}.png")

import time

for _, row in image_subset.iterrows():
    img_path = f"{IMAGE_DIR}/{row['id']}.png"

    if not os.path.exists(img_path):
        img = fetch_esri_image(
            lat=row["lat"],
            lon=row["long"],
            zoom=19
        )

        if img is not None:
            save_esri_image(img, row["id"])
            time.sleep(0.3)  # respectful delay

import random

random_id = random.choice(os.listdir(IMAGE_DIR))
img = Image.open(f"{IMAGE_DIR}/{random_id}")

plt.figure(figsize=(4,4))
plt.imshow(img)
plt.axis("off")
plt.title("Random ESRI Aerial Image")
plt.show()

num_images = len(os.listdir(IMAGE_DIR))
num_metadata = len(pd.read_csv(f"{PROJECT_ROOT}/data/image_metadata_{IMAGE_VERSION}.csv"))


print("Images on disk:", num_images)
print("Metadata rows:", num_metadata)

# Sanity check for image coverage
metadata_ids = set(image_subset["id"].astype(int))

image_ids = set(
    int(float(f.replace(".png", "")))
    for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png")
)

missing_images = metadata_ids - image_ids

print("Expected images:", len(metadata_ids))
print("Downloaded images:", len(image_ids))
print("Missing images:", len(missing_images))

if len(missing_images) > 0:
    print("Sample missing IDs:", list(missing_images)[:5])

"""Each aerial image was uniquely associated with a property using the property identifier present in the tabular dataset. Images were saved using this identifier as the filename, enabling a direct one-to-one mapping between tabular records and visual data. This design ensures seamless multimodal feature fusion and supports explainability analyses such as Grad-CAM at the individual property level."""
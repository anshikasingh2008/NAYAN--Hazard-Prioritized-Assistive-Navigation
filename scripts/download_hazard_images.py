import os
import urllib.request
from pycocotools.coco import COCO
from tqdm import tqdm
import time

# --- CONFIGURATION ---
ANNOTATION_FILE = 'annotations/instances_train2017.json'

# Use EXACT COCO category names
hazard_categories = [
    'person',
    'car',
    'bicycle',
    'motorcycle',
    'bus',
    'truck',
    'traffic light',
    'stop sign',
    'fire hydrant'
]

MAX_IMAGES = 500
# --- END OF CONFIGURATION ---

print("=" * 50)
print("NAYAN - COCO Hazard Image Downloader")
print("=" * 50)

# Check if annotation file exists
if not os.path.exists(ANNOTATION_FILE):
    print(f"❌ Error: {ANNOTATION_FILE} not found!")
    exit()

print(f"✅ Annotation file found: {ANNOTATION_FILE}")

# Load COCO
print("Loading COCO annotations...")
coco = COCO(ANNOTATION_FILE)

# Get category IDs
cat_ids = coco.getCatIds(catNms=hazard_categories)

# Show which categories were found
found_categories = coco.loadCats(cat_ids)
print(f"Found {len(cat_ids)} hazard categories:")
for cat in found_categories:
    print(f"  ✅ {cat['id']}: {cat['name']}")

if len(cat_ids) == 0:
    print("\n❌ No categories found! Check category names.")
    exit()

# IMPORTANT FIX: Get image IDs for EACH category and combine
all_img_ids = []
for cat_id in cat_ids:
    img_ids = coco.getImgIds(catIds=cat_id)
    all_img_ids.extend(img_ids)
    print(f"  Category {cat_id}: {len(img_ids)} images")

# Remove duplicates
all_img_ids = list(set(all_img_ids))
print(f"\n📊 Total unique images with hazards: {len(all_img_ids)}")

# Limit images
if MAX_IMAGES and len(all_img_ids) > MAX_IMAGES:
    all_img_ids = all_img_ids[:MAX_IMAGES]
    print(f"📥 Downloading first {len(all_img_ids)} images")

# Create directory
os.makedirs('coco_images/train', exist_ok=True)

# Download images
print("\n⏳ Starting download...")
successful = 0
failed = 0

for img_id in tqdm(all_img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']
    img_path = f"coco_images/train/{img_info['file_name']}"
    
    if not os.path.exists(img_path):
        try:
            urllib.request.urlretrieve(img_url, img_path)
            successful += 1
            time.sleep(0.05)
        except Exception as e:
            failed += 1

print("\n" + "=" * 50)
print("✅ DOWNLOAD COMPLETE")
print("=" * 50)
print(f"✅ Successfully downloaded: {successful} images")
print(f"❌ Failed: {failed} images")
print(f"📁 Images saved in: coco_images/train/")
import os
import json
from pycocotools.coco import COCO

ANNOTATIONS_PATH = "annotations/instances_train2017.json"
IMAGES_DIR = "coco_images/train"
LABELS_DIR = "labels/train"

HAZARD_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "fire hydrant", "bench", "chair", "stop sign"
]

os.makedirs(LABELS_DIR, exist_ok=True)

print("Loading COCO annotations (this can take a minute)...")
coco = COCO(ANNOTATIONS_PATH)

cat_ids = coco.getCatIds(catNms=HAZARD_CLASSES)
cat_id_to_yolo_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
yolo_classes = [coco.loadCats([c])[0]["name"] for c in cat_ids]

downloaded_files = set(os.listdir(IMAGES_DIR))
print(f"Found {len(downloaded_files)} images already downloaded.")

converted = 0
for filename in downloaded_files:
    if not filename.lower().endswith(".jpg"):
        continue

    image_id = int(filename.split(".")[0])
    img_info = coco.loadImgs([image_id])[0]
    img_w, img_h = img_info["width"], img_info["height"]

    ann_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)

    if not anns:
        continue

    label_lines = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        yolo_class = cat_id_to_yolo_idx[ann["category_id"]]
        label_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    label_path = os.path.join(LABELS_DIR, filename.replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))
    converted += 1

print(f"Converted {converted} images to YOLO label format -> {LABELS_DIR}")

yaml_content = f"""train: {os.path.abspath(IMAGES_DIR)}
val: {os.path.abspath(IMAGES_DIR)}
nc: {len(yolo_classes)}
names: {yolo_classes}
"""
with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("Wrote data.yaml — ready for training.")
print(f"Classes used ({len(yolo_classes)}): {yolo_classes}")

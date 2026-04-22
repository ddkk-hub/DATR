"""
Convert Cityscapes gtFine polygon annotations to COCO-format JSON.
Run once after downloading Cityscapes:

    python scripts/cityscapes_to_coco.py \
        --root D:/datasets/Cityscapes \
        --output D:/datasets/Cityscapes/annotations

Creates:
    D:/datasets/Cityscapes/annotations/
        gtFine_train.json
        gtFine_val.json
"""
import argparse
import json
import os
from collections import defaultdict

# Cityscapes 8 classes that map to the 9-class model output
# (person + rider merged = person_rider, but we keep separate)
CITYSCAPES_CATEGORIES = [
    {"id": 1,  "name": "person",       "supercategory": "human"},
    {"id": 2,  "name": "rider",        "supercategory": "human"},
    {"id": 3,  "name": "car",          "supercategory": "vehicle"},
    {"id": 4,  "name": "truck",        "supercategory": "vehicle"},
    {"id": 5,  "name": "bus",           "supercategory": "vehicle"},
    {"id": 6,  "name": "train",         "supercategory": "vehicle"},
    {"id": 7,  "name": "motorcycle",   "supercategory": "vehicle"},
    {"id": 8,  "name": "bicycle",      "supercategory": "vehicle"},
    {"id": 11, "name": "license plate", "supercategory": "vehicle"},  # ignored (id=11 in CS)
]

# Map Cityscapes label IDs to COCO category IDs
# Cityscapes polygon labels: person=24, rider=25, car=26, truck=28, bus=29, train=30, motorcycle=32, bicycle=33, license plate=-1
CITYSCAPES_LABEL_MAP = {
    "person":       1,
    "rider":        2,
    "car":          3,
    "truck":         4,
    "bus":          5,
    "train":        6,
    "motorcycle":   7,
    "bicycle":      8,
    # "license plate" excluded (too small, not useful)
}

def cityscapes_to_coco(gtFine_root, img_root, split):
    """
    Convert Cityscapes gtFine polygons to COCO format.
    
    Args:
        gtFine_root: e.g. D:/datasets/Cityscapes/gtFine/train
        img_root:     e.g. D:/datasets/Cityscapes/leftImg8bit/train
        split:        'train' or 'val'
    """
    images = []
    annotations = []
    ann_id = 1

    categories = [{"id": v, "name": k, "supercategory": CITYSCAPES_CATEGORIES[next(i for i, c in enumerate(CITYSCAPES_CATEGORIES) if c["name"] == k)]["supercategory"]}
                  for k, v in CITYSCAPES_LABEL_MAP.items()]

    # Walk through all city folders in gtFine
    for city_name in sorted(os.listdir(gtFine_root)):
        city_img_dir = os.path.join(img_root, city_name)
        city_ann_dir = os.path.join(gtFine_root, city_name)
        if not os.path.isdir(city_ann_dir):
            continue

        for ann_file in sorted(os.listdir(city_ann_dir)):
            if not ann_file.endswith('_gtFine_polygons.json'):
                continue

            # Parse annotation file
            ann_path = os.path.join(city_ann_dir, ann_file)
            with open(ann_path, encoding='utf-8') as f:
                ann_data = json.load(f)

            # Image info - construct from annotation filename if not in JSON
            # annotation: aachen_000000_000019_gtFine_polygons.json
            # image: aachen_000000_000019_leftImg8bit.png
            base_name = ann_file.replace('_gtFine_polygons.json', '')
            img_filename = f"{base_name}_leftImg8bit.png"
            img_id = len(images) + 1

            # Find actual image path - img_root is already e.g. leftImg8bit/train
            img_path = os.path.join(img_root, city_name, img_filename)
            if not os.path.exists(img_path):
                print(f"  WARNING: image not found: {img_path}")
                continue

            width = ann_data['imgWidth']
            height = ann_data['imgHeight']

            images.append({
                "id": img_id,
                "file_name": img_filename.replace('\\', '/'),
                "width": width,
                "height": height,
            })

            # Process objects
            for obj in ann_data.get('objects', []):
                label = obj.get('label', '')
                # Map label (e.g. "person" → 1)
                cat_id = CITYSCAPES_LABEL_MAP.get(label)
                if cat_id is None:
                    continue  # skip "license plate", "road", etc.

                polygons = obj.get('polygon', [])
                if not polygons:
                    continue

                # Merge all polygon parts into one flat list per object
                # polygon is already a list of [x, y] point pairs
                all_points = []
                for part in polygons:
                    # Each part is [x, y], extend adds this point pair
                    all_points.append(part)

                if len(all_points) < 6:
                    continue

                # Compute bounding box
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                area = (x_max - x_min) * (y_max - y_min)
                if area <= 0:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": [all_points],
                    "area": float(area),
                    "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                    "iscrowd": 0,
                })
                ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    parser = argparse.ArgumentParser(description='Convert Cityscapes to COCO format')
    parser.add_argument('--root', type=str, required=True,
                        help='Cityscapes dataset root (contains leftImg8bit/ and gtFine/)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for JSON files')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                        help='Splits to convert (train val)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for split in args.splits:
        print(f"\nConverting {split} split...")
        gtFine_dir = os.path.join(args.root, 'gtFine', split)
        img_dir = os.path.join(args.root, 'leftImg8bit', split)

        if not os.path.isdir(gtFine_dir):
            print(f"  SKIP: {gtFine_dir} not found")
            continue

        coco = cityscapes_to_coco(gtFine_dir, img_dir, split)
        out_file = os.path.join(args.output, f'gtFine_{split}.json')

        print(f"  Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_file}")


if __name__ == '__main__':
    main()

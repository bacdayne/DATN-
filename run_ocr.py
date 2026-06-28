from datasets import load_dataset
from paddleocr import PaddleOCR

import os
import json
import cv2
import numpy as np
import argparse


def make_dirs():
    os.makedirs("ocr", exist_ok=True)
    os.makedirs("ocr/images", exist_ok=True)
    os.makedirs("ocr/boxes", exist_ok=True)
    os.makedirs("ocr/debug", exist_ok=True)


def box_to_quad(box):
    return {
        "x1": float(box[0][0]),
        "y1": float(box[0][1]),
        "x2": float(box[1][0]),
        "y2": float(box[1][1]),
        "x3": float(box[2][0]),
        "y3": float(box[2][1]),
        "x4": float(box[3][0]),
        "y4": float(box[3][1]),
    }


def get_center(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]

    return sum(xs) / 4, sum(ys) / 4


def draw_ocr_boxes(image_path, ocr_boxes, output_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    for item in ocr_boxes:
        pts = np.array(item["box"], dtype=np.int32)

        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        x = int(pts[0][0])
        y = int(pts[0][1])

        label = item["text"][:30]

        cv2.rectangle(
            image,
            (x, max(y - 20, 0)),
            (x + len(label) * 9, y),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            image,
            label,
            (x, max(y - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(output_path, image)

    return output_path


def run_paddleocr(ocr, image_path, debug=False):
    result = ocr.ocr(
        image_path,
        cls=True
    )

    ocr_boxes = []

    if result is None or len(result) == 0 or result[0] is None:
        return {
            "ocr_text": "",
            "ocr_boxes": []
        }

    for idx, line in enumerate(result[0]):
        box = line[0]
        text = line[1][0].strip()
        confidence = float(line[1][1])

        if text == "":
            continue

        x_center, y_center = get_center(box)

        item = {
            "id": idx,
            "text": text,
            "confidence": confidence,
            "quad": box_to_quad(box),
            "box": [
                [float(x), float(y)]
                for x, y in box
            ],
            "x_center": float(x_center),
            "y_center": float(y_center)
        }

        ocr_boxes.append(item)

        if debug:
            print(f"{text} | conf={confidence:.4f}")

    ocr_boxes = sorted(
        ocr_boxes,
        key=lambda x: (
            x["y_center"],
            x["x_center"]
        )
    )

    ocr_text = "\n".join(
        item["text"]
        for item in ocr_boxes
    )

    return {
        "ocr_text": ocr_text,
        "ocr_boxes": ocr_boxes
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"]
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true"
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    args = parser.parse_args()

    make_dirs()

    print("Loading CORD-v2 dataset...")
    dataset = load_dataset(
        "naver-clova-ix/cord-v2"
    )

    data = dataset[args.split]

    print("Loading PaddleOCR...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=args.use_gpu,
        show_log=False
    )

    all_results = []

    total = min(args.limit, len(data))

    for idx in range(total):
        print("=" * 60)
        print(f"OCR image {idx + 1}/{total}")
        print("=" * 60)

        sample = data[idx]

        image = sample["image"]

        image_path = os.path.join(
            "ocr",
            "images",
            f"{args.split}_{idx}.png"
        )

        image.save(image_path)

        ground_truth = json.loads(
            sample["ground_truth"]
        )

        ocr_result = run_paddleocr(
            ocr,
            image_path,
            debug=args.debug
        )

        box_path = os.path.join(
            "ocr",
            "boxes",
            f"{args.split}_{idx}_boxes.json"
        )

        text_path = os.path.join(
            "ocr",
            "boxes",
            f"{args.split}_{idx}_text.txt"
        )

        debug_image_path = os.path.join(
            "ocr",
            "debug",
            f"{args.split}_{idx}_debug.png"
        )

        with open(
            box_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                ocr_result["ocr_boxes"],
                f,
                ensure_ascii=False,
                indent=2
            )

        with open(
            text_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(
                ocr_result["ocr_text"]
            )

        draw_ocr_boxes(
            image_path,
            ocr_result["ocr_boxes"],
            debug_image_path
        )

        item = {
            "image_index": idx,
            "split": args.split,
            "image_path": image_path,
            "ocr_text_path": text_path,
            "ocr_boxes_path": box_path,
            "debug_image_path": debug_image_path,
            "ocr_text": ocr_result["ocr_text"],
            "ocr_boxes": ocr_result["ocr_boxes"],
            "ground_truth": ground_truth["gt_parse"]
        }

        all_results.append(item)

        print("Saved image:", image_path)
        print("Saved OCR text:", text_path)
        print("Saved OCR boxes:", box_path)
        print("Saved debug image:", debug_image_path)

        print("\nOCR TEXT:")
        print(ocr_result["ocr_text"][:1000])
        print()

    output_json = os.path.join(
        "ocr",
        f"ocr_results_{args.split}.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 60)
    print("DONE")
    print("Saved:", output_json)
    print("=" * 60)


if __name__ == "__main__":
    main()
import unittest

from review_labels import box_iou, inspect_boxes


def make_box(line, class_id, x, y, width, height):
    return {
        "line": line,
        "class_id": class_id,
        "class_name": "forklift" if class_id == 0 else "person",
        "xywh": [x, y, width, height],
        "xyxy": [x - width / 2, y - height / 2, x + width / 2, y + height / 2],
        "area": width * height,
        "issue_codes": [],
    }


class LabelReviewTests(unittest.TestCase):
    def test_iou_for_identical_boxes(self):
        box = make_box(1, 0, .5, .5, .4, .4)
        self.assertAlmostEqual(box_iou(box, box), 1.0)

    def test_full_frame_and_inner_same_class_are_flagged(self):
        boxes = [
            make_box(1, 1, .5, .5, .98, .98),
            make_box(2, 1, .6, .6, .2, .3),
        ]

        findings = inspect_boxes(boxes)
        codes = {item["code"] for item in findings}

        self.assertIn("large_box", codes)
        self.assertIn("large_box_with_inner_same_class", codes)
        self.assertIn("large_box", boxes[0]["issue_codes"])
        self.assertIn("large_box_with_inner_same_class", boxes[1]["issue_codes"])

    def test_duplicate_and_cross_class_overlap_are_separate(self):
        boxes = [
            make_box(1, 0, .5, .5, .4, .4),
            make_box(2, 0, .5, .5, .4, .4),
            make_box(3, 1, .5, .5, .4, .4),
        ]

        codes = {item["code"] for item in inspect_boxes(boxes)}

        self.assertIn("duplicate_same_class", codes)
        self.assertIn("cross_class_overlap", codes)


if __name__ == "__main__":
    unittest.main()

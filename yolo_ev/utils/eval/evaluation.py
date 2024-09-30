import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluation(Gt, Dt, num_data):
    coco_gt= COCO()
    coco_gt.dataset = Gt

    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(Dt)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, num_data + 1, dtype=int)
    coco_eval.evaluate()
    coco_eval.accumulate()

    out_keys = ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L')
    out_dict = {k: 0.0 for k in out_keys}

    for idx, key in enumerate(out_keys):
        out_dict[key] = coco_eval.stats[idx]
    return out_dict

def to_coco_format(gts, detections, categories, height=240, width=304):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox in gt:
            x1, y1 = bbox['x'], bbox['y']
            w, h = bbox['w'], bbox['h']
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox['class_id']) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox in pred:
            image_result = {
                'image_id': im_id,
                'category_id': int(bbox['class_id']) + 1,
                'score': float(bbox['class_confidence']),
                'bbox': [bbox['x'], bbox['y'], bbox['w'], bbox['h']],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    return dataset, results

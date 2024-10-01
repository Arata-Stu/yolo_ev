import os
import copy
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import psutil

def remove_useless_info(coco):
    # 省メモリのためにCOCOデータセットの不要情報を削除
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)

class COCODataset(Dataset):

    def __init__(self, data_dir, json_file, name="train2017", img_size=(256, 256), transform=None, cache=False, cache_type="ram"):
        
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)

        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.transform = transform
        self.annotations = self._load_coco_annotations()
        self.path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        
        # キャッシュ設定
        self.cache = cache
        self.cache_type = cache_type
        self.imgs = [None] * self.num_imgs if self.cache and self.cache_type == "ram" else None

        if self.cache and self.cache_type == "disk":
            self.cache_dir = os.path.join(self.data_dir, f"cache_{self.name}")
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

        if self.cache:
            self.cache_images()

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, "images", self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"
        return img

    def read_img(self, index):
        # キャッシュが有効であれば、キャッシュから画像を読み込む
        if self.cache:
            if self.cache_type == "ram" and self.imgs[index] is not None:
                return copy.deepcopy(self.imgs[index])
            elif self.cache_type == "disk":
                cache_file = os.path.join(self.cache_dir, f"{index}.npy")
                if os.path.exists(cache_file):
                    return np.load(cache_file)
        
        # キャッシュがない場合は画像をリサイズして読み込む
        img = self.load_resized_img(index)

        # キャッシュに保存
        if self.cache_type == "ram":
            self.imgs[index] = copy.deepcopy(img)
        elif self.cache_type == "disk":
            np.save(os.path.join(self.cache_dir, f"{index}.npy"), img)

        return img

    def cache_images(self):
        # すべての画像をキャッシュに保存
        for index in range(self.num_imgs):
            self.read_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)
        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.transform is not None:
            img, target = self.transform(img, target, self.img_size)
        return img, target, img_info, img_id

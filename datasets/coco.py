import os
import numpy as np
import json_tricks as json
import xtcocotools
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
from collections import defaultdict
from .base import BaseDataset
from .utils import oks_nms, soft_oks_nms


class COCOPose(BaseDataset):
    """CocoDataset dataset for top-down pose estimation.

    `Microsoft COCO: Common Objects in Context' ECCV'2014
    More details can be found in the `paper
    <https://arxiv.org/abs/1405.0312>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode)

        # init parameters
        self.use_nms = data_cfg['use_nms']
        self.soft_nms = data_cfg['soft_nms']
        self.oks_thr = data_cfg['oks_thr']
        self.num_scales = data_cfg['num_scales']

        self.ann_info['num_scales'] = self.num_scales
        self.ann_info['num_joints'] = 17
        self.ann_info['num_output_channels'] = 17
        self.ann_info['flip_index'] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        self.ann_info['dataset_channel'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.ann_info['inference_channel'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array([
            1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
        ], dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0

        self.dataset_name = 'coco'

        # load annotations
        self.coco = COCO(ann_file)
        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.db = self.coco.getImgIds()
        if not test_mode:
            self.db = [
                img_id for img_id in self.db if
                len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
        self.num_images = len(self.db)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

    def __len__(self):
        """Get dataset length."""
        return len(self.db)

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _get_single(self, idx):
        """Get anno for a single image.

        Args:
            idx (int): image idx

        Returns:
            dict: info for model training
        """
        coco = self.coco
        img_id = self.db[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)
        anno = [obj.copy() for obj in anno if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0]
        mask = self._get_mask(anno, idx)
        joints = self._get_joints(anno)
        boxes = self._get_boxes(anno)

        db_rec = dict()
        db_rec['dataset'] = self.dataset_name
        db_rec['image_file'] = os.path.join(self.img_prefix, self.id2name[img_id])
        db_rec['mask'] = [mask.copy() for _ in range(self.num_scales)]
        db_rec['joints'] = [joints.copy() for _ in range(self.num_scales)]
        db_rec['boxes'] = boxes

        return db_rec

    def _get_joints(self, anno):
        """Get joints for all people in an image."""
        num_people = len(anno)

        joints = np.zeros((num_people, self.ann_info['num_joints'], 3), dtype=np.float32)

        for i, obj in enumerate(anno):
            joints[i, :, :3] = np.array(obj['keypoints']).reshape([-1, 3])

        return joints

    def _get_boxes(self, anno):
        """Get joints for all people in an image."""
        num_people = len(anno)

        boxes = np.zeros((num_people, 4), dtype=np.float32)
        for i, obj in enumerate(anno):
            boxes[i] = np.array(obj['bbox'])

        return boxes

    def _get_mask(self, anno, idx):
        """Get ignore masks to mask out losses."""
        coco = self.coco
        img_info = coco.loadImgs(self.db[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)

        for obj in anno:
            if 'segmentation' in obj:
                if obj['iscrowd']:
                    rle = xtcocotools.mask.frPyObjects(obj['segmentation'],
                                                       img_info['height'],
                                                       img_info['width'])
                    m += xtcocotools.mask.decode(rle)
                elif obj['num_keypoints'] == 0:
                    rles = xtcocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'],
                        img_info['width'])
                    for rle in rles:
                        m += xtcocotools.mask.decode(rle)

        return m < 0.5

    def evaluate(self, results, res_folder='output', metric='mAP', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - num_people: P
            - num_keypoints: K

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (list[np.ndarray(P, K, 3+tag_num)]): \
                    Pose predictions for all people in images.
                - scores (list[P]): List of person scores.
                - image_path (list[str]): For example, ['coco/images/\
                    val2017/000000397133.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model outputs.

            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        os.makedirs(res_folder, exist_ok=True)
        res_file = os.path.join(res_folder, 'result_keypoints.json')

        preds = []
        scores = []
        image_paths = []

        for result in results:
            preds.append(result['preds'])
            scores.append(result['scores'])
            image_paths.append(result['image_path'])

        kpts = defaultdict(list)
        # iterate over images
        for idx, _preds in enumerate(preds):
            str_image_path = image_paths[idx]
            image_id = self.name2id[os.path.basename(str_image_path)]
            # iterate over people
            for idx_person, kpt in enumerate(_preds):
                # use bbox area
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (
                    np.max(kpt[:, 1]) - np.min(kpt[:, 1]))

                kpts[image_id].append({
                    'keypoints': kpt[:, 0:3],
                    'score': scores[idx][idx_person],
                    'image_id': image_id,
                    'area': area,
                })

        valid_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, self.oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_coco_keypoint_results(valid_kpts, res_file)

        mean_ap, info_str = self._do_python_keypoint_eval(res_file)

        return mean_ap, info_str

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            for img_kpt, key_point in zip(img_kpts, key_points):
                kpt = key_point.reshape((self.ann_info['num_joints'], 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpt['image_id'],
                    'category_id': cat_id,
                    'keypoints': key_point.tolist(),
                    'score': img_kpt['score'],
                    'bbox': [left_top[0], left_top[1], w, h]
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP(50)', 'AP(75)', 'AP (M)', 'AP (L)', 'AR', 'AR(50)', 'AR(75)', 'AR (M)', 'AR (L)']
        values = coco_eval.stats[0:len(stats_names)]
        info_str = '||'.join(['{}:{:.4f}'.format(stats_names[k], v) for k, v in enumerate(values)])
        mean_ap = values[0]

        return mean_ap, info_str

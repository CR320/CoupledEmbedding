import numpy as np
from xtcocotools.cocoeval import COCOeval
from .coco import COCOPose


class CrowdPose(COCOPose):
    """CrowdPose dataset for bottom-up pose estimation.

    "CrowdPose: Efficient Crowded Scenes Pose Estimation and
    A New Benchmark", CVPR'2019.
    More details can be found in the `paper
    <https://arxiv.org/abs/1812.00324>`__.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    CrowdPose keypoint indexes::

        0: 'left_shoulder',
        1: 'right_shoulder',
        2: 'left_elbow',
        3: 'right_elbow',
        4: 'left_wrist',
        5: 'right_wrist',
        6: 'left_hip',
        7: 'right_hip',
        8: 'left_knee',
        9: 'right_knee',
        10: 'left_ankle',
        11: 'right_ankle',
        12: 'top_head',
        13: 'neck'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
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
        self.ann_info['num_joints'] = 14
        self.ann_info['num_output_channels'] = 14
        self.ann_info['flip_index'] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13]
        self.ann_info['dataset_channel'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.ann_info['inference_channel'] = [13, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array([
            0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2, 0.2, 0.5
        ], dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.sigmas = np.array([
            .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87,
            .89, .89, .79, .79
        ]) / 10.0

        self.dataset_name = 'crowdpose'

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints_crowd', self.sigmas, use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP(50)', 'AP(75)', 'AR', 'AR(50)', 'AR(75)', 'AP(E)', 'AP(M)', 'AP(H)']
        values = coco_eval.stats[0:len(stats_names)]
        info_str = '||'.join(['{}:{:.4f}'.format(stats_names[k], v) for k, v in enumerate(values)])
        mean_ap = values[0]

        return mean_ap, info_str

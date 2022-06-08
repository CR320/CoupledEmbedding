import copy
import numpy as np
from torch.utils.data import Dataset
from .pipeline.loading import Compose


class BaseDataset(Dataset):
    """Base class for top-down datasets.

    All top-down datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`

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

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.ann_info['image_size'] = data_cfg['image_size']
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.pipeline = Compose(self.pipeline)
        self.img_ids = None

    def _get_single(self, idx):
        """Get anno for a single image."""
        raise NotImplementedError

    def evaluate(self, results, *args, **kwargs):
        """Evaluate keypoint results."""

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.img_ids)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

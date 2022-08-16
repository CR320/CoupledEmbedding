# ------------------------------------------------------------------------------
# Adapted from https://github.com/open-mmlab/mmpose
# Original licence: Copyright (c) OpenMMLab, under the Apache License.
# ------------------------------------------------------------------------------
import torch
import importlib
import collections
from collections.abc import Sequence


def collate_function(batches):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    """

    if not isinstance(batches, Sequence):
        raise TypeError(f'{batches.dtype} is not supported.')

    data = dict()
    meta_data = dict()
    for batch in batches:
        for key in batch:
            if type(batch[key]).__name__ == 'ndarray':
                if key not in data:
                    data[key] = list()
                data[key].append(torch.from_numpy(batch[key]))
            elif type(batch[key]).__name__ == 'list' and type(batch[key][0]).__name__ == 'ndarray':
                if key not in data:
                    data[key] = [list() for _ in range(len(batch[key]))]
                for i in range(len(batch[key])):
                    data[key][i].append(torch.from_numpy(batch[key][i]))
            else:
                if key not in meta_data:
                    meta_data[key] = list()
                meta_data[key].append(batch[key])

    for key in data:
        if type(data[key][0]).__name__ == 'Tensor':
            data[key] = torch.stack(data[key], dim=0)
        else:
            for i in range(len(data[key])):
                data[key][i] = torch.stack(data[key][i], dim=0)

    data.update(meta_data)

    return data


class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms: List of transform objects to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = list()
        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                # raise TypeError('transform must be callable')
                lib = importlib.import_module('datasets')
                self.transforms.append(self.build(lib, transform.copy()))

    @staticmethod
    def build(lib, args):
        assert 'type' in args
        model = getattr(lib, args.type)
        args.pop('type')

        return model(**args)

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
          This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
          The contents of the `meta_name` dictionary depends on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = meta

        return data
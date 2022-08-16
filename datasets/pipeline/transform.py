# ------------------------------------------------------------------------------
# Adapted from https://github.com/open-mmlab/mmpose
# Original licence: Copyright (c) OpenMMLab, under the Apache License.
# ------------------------------------------------------------------------------
import cv2
import mmcv
import numpy as np
from .utils import get_affine_transform, warp_affine_joints, warp_affine_boxes


class LoadImageFromFile:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        image_file = results['image_file']
        img = mmcv.imread(image_file, self.color_type, self.channel_order, backend='cv2')
        if img is None:
            raise ValueError('Fail to read {}'.format(image_file))
        results['image'] = img

        return results


class RandomAffine:
    """Data augmentation with random scaling & rotating.

    Args:
        rot_factor (int): Rotating to [-rotation_factor, rotation_factor]
        scale_factor (list): Scaling to [1-scale_factor, 1+scale_factor]
        scale_type: wrt ``long`` or ``short`` length of the image.
        trans_factor: Translation factor.
    """

    def __init__(self,
                 rot_factor,
                 scale_factor,
                 scale_type,
                 trans_factor):
        self.max_rotation = rot_factor
        self.min_scale = scale_factor[0]
        self.max_scale = scale_factor[1]
        self.scale_type = scale_type
        self.trans_factor = trans_factor

    def _get_scale(self, image_size, resized_size):
        w, h = image_size
        w_resized, h_resized = resized_size
        if w / w_resized < h / h_resized:
            if self.scale_type == 'long':
                w_pad = h / h_resized * w_resized
                h_pad = h
            elif self.scale_type == 'short':
                w_pad = w
                h_pad = w / w_resized * h_resized
            else:
                raise ValueError(f'Unknown scale type: {self.scale_type}')
        else:
            if self.scale_type == 'long':
                w_pad = w
                h_pad = w / w_resized * h_resized
            elif self.scale_type == 'short':
                w_pad = h / h_resized * w_resized
                h_pad = h
            else:
                raise ValueError(f'Unknown scale type: {self.scale_type}')

        scale = np.array([w_pad, h_pad], dtype=np.float32)

        return scale

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        image, boxes, mask, joints = results['image'], results['boxes'], results['mask'], results['joints']

        self.input_size = np.array([results['ann_info']['image_size'],
                                    results['ann_info']['image_size']])
        self.output_size = results['ann_info']['heatmap_size']

        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        height, width = image.shape[:2]
        center = np.array([width / 2, height / 2])
        img_scale = np.array([width, height], dtype=np.float32)
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale
        img_scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.trans_factor > 0:
            dx = np.random.randint(-self.trans_factor * img_scale[0] / 200.0,
                                   self.trans_factor * img_scale[0] / 200.0)
            dy = np.random.randint(-self.trans_factor * img_scale[1] / 200.0,
                                   self.trans_factor * img_scale[1] / 200.0)

            center[0] += dx
            center[1] += dy

        for i, _output_size in enumerate(self.output_size):
            _output_size = np.array([_output_size, _output_size])
            scale = self._get_scale(img_scale, _output_size)
            mat_output = get_affine_transform(
                center=center,
                scale=scale / 200.0,
                rot=aug_rot,
                output_size=_output_size)
            mask[i] = cv2.warpAffine((mask[i] * 255).astype(np.uint8),
                                     mat_output,
                                     (int(_output_size[0]), int(_output_size[1]))) / 255
            mask[i] = (mask[i] > 0.5).astype(np.float32)

            joints[i][:, :, 0:2] = warp_affine_joints(joints[i][:, :, 0:2], mat_output)

        scale = self._get_scale(img_scale, self.input_size)
        mat_input = get_affine_transform(
            center=center,
            scale=scale / 200.0,
            rot=aug_rot,
            output_size=self.input_size)
        image = cv2.warpAffine(image,
                               mat_input,
                               (int(self.input_size[0]), int(self.input_size[1])))
        boxes = warp_affine_boxes(boxes, mat_input)

        results['image'], results['boxes'], results['mask'], results['joints'] = image, boxes, mask, joints

        return results


class RandomFlip:
    """Data augmentation with random image flip for bottom-up.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        image, mask, joints = results['image'], results['mask'], results['joints']
        self.flip_index = results['ann_info']['flip_index']
        self.output_size = results['ann_info']['heatmap_size']

        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if np.random.random() < self.flip_prob:
            image = image[:, ::-1].copy() - np.zeros_like(image)
            for i, _output_size in enumerate(self.output_size):
                _output_size = np.array([_output_size, _output_size], dtype=np.int)
                mask[i] = mask[i][:, ::-1].copy()
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = _output_size[0] - joints[i][:, :, 0] - 1
        results['image'], results['mask'], results['joints'] = image, mask, joints

        return results


class HeatmapGenerator:
    """Generate heatmaps for bottom-up models.

    Args:
        num_joints (int): Number of keypoints
        output_size (np.ndarray): Size (w, h) of feature map
        sigma (int): Sigma of the heatmaps.
    """

    def __init__(self, output_size, num_joints, sigma=-1):
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size], dtype=np.int)
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_size.prod()**0.5 / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, joints):
        """Generate heatmaps."""
        hms = np.zeros([self.num_joints,
                        self.output_size[1],
                        self.output_size[0]], dtype=np.float32)

        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                            x >= self.output_size[0] or y >= self.output_size[1]:
                        continue

                    ul = int(np.round(x - 3 * sigma -
                                      1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma +
                                      2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0,
                               -ul[0]), min(br[0], self.output_size[0]) - ul[0]
                    a, b = max(0,
                               -ul[1]), min(br[1], self.output_size[1]) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_size[0])
                    aa, bb = max(0, ul[1]), min(br[1], self.output_size[1])
                    hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])

        return hms


class JointsEncoder:
    """Encodes the visible joints into (coordinates, score); The coordinate of
    one joint and its score are of `int` type.

    (idx * output_size**2 + y * output_size + x, 1) or (0, 0).

    Args:
        max_num_people(int): Max number of people in an image
        num_joints(int): Number of keypoints
        output_size(np.ndarray): Size (w, h) of feature map
        tag_per_joint(bool):  Option to use one tag map per joint.
    """

    def __init__(self, max_num_people, num_joints, output_size, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size], dtype=np.int)
        self.tag_per_joint = tag_per_joint

    def __call__(self, joints):
        """
        Note:
            - number of people in image: N
            - number of keypoints: K
            - max number of people in an image: M

        Args:
            joints (np.ndarray[N,K,3])

        Returns:
            visible_kpts (np.ndarray[M,K,2]).
        """
        visible_kpts = np.zeros((self.max_num_people, self.num_joints, 2),
                                dtype=np.float32)
        for i in range(len(joints)):
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if (pt[2] > 0 and 0 <= y < self.output_size[1]
                        and 0 <= x < self.output_size[0]):
                    if self.tag_per_joint:
                        visible_kpts[i][tot] = \
                            (idx * self.output_size.prod()
                             + y * self.output_size[0] + x, 1)
                    else:
                        visible_kpts[i][tot] = (y * self.output_size[0] + x, 1)
                    tot += 1
        return visible_kpts


class ScalesEncoder:
    """Encodes the visible instances' normalized scales

    Args:
        max_num_people(int): Max number of people in an image
        image_size(int): size of image
    """

    def __init__(self, max_num_people, image_size):
        self.max_num_people = max_num_people
        self.image_size = image_size

    def __call__(self, boxes):
        scales = np.zeros(self.max_num_people)
        e1 = np.sum((boxes[:, 0, :] - boxes[:, 1, :]) ** 2, axis=1) ** 0.5
        e2 = np.sum((boxes[:, 0, :] - boxes[:, 2, :]) ** 2, axis=1) ** 0.5
        _scales = (e1 * e2 / (self.image_size * self.image_size)) ** 0.5
        scales[0: len(_scales)] = _scales

        return scales


class FormatGroundTruth:
    """Generate multi-scale heatmap target for associate embedding.

    Args:
        sigma (int): Sigma of heatmap Gaussian
        max_num_people (int): Maximum number of people in an image
    """

    def __init__(self, sigma, max_num_people):
        self.sigma = sigma
        self.max_num_people = max_num_people

    def __call__(self, results):
        """Generate multi-scale heatmap target for bottom-up."""
        num_joints, heatmap_size, input_size = \
            results['ann_info']['num_joints'], results['ann_info']['heatmap_size'], results['ann_info']['image_size']
        heatmap_generator = [
            HeatmapGenerator(output_size, num_joints, self.sigma)
            for output_size in heatmap_size
        ]
        joints_encoder = [
            JointsEncoder(self.max_num_people, num_joints, output_size, True)
            for output_size in heatmap_size
        ]
        scales_encoder = ScalesEncoder(self.max_num_people, input_size)

        target_list = list()
        mask_list, joints_list = results['mask'], results['joints']

        for scale_id in range(results['ann_info']['num_scales']):
            target_t = heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = joints_encoder[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        results['box_scales'] = scales_encoder(results['boxes'])
        results['masks'], results['joints'] = mask_list, joints_list
        results['target_hms'] = target_list

        return results


class ResizeAlign:
    """Align transform for bottom-up.
        base_size (int): base size
        size_divisor (int): size_divisor
    """

    def __init__(self, size_divisor, scale_factors):
        self.size_divisor = size_divisor
        self.scale_factors = scale_factors

    def _ceil_to_multiples_of(self, x):
        """Transform x to the integral multiple of the base."""
        return int(np.ceil(x / self.size_divisor)) * self.size_divisor

    def _get_image_size(self, image_shape, input_size):
        # calculate the size for min_scale
        h, w = image_shape
        input_size = self._ceil_to_multiples_of(input_size)

        if w < h:
            w_resized = int(input_size)
            h_resized = int(self._ceil_to_multiples_of(input_size / w * h))
            scale_w = w / 200
            scale_h = h_resized / w_resized * w / 200
        else:
            h_resized = int(input_size)
            w_resized = int(self._ceil_to_multiples_of(input_size / h * w))
            scale_h = h / 200
            scale_w = w_resized / h_resized * h / 200

        base_size = (w_resized, h_resized)
        center = [round(w / 2.0), round(h / 2.0)]
        scale = [scale_w, scale_h]

        return base_size, center, scale

    def __call__(self, results):
        """Resize multi-scale size and align transform for bottom-up."""
        input_size = results['ann_info']['image_size']
        image = results['image']

        # get base_size, center & scale info for input image
        base_size, center, scale = self._get_image_size(image.shape[0:2], input_size)
        results['base_size'], results['center'], results['scale'] = base_size, center, scale

        # multi-scale resize
        assert self.scale_factors[0] == 1
        resized_images = list()
        for scale_factor in self.scale_factors:
            scaled_size = (int(base_size[0] * scale_factor), int(base_size[1] * scale_factor))
            trans = get_affine_transform(np.array(center), np.array(scale), 0, scaled_size)
            resized_images.append(cv2.warpAffine(image, trans, scaled_size))

        results['image'] = resized_images

        return results


class NormalizeImage:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        images = results['image']

        if isinstance(images, np.ndarray):
            images = mmcv.imnormalize(images, self.mean, self.std, self.to_rgb)
            norm_images = images.transpose(2, 0, 1)
        elif isinstance(images, list):
            norm_images = list()
            for image in images:
                image = mmcv.imnormalize(image, self.mean, self.std, self.to_rgb)
                norm_images.append(image.transpose(2, 0, 1))
        else:
            raise TypeError('Unsupported image type:{}'.format(type(images)))

        results['image'] = norm_images

        return results

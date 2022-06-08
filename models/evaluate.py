import torch
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
from mmcv import ProgressBar
from datasets.pipeline.utils import get_affine_transform, warp_affine_joints


def project2image(heatmaps, tagmaps, projected_size):
    resized_heatmaps = F.interpolate(heatmaps, size=projected_size, mode='bilinear', align_corners=False)
    resized_tagmaps = F.interpolate(tagmaps, size=projected_size, mode='bilinear', align_corners=False)

    return resized_heatmaps, resized_tagmaps


def flip_aggregation(heatmaps, tagmaps, with_flip, flip_index):
    if with_flip is True:
        hms, flipped_hms = heatmaps[0], heatmaps[1]
        flipped_hms = torch.flip(flipped_hms, [2])
        flipped_hms = flipped_hms[flip_index]
        avg_heatmaps = (hms + flipped_hms) / 2

        tms, flipped_tms = tagmaps[0], tagmaps[1]
        flipped_tms = torch.flip(flipped_tms, [2])
        flipped_tms = flipped_tms[flip_index]
        con_tagmaps = torch.stack([tms, flipped_tms], dim=3)
    else:
        avg_heatmaps = heatmaps
        con_tagmaps = tagmaps

    return avg_heatmaps, con_tagmaps


def nms(heatmaps, base_kernel_size):
    heatmaps = heatmaps.unsqueeze(0)
    local_max = F.max_pool2d(heatmaps,
                             kernel_size=base_kernel_size,
                             stride=1,
                             padding=int(base_kernel_size / 2))
    filtered_hms = heatmaps * torch.eq(heatmaps, local_max)

    return filtered_hms.squeeze(0)


def sample_candidates(heatmaps, tagmaps, top_k):
    K, H, W = heatmaps.size()
    heatmaps = heatmaps.view(K, -1)
    val_k, ind = heatmaps.topk(top_k, dim=1)
    tags = tagmaps.view(K, W * H, -1)
    tag_k = torch.stack([torch.gather(tags[..., i], dim=1, index=ind) for i in range(tags.size(2))], dim=2)

    x = ind % W
    y = ind // W
    ind_k = torch.stack((x, y), dim=2)

    c_joints = {
        'val_k': val_k,
        'tag_k': tag_k,
        'loc_k': ind_k,
    }

    return c_joints


def munkres_group(c_joints, val_th, tag_th, keys_index):
    num_keys = len(keys_index)
    instances = list()
    tag_ids = list()
    for k in keys_index:
        kth_vals = c_joints['val_k'][k]
        kth_tags = c_joints['tag_k'][k]
        kth_locs = c_joints['loc_k'][k]
        is_prior = kth_vals > val_th
        if is_prior.sum() == 0:
            continue
        prior_joints = torch.cat((kth_locs[is_prior], kth_vals[is_prior][:, None], kth_tags[is_prior]), dim=1)

        # init instances
        if len(instances) == 0:
            for joint in prior_joints:
                inst = torch.zeros(num_keys, prior_joints.shape[1]).type_as(prior_joints)
                inst[k] = joint
                instances.append(inst)
                tag_ids.append(joint[3:])
            continue

        # tags diff matrix
        src_tags = prior_joints[:, 3:]
        dst_tags = torch.stack(tag_ids, dim=0)
        tag_diff = src_tags[:, None, :] - dst_tags[None, :, :]
        tag_diff_norm = torch.norm(tag_diff, p=2, dim=2)

        # cost matrix
        cost = torch.round(tag_diff_norm / tag_th)
        src_scores = prior_joints[:, 2:3].expand(cost.shape[0], cost.shape[1])
        cost[cost > 1] = 1e5
        cost[cost <= 1] -= src_scores[cost <= 1]
        cost_np = cost.cpu().numpy()
        num_src, num_dst = cost.shape
        if num_src > num_dst:
            cost_np = np.concatenate([cost_np,
                                      np.ones([num_src, num_src - num_dst], dtype=np.float32) * 1e5], axis=1)

        # munkres matching
        m = Munkres()
        pairs = m.compute(cost_np.copy())
        pairs = np.array(pairs).astype(int)

        # integrate new joints
        for (src, dst) in pairs:
            if dst < num_dst and tag_diff_norm[src, dst] <= tag_th:
                inst = instances[dst]
                inst[k] = prior_joints[src]
                tag_id = inst[inst[:, 2] > 0][:, 3:].mean(dim=0)
                tag_ids[dst] = tag_id
            else:
                inst = torch.zeros(num_keys, prior_joints.shape[1]).type_as(prior_joints)
                inst[k] = prior_joints[src]
                instances.append(inst)
                tag_ids.append(prior_joints[src, 3:])

    return instances


def adjust_coordinates(instances, heatmaps):
    K, H, W = heatmaps.shape
    for inst_id, inst in enumerate(instances):
        for joint_id, joint in enumerate(inst):
            if joint[2] > 0:
                x, y = joint[0:2]
                xx, yy = int(x), int(y)
                tmp = heatmaps[joint_id]
                if tmp[min(H - 1, yy + 1), xx] > tmp[max(0, yy - 1), xx]:
                    y += 0.25
                else:
                    y -= 0.25

                if tmp[yy, min(W - 1, xx + 1)] > tmp[yy, max(0, xx - 1)]:
                    x += 0.25
                else:
                    x -= 0.25
                instances[inst_id][joint_id, 0] = x + 0.5
                instances[inst_id][joint_id, 1] = y + 0.5

    return instances


def refine(instance, heatmaps, tagmaps):
    H, W = heatmaps.shape[1:3]
    # mean tag of current detected people
    valid_ind = instance[:, 2] > 0
    prev_tag = instance[valid_ind][:, 3:].mean(dim=0)

    # add keypoint if it is not detected
    missing_key = torch.where(~valid_ind)[0].cpu().numpy()
    for idx in missing_key:
        _tag = tagmaps[idx]
        _heatmap = heatmaps[idx]

        # distance of all tag values with mean tag of current detected people
        distance_tag = torch.norm(_tag - prev_tag[None, None, :], p=2, dim=2)
        norm_heatmap = _heatmap - torch.round(distance_tag)

        # find maximum position
        max_ind = int(torch.argmax(norm_heatmap))
        y, x = np.unravel_index(max_ind, _heatmap.shape)
        xx = x.copy()
        yy = y.copy()

        # detection score at maximum position
        val = _heatmap[y, x].item()
        # offset by 0.5
        x += 0.5
        y += 0.5

        # add a quarter offset
        if _heatmap[yy, min(W - 1, xx + 1)] > _heatmap[yy, max(0, xx - 1)]:
            x += 0.25
        else:
            x -= 0.25

        if _heatmap[min(H - 1, yy + 1), xx] > _heatmap[max(0, yy - 1), xx]:
            y += 0.25
        else:
            y -= 0.25

        instance[idx, 0:3] = torch.tensor([x, y, val]).type_as(instance)

    return instance


def project2pri(instances, center, scale, output_size):
    trans = get_affine_transform(center, scale, 0, output_size, inv=True)
    target_instances = list()
    for p in range(len(instances)):
        ins = instances[p]
        target_ins = np.zeros([ins.shape[0], 3], dtype=np.float)
        target_ins[:, -1] = ins[:, 2]
        for k in range(len(target_ins)):
            if target_ins[k, -1] > 0:
                target_ins[k, 0:2] = warp_affine_joints(ins[k, 0:2], trans)
        target_instances.append(target_ins)

    return target_instances


def post_processing(heatmaps, tagmaps, eval_cfg):
    # project feature maps into image size
    heatmaps, tagmaps = project2image(heatmaps,
                                      tagmaps,
                                      projected_size=eval_cfg['base_size'])

    # aggregate flipped feature maps
    heatmaps, tagmaps = flip_aggregation(heatmaps,
                                         tagmaps,
                                         with_flip=eval_cfg['with_flip'],
                                         flip_index=eval_cfg['flip_index'])

    # apply nms for heatmaps
    filtered_hms = nms(heatmaps, base_kernel_size=eval_cfg['nms_kernel_size'])

    # sample top K candidates
    c_joints = sample_candidates(filtered_hms,
                                 tagmaps,
                                 top_k=eval_cfg['max_num_people'])

    # group candidates into instances
    instances = munkres_group(c_joints,
                              val_th=eval_cfg['val_th'],
                              tag_th=eval_cfg['tag_th'],
                              keys_index=eval_cfg['inference_channel'])

    # adjust the coordinates for better accuracy.
    instances = adjust_coordinates(instances, heatmaps)

    # calculate scores
    scores = [inst[:, 2].mean().item() for inst in instances]

    # identify missing joints in initial keypoint predictions
    if eval_cfg['with_refine'] is True:
        for i, inst in enumerate(instances):
            instances[i] = refine(inst, heatmaps, tagmaps)

    # to numpy
    instances = [inst.cpu().numpy() for inst in instances]

    # project coordinates to primitive size
    instances = project2pri(instances,
                            np.array(eval_cfg['center']),
                            np.array(eval_cfg['scale']),
                            eval_cfg['base_size'][::-1])

    return instances, scores


def inference(model, data, eval_cfg):
    if eval_cfg['with_flip']:
        input = data['image']
        flipped_input = torch.flip(input, [3])
        input = torch.cat([input, flipped_input], dim=0)
    else:
        input = data['image']

    img_metas = data['img_metas'][0]
    eval_cfg.update(img_metas)
    eval_cfg.update(base_size=data['image'].shape[2:])

    with torch.no_grad():
        heatmaps, tagmaps = model(input, phase='inference')

    # post processing
    preds, scores = post_processing(heatmaps=heatmaps, tagmaps=tagmaps, eval_cfg=eval_cfg)

    return preds, scores


def eval(model, data_loader, eval_cfg, gpu_id, distributed=False):
    # test
    model = model.eval()
    results = list()

    if not distributed or gpu_id == 0:
        prog_bar = ProgressBar(len(data_loader))

    for i, data in enumerate(data_loader):
        # format data
        data['image'] = data.pop('image').to('cuda:{}'.format(gpu_id))

        # inference
        preds, scores = inference(model, data, eval_cfg)
        results.append(dict(
            preds=preds,
            scores=scores,
            image_path=data['img_metas'][0]['image_file']
        ))

        if not distributed or gpu_id == 0:
            prog_bar.update()

    return results

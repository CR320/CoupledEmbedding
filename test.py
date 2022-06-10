import os
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.multiprocessing.queue import Queue
from mmcv import Config
from models import build_model
from models.evaluate import eval
from datasets import build_dataset, build_val_loader
from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path of config file')
    parser.add_argument('mode_path', type=str,
                        help='Path of checkpoint file')
    parser.add_argument('--set', type=str, default='test',
                        help='test or val')
    parser.add_argument('--out_dir', type=str, default='./output',
                        help='saving directory')
    parser.add_argument('--distributed', type=bool, default=False,
                        help='if True, use Distributed Data Parallel')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus for training')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--nr', type=int, default=0,
                        help='ranking within the nodes for distributed training')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='number of nodes for distributed training')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                        help='master address used to set up distributed training')
    parser.add_argument('--master_port', type=str, default='1234',
                        help='master port used to set up distributed training')
    return parser.parse_args()


def main_worker(gpu, cfg, args, results_queue=None):
    # init environment
    if args.distributed:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        rank = args.nr * args.num_gpus + gpu
        torch.cuda.set_device(rank % args.num_gpus)
        dist.init_process_group(
            backend=args.backend,
            init_method='env://',
            world_size=args.num_gpus * args.num_nodes,
            rank=rank
        )

    # init logger
    logger = get_logger(__name__.split('.')[0])

    # init dataset and loader
    logger.info('init dataset...')
    if args.set == 'test':
        dataset = build_dataset(cfg.set_cfg.test, cfg.data_cfg, is_training=False)
    elif args.set == 'val':
        dataset = build_dataset(cfg.set_cfg.val, cfg.data_cfg, is_training=False)
    else:
        raise ValueError('Unknown set name: {}'.format(args.set))

    data_loader = build_val_loader(dataset, distributed=args.distributed)
    logger.info('Validation data loader: there are {} samples in each loader'.format(len(data_loader)))

    # init model
    logger.info('init model...')
    model = build_model(cfg.model)
    checkpoint = torch.load(args.mode_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    gpu_id = gpu if args.distributed else gpu[0]
    model.to('cuda:{}'.format(gpu_id)).eval()

    # evaluate
    logger.info('testing...')
    results = eval(model, data_loader, cfg.eval_cfg, gpu_id, args.distributed)

    if args.distributed:
        if gpu_id != 0:
            results_queue.put_nowait(results)
        else:
            for _ in range(args.num_gpus - 1):
                results += results_queue.get(block=True)
        dist.barrier()

    if not args.distributed or gpu_id == 0:
        _, info_str = dataset.evaluate(results, args.out_dir, metric='mAP')
        logger.info('Validation Result: {}'.format(info_str))


if __name__ == '__main__':
    # load args and configs
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.backbone.pop('pre_weights')
    os.makedirs(args.out_dir, exist_ok=True)

    # launch main worker
    if args.distributed:
        results_queue = Queue(maxsize=args.num_gpus - 1, ctx=mp.get_context(method='spawn'))
        mp.spawn(main_worker, nprocs=args.num_gpus, args=(cfg, args, results_queue))
    else:
        main_worker([i for i in range(args.num_gpus)], cfg, args)
import os
import time
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.multiprocessing.queue import Queue
from mmcv import Config
from mmcv.utils import collect_env
from utils import set_random_seed, save_model, get_logger, SmoothedLossContainer, WarmupMultiStepLR
from datasets import build_dataset, build_loader, build_val_loader
from models import build_trainer
from models.evaluate import eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='path of config file')
    parser.add_argument('--distributed', type=bool, default=False,
                        help='if True, use Distributed Data Parallel')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed to be used (also set in torch & numpy)')
    parser.add_argument('--deterministic', type=bool, default=True,
                        help='if True, use deterministic convolution algorithms')
    parser.add_argument('--benchmark', type=bool, default=False,
                        help='if True, use benchmark')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='logger level for rank 0')
    parser.add_argument('--work_dir', type=str, default='work_dir/debug',
                        help="directory to save logs and models")
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


def training(epoch, model, data_loader, optimizer, lr_scheduler, logger, loss_container):
    model.train()
    for i, data in enumerate(data_loader):
        # forward
        losses = model(data)
        loss_container.update(losses)

        # backward
        total_loss = sum(losses.values())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # print loss
        if (i + 1) % loss_container.log_interval == 0:
            loss_values = loss_container.fetch_mean()
            loss = ','.join(['{}:{:.3e}'.format(k, loss_values[k]) for k in loss_values])
            logger.info('Epoch:[{}][{}/{}]'.format(epoch, i+1, len(data_loader)) +
                        '  lr:{:.3e}'.format(optimizer.param_groups[0]['lr']) + '  ' + loss)


def main_worker(gpu, cfg, args, results_queue=None):
    # init environment
    set_random_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = args.benchmark
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
    log_file = os.path.join(cfg.work_dir, 'run.log')
    logger = get_logger(__name__.split('.')[0], log_file, args.log_level)

    # collect environment info
    dash_line = '-' * 60
    env_info_dict = collect_env()
    env_info = '\n'.join(['{}:{}'.format(k, v) for k, v in env_info_dict.items()])
    logger.info('Environment info:\n' + dash_line + '\n' + env_info + '\n' + dash_line)

    # init dataset
    set_info = '\n'.join(['{}:{}'.format(k, v) for k, v in cfg.set_cfg.items()])
    logger.info('Set info:\n' + dash_line + '\n' + set_info + '\n' + dash_line)

    # init training data loader
    dataset = build_dataset(cfg.set_cfg.train, cfg.data_cfg, is_training=True)
    data_loader = build_loader(cfg.set_cfg, args.distributed, dataset, args.random_seed, args.num_gpus)
    logger.info('Training data loader: iterations in each epoch are {}'.format(len(data_loader)))

    # init validation data loader
    val_data = build_dataset(cfg.set_cfg.val, cfg.data_cfg, is_training=False)
    val_loader = build_val_loader(val_data, distributed=args.distributed)
    logger.info('Validation data loader: there are {} samples in each loader'.format(len(val_loader)))

    # init model
    model_info = '\n'.join(['{}:{}'.format(k, v) for k, v in cfg.model.items()])
    logger.info('Trainer info:\n' + dash_line + '\n' + model_info + '\n' + dash_line)
    model = build_trainer(cfg.trainer)

    # put model on gpus
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])
    else:
        model = DataParallel(model.cuda(gpu[0]), device_ids=gpu)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    logger.info('number of params:{}'.format(n_parameters))
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]},
    ]

    # init optimizer
    opt_args = cfg.solver.optimizer
    opt_method = getattr(torch.optim, opt_args.type)
    opt_args.pop('type')
    opt_args['params'] = param_dicts
    optimizer = opt_method(**opt_args)

    # init learning rate scheduler
    sch_args = cfg.solver.lr_scheduler
    sch_args['optimizer'] = optimizer
    sch_args['milestones'] = [v*len(data_loader) for v in sch_args['milestones']]
    lr_scheduler = WarmupMultiStepLR(**sch_args)

    # resume
    start_epoch = 1
    if cfg.solver.resume_from is not None:
        logger.info('Resume: load weight from {}'.format(cfg.solver.resume_from))
        checkpoint = torch.load(cfg.solver.resume_from, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'meta' in checkpoint:
            start_epoch = checkpoint['meta']['epoch']
        else:
            logger.warn('None mata info in checkpoint file')

    # init smoothed loss container
    slc = SmoothedLossContainer(log_key=cfg.solver.log_loss, log_interval=cfg.solver.log_interval)

    best_result = dict(score=-1, epoch=0, info='')
    gpu_id = gpu if args.distributed else gpu[0]
    for epoch in range(start_epoch, cfg.solver.total_epochs + 1):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        # training model
        training(epoch=epoch,
                 model=model,
                 data_loader=data_loader,
                 optimizer=optimizer,
                 lr_scheduler=lr_scheduler,
                 logger=logger,
                 loss_container=slc)

        # evaluate model
        if epoch % cfg.solver.eval_interval == 0:
            logger.info('Evaluating with {} Processes....'.format(args.num_gpus))
            results = eval(model.module.net, val_loader, cfg.eval_cfg, gpu_id, args.distributed)
            if args.distributed:
                if gpu_id != 0:
                    results_queue.put_nowait(results)
                else:
                    for _ in range(args.num_gpus - 1):
                        results += results_queue.get(block=True)

            filename = None
            if not args.distributed or gpu_id == 0:
                score, info_str = val_data.evaluate(results, args.work_dir, metric='mAP')
                logger.info('Validation Result: {}'.format(info_str))
                if score > best_result['score']:
                    best_result['score'] = score
                    best_result['epoch'] = epoch
                    best_result['info'] = info_str
                    filename = 'best.pth'

        else:
            filename = None

        meta = dict(epoch=epoch+1, iter=0)
        if args.distributed:
            if gpu_id == 0:
                save_model(model.module, optimizer, lr_scheduler, meta, cfg.work_dir, filename=filename)
            dist.barrier()
        else:
            save_model(model.module, optimizer, lr_scheduler, meta, cfg.work_dir, filename=filename)

    if not args.distributed or gpu_id == 0:
        logger.info('Best Validation Result: Epoch:{} | {}'.format(best_result['epoch'], best_result['info']))


if __name__ == '__main__':
    # load args and configs
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # create cache directory
    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    if args.work_dir is None:
        cfg.work_dir = os.path.join('work_dir', 'job_{}'.format(timestamp))
    else:
        cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)

    # dump config in work_dir
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))

    # launch main worker
    if args.distributed:
        results_queue = Queue(maxsize=args.num_gpus - 1, ctx=mp.get_context(method='spawn'))
        mp.spawn(main_worker, nprocs=args.num_gpus, args=(cfg, args, results_queue))
    else:
        main_worker([i for i in range(args.num_gpus)], cfg, args)

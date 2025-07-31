import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import re
import random
from functools import partial
from pathlib import Path
from collections import OrderedDict

from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets.datasets import Phoenix2014, collate_fn
from engine import train_one_epoch, evaluate_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import contextlib
from .models.model import Model


def get_args():
    parser = argparse.ArgumentParser('Training and evaluation script for Sign Language Recognition', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--headdim', default=64, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--expand', default=2, type=int)

    parser.add_argument('--head_drop_rate', type=float, default=0.1, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--model_ema_eval', action='store_true', default=False, help='Using ema to eval during training.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--temp_scale', type=float, default=0.)
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m5-mstd0.25', metavar='NAME')

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model_ema', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--prefix', default='', type=str, help='prefix for data')
    parser.add_argument('--gloss_dict_path', default='', type=str)
    parser.add_argument('--meta_dir_path', default='', type=str)
    parser.add_argument('--frame_interval', default=1, type=int)
    parser.add_argument('--use_heatmap', action='store_true', default=False)
    parser.add_argument('--kp_path', default='', type=str)
    parser.add_argument('--kernel_size', default=193, type=int)
    parser.add_argument('--sigma', default=32., type=float)
    # parser.add_argument('--data_set', default='Kinetics', choices=[
        # 'Kinetics', 'Kinetics_sparse', 
        # 'SSV2', 'UCF101', 'HMDB51', 'image_folder',
        # 'mitv1_sparse', 'LVU', 'COIN', 'Breakfast'
        # ], type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_best', action='store_true',
                        help='Whether test the best model')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--no_amp', action='store_true')
    parser.set_defaults(no_amp=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=True)
    parser.add_argument('--bf16', default=True, action='store_true')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    gloss_dict = np.load(args.gloss_dict_path, allow_pickle=True).item()

    dataset_train = Phoenix2014(
        prefix=args.prefix,
        gloss_dict=gloss_dict,
        input_size=256,
        crop_size=args.input_size,
        mode='train',
        meta_dir_path=args.meta_dir_path,
        frame_interval=args.frame_interval,
        temp_scale=args.temp_scale,
        use_heatmap=args.use_heatmap,
        kp_path=args.kp_path,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
        args=args
    )
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val = Phoenix2014(
            prefix=args.prefix,
            gloss_dict=gloss_dict,
            input_size=256,
            crop_size=args.input_size,
            mode='dev',
            meta_dir_path=args.meta_dir_path,
            frame_interval=args.frame_interval,
            temp_scale=args.temp_scale,
            use_heatmap=args.use_heatmap,
            kp_path=args.kp_path,
            kernel_size=args.kernel_size,
            sigma=args.sigma,
            args=args
        )
    dataset_test = Phoenix2014(
        prefix=args.prefix,
        gloss_dict=gloss_dict,
        input_size=256,
        crop_size=args.input_size,
        mode='test',
        meta_dir_path=args.meta_dir_path,
        frame_interval=args.frame_interval,
        temp_scale=args.temp_scale,
        use_heatmap=args.use_heatmap,
        kp_path=args.kp_path,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
        args=args
    )

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    collate_func = collate_fn

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=True
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=2 * args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=collate_func,
            persistent_workers=True
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=2 * args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=collate_func,
            persistent_workers=True
        )
    else:
        data_loader_test = None

    model = Model(
        img_size=args.input_size, patch_size=args.patch_size,
        d_model=args.d_model, n_layer=args.depth, d_intermediate=args.d_model * 4, channels=3,
        gloss_dict=gloss_dict, drop_path_rate=args.drop_path, head_drop_rate=args.head_drop_rate,
        ssm_cfg={'spatial': {'expand': args.expand, 'headdim': args.headdim}, 'temporal': {'expand': args.expand, 'headdim': args.headdim}},
        attn_cfg={'spatial': {'num_heads': args.d_model // args.headdim}, 'temporal': {}},
        attn_layer_idx={'spatial': [4], 'temporal': []},
        rms_norm=False, fused_add_norm=True, residual_in_fp32=True
    )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        if 'head.weight' in checkpoint_model.keys():
            print("Removing head from pretrained checkpoint")
            del checkpoint_model['head.weight']
            del checkpoint_model['head.bias']
                    
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            match = re.match(r'layers\.(\d+)\.(.+)', key)
            if not match:
                new_dict[key] = checkpoint_model[key]
            else:
                idx, tail = match.groups()
                if tail.startswith('mixer.'):
                    new_dict[f"layers.{idx}.spatial_mixer.{tail[len('mixer.'):]}"] = checkpoint_model[key]
                elif tail.startswith('norm2.'):
                    new_dict[f"layers.{idx}.mlp_norm.{tail[len('norm2.'):]}"] = checkpoint_model[key]
                elif tail.startswith('norm.'):
                    new_dict[f"layers.{idx}.spatial_norm.{tail[len('norm.'):]}"] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device, memory_format=torch.channels_last)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    amp_autocast = contextlib.nullcontext()
    loss_scaler = "none"
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)

        if not args.no_amp:
            print(f"Use bf16: {args.bf16}")
            dtype = torch.bfloat16 if args.bf16 else torch.float16
            amp_autocast = torch.cuda.amp.autocast(dtype=dtype)
            loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # if mixup_fn is not None:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CTCLoss(blank=0)

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        val_stats = evaluate_one_epoch(
            data_loader_val, model, device, amp_autocast,
            ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
        )
        val_ema_stats = evaluate_one_epoch(
            data_loader_val, model_ema.ema, device, amp_autocast,
            ds=False, no_amp=args.no_amp, bf16=args.bf16,
        )
        test_stats = evaluate_one_epoch(
            data_loader_test, model, device, amp_autocast,
            ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
        )
        test_ema_stats = evaluate_one_epoch(
            data_loader_test, model_ema.ema, device, amp_autocast,
            ds=False, no_amp=args.no_amp, bf16=args.bf16,
        )

        exit(0)
        # preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        # test_stats = final_test()
        # torch.distributed.barrier()
        # if global_rank == 0:
        #     print("Start merging results...")
        #     final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        #     print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        #     log_stats = {'Final top-1': final_top1,
        #                 'Final Top-5': final_top5}
        #     if args.output_dir and utils.is_main_process():
        #         with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #             f.write(json.dumps(log_stats) + "\n")
        # exit(0)
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_wer = 100.
    if args.model_ema and args.model_ema_eval:
        best_wer_ema = 100.
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, amp_autocast, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            no_amp=args.no_amp, bf16=args.bf16
        )
        if args.output_dir and args.save_ckpt:
            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            #     utils.save_model(
            #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #         loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_name='latest', model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = evaluate_one_epoch(
                data_loader_val, model, device, amp_autocast,
                ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
            )
            timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{timestep}] WER of the network on the {len(dataset_val)} val videos: {test_stats['wer']:.2f}")
            if best_wer > test_stats["wer"]:
                best_wer = test_stats["wer"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_name='best', model_ema=model_ema)

            print(f'Best WER: {best_wer:.2f}%')
            if log_writer is not None:
                log_writer.update(val_wer=test_stats['wer'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate_one_epoch(
                    data_loader_val, model_ema.ema, device, amp_autocast,
                    ds=False, no_amp=args.no_amp, bf16=args.bf16,
                )
                timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{timestep}] WER of the EMA network on the {len(dataset_val)} val videos: {test_stats_ema['wer']:.2f}")
                if best_wer_ema > test_stats_ema["wer"]:
                    best_wer_ema = test_stats_ema["wer"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, model_name='best-ema', model_ema=model_ema)

                print(f'Best WER: {best_wer_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_wer_ema=test_stats_ema['wer'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if args.test_best:
        print("Auto testing the best model")
        args.eval = True
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    test_stats = evaluate_one_epoch(
        data_loader_test, model, device, amp_autocast,
        ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
    )
    torch.distributed.barrier()

    if args.output_dir and utils.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)

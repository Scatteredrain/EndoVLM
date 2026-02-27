# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    mode = args.mode

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # samples 是 dict

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs = samples["images"].to(device, non_blocking=True)
        full_texts = samples["full_texts"].to(device, non_blocking=True)
        sub_texts = samples["sub_texts"].to(device, non_blocking=True)
        text_normal_flags = samples["normal_flags"].to(device, non_blocking=True)
        text_anatomy_flags = samples["anatomy_flags"].to(device, non_blocking=True)
        image_counts = samples["image_counts"]
        sub_text_counts = samples["sub_text_counts"]
        # samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            loss, loss_mae, loss_clip_global, loss_clip_FG, unique_selected_patch_idxs = \
                model(imgs, full_texts, sub_texts, text_normal_flags, text_anatomy_flags,
                    image_counts=image_counts, sub_text_counts=sub_text_counts, 
                        mode='ablation' if args.ablation else 'both', K=args.pooing_topk, mask_ratio=args.mask_ratio)
        loss_value = loss.item()
        ## specifical loss
        loss_mae_value = float(loss_mae.detach().item()) if torch.is_tensor(loss_mae) else float(loss_mae)
        loss_clip_global_value = float(loss_clip_global.detach().item()) if torch.is_tensor(loss_clip_global) else float(loss_clip_global)
        loss_clip_FG_value = float(loss_clip_FG.detach().item()) if torch.is_tensor(loss_clip_FG) else float(loss_clip_FG)

        # ##
        # if args.debug:
        #     print("loss_mae_reduce: ", loss_mae_value)
        #     print("loss_clip_global_reduce: ", loss_clip_global_value)
        #     print("loss_clip_FG_reduce: ", loss_clip_FG_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        metric_logger.update(loss_mae=loss_mae_value)
        metric_logger.update(loss_clip_global=loss_clip_global_value)
        metric_logger.update(loss_clip_FG=loss_clip_FG_value)

        loss_mae_reduce = misc.all_reduce_mean(loss_mae_value)
        loss_clip_global_reduce = misc.all_reduce_mean(loss_clip_global_value)
        loss_clip_FG_reduce = misc.all_reduce_mean(loss_clip_FG_value)


        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            
            log_writer.add_scalar('lr', lr, epoch_1000x)

            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_mae', loss_mae_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_clip_global', loss_clip_global_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_clip_FG', loss_clip_FG_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
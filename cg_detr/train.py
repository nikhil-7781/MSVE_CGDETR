import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cg_detr.config import BaseOptions
from cg_detr.start_end_dataset import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
from cg_detr.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def prepare_batch_inputs_msve(batch, device, non_blocking=False, use_msve=False, raw_video_layout='BTCHW'):
    """
    Enhanced batch preparation function that handles both traditional features and raw video for MSVE.

    Args:
        batch: dataloader output. Either (batch_meta, batched_model_inputs) or
               (batch_meta, batched_model_inputs, raw_video_batch).
        device: torch device
        non_blocking: whether to use non-blocking .to(device)
        use_msve: if True, try to attach raw video tensors into model_inputs
        raw_video_layout: expected layout of raw_video_batch ('BTCHW' or 'BCTHW').
                          Collate returns BTCHW by default in our suggested collate.

    Returns:
        model_inputs: dict ready for model.forward (may include 'src_vid_raw' and 'src_vid_raw_mask')
        targets: targets dict (or None)
    """
    # Unpack batch safely
    if len(batch) == 3:
        batch_meta, batched_model_inputs, raw_video_batch = batch
    else:
        batch_meta, batched_model_inputs = batch
        raw_video_batch = None

    # Call the existing prepare function with the batched_model_inputs dict (important!)
    model_inputs, targets = prepare_batch_inputs(batched_model_inputs, device, non_blocking)

    # Default: no raw video attached
    model_inputs["src_vid_raw"] = None
    model_inputs["src_vid_raw_mask"] = None

    # If MSVE enabled and we have a raw_video tensor, prepare it
    if use_msve and (raw_video_batch is not None):
        rv = raw_video_batch

        # Convert numpy -> torch if necessary
        if isinstance(rv, np.ndarray):
            rv = torch.from_numpy(rv)

        if not isinstance(rv, torch.Tensor):
            logger.warning("Raw video batch not recognized (not numpy or torch.Tensor). Skipping MSVE for this batch.")
            model_inputs["src_vid_raw"] = None
        else:
            # Ensure float32 (model likely expects float inputs)
            if rv.dtype != torch.float32:
                rv = rv.float()

            # Move to device
            rv = rv.to(device, non_blocking=non_blocking)

            # rv layout: expected by our pipeline is (B, T, C, H, W) (BTCHW)
            # If your MSVE expects (B, C, T, H, W) pass raw_video_layout='BCTHW'
            if raw_video_layout == 'BCTHW':
                # Permute BTCHW -> BCTHW
                rv = rv.permute(0, 2, 1, 3, 4).contiguous()

            model_inputs["src_vid_raw"] = rv

            # Also attach raw video mask if provided by collate (batched_model_inputs)
            raw_mask = batched_model_inputs.get('raw_video_mask', None)
            if raw_mask is not None:
                # raw_mask may be numpy or torch; convert & move to device
                if isinstance(raw_mask, np.ndarray):
                    raw_mask = torch.from_numpy(raw_mask)
                if isinstance(raw_mask, torch.Tensor):
                    # boolean mask expected
                    if raw_mask.dtype != torch.bool:
                        raw_mask = raw_mask.bool()
                    raw_mask = raw_mask.to(device, non_blocking=non_blocking)
                    model_inputs["src_vid_raw_mask"] = raw_mask
                else:
                    model_inputs["src_vid_raw_mask"] = None
            else:
                # If no mask available, construct one from rv (all ones up to T)
                T = rv.shape[1] if raw_video_layout == 'BCTHW' else rv.shape[1]
                # Here we assume all frames are valid; if padding exists you'd want the mask from collate
                model_inputs["src_vid_raw_mask"] = torch.ones((rv.size(0), T), dtype=torch.bool, device=device)

    # If MSVE not used or no raw video, fields remain None
    return model_inputs, targets



def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    # Check if model uses MSVE
    use_msve = hasattr(model, 'use_msve') and model.use_msve
    
    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()

        # Use enhanced batch preparation for MSVE support
        model_inputs, targets = prepare_batch_inputs_msve(
            batch, opt.device, non_blocking=opt.pin_memory, use_msve=use_msve
        )

        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()

        outputs = model(**model_inputs, targets=targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Log MSVE-specific information if available
        if use_msve and "motion_energy" in outputs:
            # Log motion energy statistics
            motion_energy = outputs["motion_energy"]
            if motion_energy is not None:
                motion_stats = {
                    "motion_energy_mean": motion_energy.mean().item(),
                    "motion_energy_std": motion_energy.std().item(),
                    "motion_energy_max": motion_energy.max().item(),
                    "motion_energy_min": motion_energy.min().item()
                }
                # Add to tensorboard every 100 batches to avoid overhead
                if batch_idx % 100 == 0:
                    for stat_name, stat_val in motion_stats.items():
                        tb_writer.add_scalar(f"Train_MSVE/{stat_name}", stat_val, 
                                           epoch_i * num_training_examples + batch_idx)
        
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    # Log MSVE usage
    if use_msve:
        logger.info(f"Training with Multi-Stream Video Encoder (MSVE) enabled")

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    # Check if MSVE is enabled
    use_msve = hasattr(model, 'use_msve') and model.use_msve
    if use_msve:
        logger.info("Multi-Stream Video Encoder (MSVE) is ENABLED")
        logger.info(f"MSVE feature dimension: {getattr(model, 'msve_feature_dim', 'Unknown')}")
        
        # Log MSVE configuration
        if hasattr(model, 'msve'):
            msve_params = sum(p.numel() for p in model.msve.parameters())
            logger.info(f"MSVE parameters: {msve_params:,}")
    else:
        logger.info("Using traditional pre-extracted features (MSVE disabled)")

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = opt.eval_epoch
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            if opt.dset_name in ['hl']:
                stop_score = metrics["brief"]["MR-full-mAP"]
            else:
                stop_score = (metrics["brief"]["MR-full-R1@0.7"] + metrics["brief"]["MR-full-R1@0.5"]) / 2

            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if opt.debug:
            break

    tb_writer.close()


def train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    # Check if MSVE is enabled
    use_msve = hasattr(model, 'use_msve') and model.use_msve
    if use_msve:
        logger.info("Multi-Stream Video Encoder (MSVE) is ENABLED for highlight detection")
        logger.info(f"MSVE feature dimension: {getattr(model, 'msve_feature_dim', 'Unknown')}")
    else:
        logger.info("Using traditional pre-extracted features for highlight detection (MSVE disabled)")

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            stop_score = metrics["brief"]["mAP"]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()


def validate_msve_configuration(opt, model):
    """
    Validate MSVE configuration and provide helpful warnings/info
    """
    use_msve = hasattr(model, 'use_msve') and model.use_msve
    
    if use_msve:
        logger.info("=" * 50)
        logger.info("MSVE CONFIGURATION VALIDATION")
        logger.info("=" * 50)
        
        # Check if raw video data path is configured
        if not hasattr(opt, 'raw_video_path') or opt.raw_video_path is None:
            logger.warning("MSVE is enabled but no raw_video_path specified in config!")
            logger.warning("Make sure your dataset can provide raw video frames.")
        
        # Check memory requirements
        batch_size = opt.bsz
        max_frames = opt.max_v_l
        estimated_memory_gb = (batch_size * max_frames * 3 * 224 * 224 * 4) / (1024**3)  # Assuming 224x224 RGB
        logger.info(f"Estimated GPU memory for raw video: {estimated_memory_gb:.2f} GB")
        
        if estimated_memory_gb > 8:
            logger.warning("High memory usage expected! Consider reducing batch_size or max_v_l")
        
        # Validate MSVE parameters
        if hasattr(model, 'msve'):
            msve_params = sum(p.numel() for p in model.msve.parameters())
            total_params = sum(p.numel() for p in model.parameters())
            msve_ratio = msve_params / total_params
            logger.info(f"MSVE parameters: {msve_params:,} ({msve_ratio:.1%} of total)")
        
        logger.info("=" * 50)
    
    return use_msve


def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        dset_domain=opt.dset_domain,
    )
    
    # Add raw video path if MSVE is enabled
    if hasattr(opt, 'use_msve') and opt.use_msve:
        if hasattr(opt, 'raw_video_path'):
            dataset_config['raw_video_path'] = opt.raw_video_path
        else:
            logger.warning("MSVE enabled but no raw_video_path specified!")
    
    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)

    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        eval_dataset = StartEndDataset(**dataset_config)
    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    
    # Validate MSVE configuration
    validate_msve_configuration(opt, model)
    
    logger.info("Start Training...")
    
    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum', 'youtube_uni']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug, opt


if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
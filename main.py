import copy
import logging
import os
import setproctitle
import sys
import time

import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.data.sampler import BatchSampler
from torch.utils.tensorboard import SummaryWriter

from utils.mixup import mixup_data, mixup_criterion
from utils import prefetch
from datasets.data import (
    load_dataset,
    get_bootstrapping_datasets,
    get_ssl_datasets,
    get_final_datasets,
    get_split_dataset_datasets,
    get_test_aug_datasets,
)
from ds_partition import select_samples, split_dataset_2
from models.models import create_model
from utils import label_smoothing, lr_scheduler, log_dataset_info, metric
from utils.pseudo_labelling import get_logits_labels_combined
import utils.utils as utils
import workspace


logger = logging.getLogger(__name__)


# ---------------
# --- Loaders ---
# ---------------


def get_random_data_loader(
    dset,
    batch_size=None,
    num_workers=4,
    replacement=True,
    num_iters=None,
    drop_last=True,
):

    num_samples = batch_size * num_iters
    ds = dset.ds

    if ds.weights is not None:
        data_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            ds.weights[ds.indices], num_samples, replacement,
        )
        logger.warning("USING WEIGHTS")
    else:
        data_sampler = torch.utils.data.sampler.RandomSampler(
            dset, replacement, num_samples
        )
        logger.warning("NOT USING WEIGHTS")

    batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
    return torch.utils.data.DataLoader(
        dset, batch_sampler=batch_sampler, num_workers=num_workers,
    )


def get_bootstrapping_loaders(config, train_dataset, test_dataset):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        config.bootstrapping.batch_size,
        num_workers=config.bootstrapping.num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        config.bootstrapping.batch_size,
        num_workers=config.bootstrapping.num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader


def get_split_dataset_loader(config, dataset):

    data_loader = torch.utils.data.DataLoader(
        dataset,
        config.bootstrapping.batch_size,
        num_workers=config.bootstrapping.num_workers,
        shuffle=False,
        drop_last=False,
    )

    return data_loader


def get_test_aug_loader(config, dataset):

    data_loader = torch.utils.data.DataLoader(
        dataset,
        config.bootstrapping.batch_size,
        num_workers=config.bootstrapping.num_workers,
        shuffle=False,
        drop_last=False,
    )

    return data_loader


# def get_ssl_loaders(config, clean_dataset, noisy_dataset, test_dataset):

#     clean_loader = get_random_data_loader(
#         clean_dataset,
#         config.ssl.batch_size,
#         num_iters=config.ssl.num_train_iter,
#         num_workers=config.ssl.num_workers,
#     )

#     noisy_loader = get_random_data_loader(
#         noisy_dataset,
#         config.ssl.batch_size * config.ssl.uratio,
#         num_iters=config.ssl.num_train_iter,
#         num_workers=4 * config.ssl.num_workers,
#     )

#     test_loader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=config.ssl.eval_batch_size,
#         num_workers=config.ssl.num_workers,
#     )

#     return clean_loader, noisy_loader, test_loader



def get_ssl_round_loaders(
    config, clean_dataset, noisy_dataset, noisy_test_dataset, test_dataset
):

    clean_loader = get_random_data_loader(
        clean_dataset,
        config.ssl.batch_size,
        num_iters=config.ssl.iterations_per_round,
        num_workers=config.ssl.num_workers,
    )

    noisy_loader = get_random_data_loader(
        noisy_dataset,
        config.ssl.batch_size * config.ssl.uratio,
        num_iters=config.ssl.iterations_per_round,
        num_workers=4 * config.ssl.num_workers,
    )

    noisy_test_loader = torch.utils.data.DataLoader(
        noisy_test_dataset,
        batch_size=config.ssl.eval_batch_size,
        num_workers=config.ssl.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.ssl.eval_batch_size,
        num_workers=config.ssl.num_workers,
    )

    return clean_loader, noisy_loader, noisy_test_loader, test_loader



def get_ssl_round_test_loaders(
    config, iterations, clean_dataset, noisy_dataset, noisy_test_dataset, test_dataset
):

    clean_loader = get_random_data_loader(
        clean_dataset,
        config.ssl.batch_size,
        num_iters=iterations,
        num_workers=config.ssl.num_workers,
    )

    noisy_loader = get_random_data_loader(
        noisy_dataset,
        config.ssl.batch_size * config.ssl.uratio,
        num_iters=iterations,
        num_workers=4 * config.ssl.num_workers,
    )

    noisy_test_loader = torch.utils.data.DataLoader(
        noisy_test_dataset,
        batch_size=config.ssl.eval_batch_size,
        num_workers=config.ssl.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.ssl.eval_batch_size,
        num_workers=config.ssl.num_workers,
    )

    return clean_loader, noisy_loader, noisy_test_loader, test_loader


def get_final_loaders(config, train_dataset, test_dataset):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        config.final_model.batch_size,
        num_workers=config.final_model.num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        config.final_model.batch_size,
        num_workers=config.final_model.num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader


# ----------------
# --- Training ---
# ----------------


def run_model_normal(model, x, n_labels, target, criterion, mixup=False, mix_alpha=1.0):

    if mixup:
        x, n_labels, y_a, y_b, lam = mixup_data(x, n_labels, target, alpha=mix_alpha)
        output = model(x, n_labels)
        loss = mixup_criterion(criterion, output, y_a, y_b, lam)

    else:
        output = model(x, n_labels)
        loss = criterion(output, target)

    return output, loss


def run_model_ssl(
    config, model, x, n_labels, target, x_ulb_w, x_ulb_s, n_ulb, criterion
):

    num_lb = x.shape[0]
    num_ulb = x_ulb_w.shape[0]
    assert num_ulb == x_ulb_s.shape[0]

    if config.ssl.is_mixup:
        x, mixed_n_labels, y_a, y_b, lam = mixup_data(
            x, n_labels, target, alpha=config.mix_alpha
        )
        mixed_n_lb_zeroed = utils.set_rows_to_null(
            mixed_n_labels, config.null_label_type
        )
        n_ulb_zeroed = utils.set_rows_to_null(n_ulb, config.null_label_type)
        all_inputs = torch.cat((x, x_ulb_w, x_ulb_s))

        if config.ssl.null_nls_for_pseudo_generation:
            all_n_labels = torch.cat((mixed_n_lb_zeroed, n_ulb_zeroed, n_ulb_zeroed))
        else:
            all_n_labels = torch.cat((mixed_n_lb_zeroed, n_ulb, n_ulb_zeroed))

    else:
        n_lb_zeroed = utils.set_rows_to_null(n_labels, config.null_label_type)
        n_ulb_zeroed = utils.set_rows_to_null(n_ulb, config.null_label_type)
        all_inputs = torch.cat((x, x_ulb_w, x_ulb_s))

        if config.ssl.null_nls_for_pseudo_generation:
            all_n_labels = torch.cat((n_lb_zeroed, n_ulb_zeroed, n_ulb_zeroed))
        else:
            all_n_labels = torch.cat((n_lb_zeroed, n_ulb, n_ulb_zeroed))

    logits = model(all_inputs, all_n_labels)
    logits_x_lb = logits[:num_lb]
    logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

    if config.ssl.is_mixup:
        sup_loss = mixup_criterion(lambda logits, target: utils.ce_loss(logits, target, reduction="mean"), logits_x_lb, y_a, y_b, lam)
    else:
        sup_loss = utils.ce_loss(logits_x_lb, target, reduction="mean")

    unsup_loss, mask, select, pseudo_lb = utils.consistency_loss(
        logits_x_ulb_s,
        logits_x_ulb_w,
        "ce",
        config.ssl.T,
        config.ssl.p_cutoff,
        use_hard_labels=config.ssl.hard_label,
    )

    total_loss = sup_loss + config.ssl.ulb_loss_ratio * unsup_loss

    return logits, total_loss


def bootstrapping_epoch(
    config, model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
):

    model.train()
    optimizer.zero_grad()

    log = {
        "batch_time": metric.AverageMeter("Batch Time", ":6.3f"),
        "data_time": metric.AverageMeter("Data Time", ":6.3f"),
        "ce_loss": metric.AverageMeter("Cross Entropy", ":6.3f"),
        "top_1_noisy": metric.AverageMeter("Acc (Noisy)", ":6.2f"),
        "top_1_true": metric.AverageMeter("Acc (True)", ":6.2f"),
    }

    total_iters = len(train_loader)
    progress = metric.ProgressMeter(
        total_iters, log["batch_time"], log["data_time"], log["ce_loss"]
    )
    end = time.time()

    train_prefetcher = prefetch.data_prefetcher(train_loader, device)
    _, images, t_labels, l_labels, n_labels = train_prefetcher.next()
    i = 0

    while images is not None:

        i += 1
        scheduler(optimizer, i, epoch)

        log["data_time"].update(time.time() - end)

        null_labels = utils.null_labels_like(n_labels, config.null_label_type)

        if config.is_amp:
            with amp.autocast():
                output, ce_loss = run_model_normal(
                    model,
                    images,
                    null_labels,
                    l_labels,
                    criterion,
                    config.bootstrapping.is_mixup,
                    config.mix_alpha,
                )
        else:
            output, ce_loss = run_model_normal(
                model,
                images,
                null_labels,
                l_labels,
                criterion,
                config.bootstrapping.is_mixup,
                config.mix_alpha,
            )

        batch_size_now = images.size(0)
        acc_noisy = metric.accuracy(output, l_labels, topk=(1,))
        acc_true = metric.accuracy(output, t_labels, topk=(1,))
        log["top_1_noisy"].update(acc_noisy[0].item(), batch_size_now)
        log["top_1_true"].update(acc_true[0].item(), batch_size_now)
        log["ce_loss"].update(ce_loss.mean().item(), batch_size_now)

        if config.is_amp:
            scaler.scale(ce_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            ce_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        log["batch_time"].update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.print(i)

        _, images, t_labels, l_labels, n_labels = train_prefetcher.next()

    logger.info(
        f'[TRAIN] {str(log["ce_loss"])}  |  {str(log["top_1_noisy"])}  |  {str(log["top_1_true"])}'
    )

    return log



def final_model_epoch(
    config, model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
):

    model.train()
    optimizer.zero_grad()

    log = {
        "batch_time": metric.AverageMeter("Batch Time", ":6.3f"),
        "data_time": metric.AverageMeter("Data Time", ":6.3f"),
        "ce_loss": metric.AverageMeter("Cross Entropy", ":6.3f"),
        "top_1_noisy": metric.AverageMeter("Acc (Noisy)", ":6.2f"),
        "top_1_true": metric.AverageMeter("Acc (True)", ":6.2f"),
    }

    total_iters = len(train_loader)
    progress = metric.ProgressMeter(
        total_iters, log["batch_time"], log["data_time"], log["ce_loss"]
    )
    end = time.time()

    train_prefetcher = prefetch.data_prefetcher(train_loader, device)
    _, images, t_labels, l_labels, n_labels = train_prefetcher.next()
    i = 0

    while images is not None:

        i += 1
        scheduler(optimizer, i, epoch)

        log["data_time"].update(time.time() - end)

        n_labels = utils.set_rows_to_null(
            n_labels, config.null_label_type
        )

        if config.is_amp:
            with amp.autocast():
                output, ce_loss = run_model_normal(
                    model,
                    images,
                    n_labels,
                    l_labels,
                    criterion,
                    config.bootstrapping.is_mixup,
                    config.mix_alpha,
                )
        else:
            output, ce_loss = run_model_normal(
                model,
                images,
                n_labels,
                l_labels,
                criterion,
                config.bootstrapping.is_mixup,
                config.mix_alpha,
            )

        batch_size_now = images.size(0)
        acc_noisy = metric.accuracy(output, l_labels, topk=(1,))
        acc_true = metric.accuracy(output, t_labels, topk=(1,))
        log["top_1_noisy"].update(acc_noisy[0].item(), batch_size_now)
        log["top_1_true"].update(acc_true[0].item(), batch_size_now)
        log["ce_loss"].update(ce_loss.mean().item(), batch_size_now)

        if config.is_amp:
            scaler.scale(ce_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            ce_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        log["batch_time"].update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.print(i)

        _, images, t_labels, l_labels, n_labels = train_prefetcher.next()

    logger.info(
        f'[TRAIN] {str(log["ce_loss"])}  |  {str(log["top_1_noisy"])}  |  {str(log["top_1_true"])}'
    )

    return log



def ssl_round_train(
    config,
    round_no,
    clean_loader,
    noisy_loader,
    noisy_test_loader,
    test_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    tb_log,
):

    model.train()
    optimizer.zero_grad()

    # @TODO: EMA will break with multiple rounds of training
    ema = utils.EMA(model, config.ssl.ema_m)
    ema.register()

    log = {
        "batch_time": metric.AverageMeter("Batch Time", ":6.3f"),
        "data_time": metric.AverageMeter("Data Time", ":6.3f"),
        "loss": metric.AverageMeter("Loss", ":6.3f"),
    }

    assert len(clean_loader) == len(noisy_loader)
    total_iters = config.ssl.training_rounds * config.ssl.iterations_per_round
    progress = metric.ProgressMeter(
        total_iters, log["batch_time"], log["data_time"], log["loss"]
    )

    # switch to train mode
    end = time.time()

    # prefetch data
    clean_prefetcher = prefetch.data_prefetcher(clean_loader, device)
    noisy_prefetcher = prefetch.data_prefetcher(noisy_loader, device)
    _, images, _, l_labels, n_labels = clean_prefetcher.next()
    _, (x_ulb_w, x_ulb_s), _, _, n_ulb = noisy_prefetcher.next()
    # i = round_no * config.ssl.iterations_per_round
    i = 0

    while images is not None:

        log["data_time"].update(time.time() - end)

        scheduler(optimizer, i)
        i += 1

        if config.is_amp:
            with amp.autocast():
                output, loss = run_model_ssl(
                    config,
                    model,
                    images,
                    n_labels,
                    l_labels,
                    x_ulb_w,
                    x_ulb_s,
                    n_ulb,
                    criterion,
                )
        else:
            output, loss = run_model_ssl(
                config,
                model,
                images,
                n_labels,
                l_labels,
                x_ulb_w,
                x_ulb_s,
                n_ulb,
                criterion,
            )

        # measure accuracy and record loss
        batch_size_now = images.size(0)
        log["loss"].update(loss.mean().item(), batch_size_now)

        if config.is_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # scheduler.step() # @TODO: This will break for multiple rounds of training
        ema.update()

        # measure elapsed time
        log["batch_time"].update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.print(i)
            tb_log.add_scalar(f"ssl/{round_no}/lr", optimizer.param_groups[0]["lr"], global_step=i)

        if i % config.ssl.num_eval_iter == 0:
            ema.apply_shadow()
            logger.info(f"Test Iteration: {i}")
            test_log = validate(config, model, test_loader, criterion, device)
            add_logs(tb_log, "ssl/test", test_log, global_step=i)
            noisy_test_log = validate(
                config, model, noisy_test_loader, criterion, device, mode="noisy"
            )
            add_logs(tb_log, "ssl/noisy_test", noisy_test_log, global_step=i)
            ema.restore()

        _, images, _, l_labels, n_labels = clean_prefetcher.next()
        if images != None:
            _, (x_ulb_w, x_ulb_s), _, _, n_ulb = noisy_prefetcher.next()

    ema.apply_shadow()
    return log


@torch.no_grad()
def validate(config, model, test_loader, criterion, device, mode="normal"):

    model.eval()

    log = {
        "batch_time": metric.AverageMeter("Batch Time", ":6.3f"),
        "ce_loss": metric.AverageMeter("Cross Entropy", ":6.3f"),
        "top_1": metric.AverageMeter("Top-1 Acc", ":6.2f"),
        "top_5": metric.AverageMeter("Top-5 Acc", ":6.2f"),
    }

    total_iters = len(test_loader)
    progress = metric.ProgressMeter(
        total_iters,
        log["batch_time"],
        log["ce_loss"],
        log["top_1"],
        log["top_5"],
        prefix="Test: ",
    )
    end = time.time()

    for i, data in enumerate(test_loader):

        if mode == "normal":
            images, target = data
            n_labels = utils.null_labels(
                (images.shape[0], config.data.num_classes), config.null_label_type
            )
        else:
            _, images, target, _, n_labels = data

        images = images.to(device, non_blocking=True)
        n_labels = n_labels.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if config.is_amp:
            with amp.autocast():
                output, ce_loss = run_model_normal(
                    model, images, n_labels, target, criterion, mixup=False
                )
        else:
            model, ce_loss = run_model_normal(
                model, images, n_labels, target, criterion, mixup=False
            )

        batch_size_now = images.size(0)
        avg_acc1, avg_acc5 = metric.accuracy(output, target, topk=(1, 5))
        log["top_1"].update(avg_acc1[0].item(), batch_size_now)
        log["top_5"].update(avg_acc5[0].item(), batch_size_now)
        log["ce_loss"].update(ce_loss.mean().item(), batch_size_now)

        log["batch_time"].update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.print(i)

    logger.info(
        f"[TEST: {mode.capitalize()}] "
        + "  |  ".join([str(meter) for meter in log.values()])
    )

    model.train()

    return log




@torch.no_grad()
def validate_imagenet(config, model, criterion, device, mode="normal"):
    import datasets.ilsvrc2012.ilsvrc2012 as imagenet
    test_dataset = imagenet.imagenet_test_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        config.bootstrapping.batch_size,
        num_workers=config.bootstrapping.num_workers,
        shuffle=False,
        drop_last=False,
    )
    log = validate(config, model, test_loader, criterion, device, mode)
    return log

@torch.no_grad()
def validate_asym(config, model, criterion, device, mode="normal"):

    from utils.pseudo_labelling import get_logits_labels_eval, get_predictions
    import datasets.cifar10.cifar10 as cifar10

    test_dataset = cifar10.cifar10_test_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        config.bootstrapping.batch_size,
        num_workers=config.bootstrapping.num_workers,
        shuffle=False,
        drop_last=False,
    )

    logits, labels = get_logits_labels_eval(
        model,
        test_loader,
        len(test_dataset),
        config.data.num_classes,
        device,
        1,
    )

    predictions, confidences, uncertainties = get_predictions(logits)

    correct = predictions == labels
    acc = np.sum(correct) / correct.shape[0]
    logger.info(f"Acc Without Noisy Labels: {acc}")
    avg_acc1, avg_acc5 = metric.accuracy(torch.tensor(np.mean(logits, axis=1)), torch.tensor(labels), topk=(1, 5))
    logger.info(f"\tTop 1: {avg_acc1}, Top 5: {avg_acc5}")



    logits, labels = get_logits_labels_eval(
        model,
        test_loader,
        len(test_dataset),
        config.data.num_classes,
        device,
        1,
        generate_noisy_labels=True
    )

    predictions, confidences, uncertainties = get_predictions(logits)

    correct = predictions == labels
    acc = np.sum(correct) / correct.shape[0]
    logger.info(f"Acc With Noisy Labels: {acc}")
    avg_acc1, avg_acc5 = metric.accuracy(torch.tensor(np.mean(logits, axis=1)), torch.tensor(labels), topk=(1, 5))
    logger.info(f"\tTop 1: {avg_acc1}, Top 5: {avg_acc5}")

    import utils.plotter as plotter
    plotter.label_variation_test(config, model, test_loader, device, os.path.join(config.save_dir, 'label_variation'))


# ------------
# --- Main ---
# ------------


def add_logs(tb_log, topic_parent, log, global_step):

    for key, value in log.items():
        tb_log.add_scalar(f"{topic_parent}/{key}", value.avg, global_step=global_step)


def bootstrapping(config, ds, model, device, tb_log):
    """Run the algorithm to find the clean set of samples"""

    # Define loss function
    if config.is_label_smoothing:
        criterion = label_smoothing.label_smoothing_CE(reduction="mean")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Get the dataloaders
    train_dataset, test_dataset = get_bootstrapping_datasets(
        config.data.dataset, ds, config.bootstrapping.train_aug
    )
    train_loader, test_loader = get_bootstrapping_loaders(
        config, train_dataset, test_dataset
    )

    # Optimizer
    param_groups = lr_scheduler.get_parameter_groups(model)
    optimizer = torch.optim.SGD(
        param_groups,
        config.bootstrapping.lr,
        momentum=config.momentum,
        weight_decay=config.bootstrapping.weight_decay,
        nesterov=config.is_nesterov,
    )

    if config.bootstrapping.lr_schedule_mode == "multistep":

        scheduler = lr_scheduler.lr_scheduler_3(
            init_lr=config.bootstrapping.lr,
            milestones=config.bootstrapping.lr_milestones,
            gamma=config.bootstrapping.lr_gamma,
        )

    else:

        scheduler = lr_scheduler.lr_scheduler(
            init_lr=config.bootstrapping.lr,
            num_epochs=config.bootstrapping.epochs,
            iters_per_epoch=len(train_loader),
            slow_start_epochs=config.bootstrapping.slow_start_epochs,
            slow_start_lr=config.bootstrapping.slow_start_lr,
            end_lr=config.bootstrapping.end_lr,
        )

    scaler = None if not config.is_amp else amp.GradScaler()

    if config.data.dataset == "webvision":
        print("SSL Round Imagenet Test")
        test_log = validate_imagenet(config, model, criterion, device)
        add_logs(tb_log, "ssl/test_ilsvrc2012", test_log, global_step=0)

    # Train the model, and get the predicted logits
    for epoch in range(1, config.bootstrapping.epochs + 1):

        logger.info(f"Train Epoch: {epoch}")

        train_log = bootstrapping_epoch(
            config,
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
        )
        add_logs(tb_log, "bootstrapping/train", train_log, global_step=epoch)

        if epoch % config.bootstrapping.test_frequency == 0:

            logger.info(f"Test Epoch: {epoch}")
            test_log = validate(config, model, test_loader, criterion, device)

            add_logs(tb_log, "bootstrapping/test", test_log, global_step=epoch)
    
    if config.data.dataset == "webvision":
        logger.info(f"Bootstrapping Imagenet Test")
        test_log = validate_imagenet(config, model, criterion, device)
        add_logs(tb_log, "bootstrapping/test_ilsvrc2012", test_log, global_step=epoch)

    return model


def ssl_round(config, round_no, clean_ds, noisy_ds, model, device, tb_log):

    # Define loss function
    if config.is_label_smoothing:
        criterion = label_smoothing.label_smoothing_CE(reduction="mean")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    param_groups = lr_scheduler.get_parameter_groups(model)
    optimizer = torch.optim.SGD(
        param_groups,
        config.ssl.lr,
        momentum=config.momentum,
        weight_decay=config.ssl.weight_decay,
        nesterov=config.ssl.is_nesterov,
    )

    scheduler = lr_scheduler.lr_scheduler_2(
        init_lr=config.ssl.lr,
        total_iters=config.ssl.iterations_per_round,
        slow_start_iters=config.ssl.slow_start_iters,
        slow_start_lr=config.ssl.slow_start_lr,
        end_lr=config.ssl.end_lr,
    )

    scaler = None if not config.is_amp else amp.GradScaler()

    # Get the dataloaders
    clean_dataset, noisy_dataset, noisy_test_dataset, test_dataset = get_ssl_datasets(
        config.data.dataset,
        clean_ds,
        noisy_ds,
    )
    clean_loader, noisy_loader, noisy_test_loader, test_loader = get_ssl_round_loaders(
        config, clean_dataset, noisy_dataset, noisy_test_dataset, test_dataset
    )

    train_log = ssl_round_train(
        config,
        round_no,
        clean_loader,
        noisy_loader,
        noisy_test_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
        tb_log,
    )
    add_logs(
        tb_log,
        "ssl/train",
        train_log,
        global_step=(round_no + 1) * config.ssl.iterations_per_round,
    )

    noisy_eval_dataset = get_split_dataset_datasets(
        config.data.dataset, noisy_ds, config.bootstrapping.eval_aug
    )
    noisy_eval_loader = get_split_dataset_loader(config, noisy_eval_dataset)
    output_ds = create_final_dataset(
        config, clean_ds, noisy_ds, model, noisy_eval_loader, device
    )

    if config.data.dataset == "webvision":
        print("SSL Round Imagenet Test")
        test_log = validate_imagenet(config, model, criterion, device)
        add_logs(tb_log, "ssl/test_ilsvrc2012", test_log, global_step=0)

    return output_ds, model



def ssl_round_test(config, clean_ds, noisy_ds, model, device, tb_log):

    # NOTE: Augmentations and training schedule have also changed
    config.ssl.uratio = 1
    config.ssl.ulb_loss_ratio = 0.0
    config.ssl.is_mixup = True

    # Define loss function
    if config.is_label_smoothing:
        criterion = label_smoothing.label_smoothing_CE(reduction="mean")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    param_groups = lr_scheduler.get_parameter_groups(model)
    optimizer = torch.optim.SGD(
        param_groups,
        config.final_model.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.is_nesterov,
    )

    scheduler = lr_scheduler.lr_scheduler_2(
        init_lr=config.final_model.lr,
        total_iters=config.final_model.epochs * len(clean_ds.indices) // config.final_model.batch_size,
        slow_start_iters=0,
        slow_start_lr=config.final_model.slow_start_lr,
        end_lr=config.final_model.end_lr,
    )

    scaler = None if not config.is_amp else amp.GradScaler()

    # Get the dataloaders (just used for placeholding)
    _, noisy_dataset, noisy_test_dataset, _ = get_ssl_datasets(
        config.data.dataset,
        clean_ds,
        noisy_ds,
    )
    train_dataset, test_dataset = get_final_datasets(config.data.dataset, clean_ds)
    train_loader, noisy_loader, noisy_test_loader, test_loader = get_ssl_round_test_loaders(
        config, config.final_model.epochs * len(clean_ds.indices) // config.final_model.batch_size, train_dataset, noisy_dataset, noisy_test_dataset, test_dataset
    )

    train_log = ssl_round_train(
        config,
        0,
        train_loader,
        noisy_loader,
        noisy_test_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
        tb_log,
    )
    add_logs(
        tb_log,
        "ssl/train",
        train_log,
        global_step=(0 + 1) * config.ssl.iterations_per_round,
    )

    noisy_eval_dataset = get_split_dataset_datasets(
        config.data.dataset, noisy_ds, config.bootstrapping.eval_aug
    )
    noisy_eval_loader = get_split_dataset_loader(config, noisy_eval_dataset)
    output_ds = create_final_dataset(
        config, clean_ds, noisy_ds, model, noisy_eval_loader, device
    )

    return output_ds, model


def final_model(config, ds, model, device, tb_log):
    """Run the algorithm to find the clean set of samples"""

    if config.is_label_smoothing:
        criterion = label_smoothing.label_smoothing_CE(reduction="mean")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Get the dataloaders
    train_dataset, test_dataset = get_final_datasets(config.data.dataset, ds)
    train_loader, test_loader = get_final_loaders(config, train_dataset, test_dataset)

    # Optimizer
    param_groups = lr_scheduler.get_parameter_groups(model)
    optimizer = torch.optim.SGD(
        param_groups,
        config.final_model.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.is_nesterov,
    )

    scheduler = lr_scheduler.lr_scheduler(
        init_lr=config.final_model.lr,
        num_epochs=config.final_model.epochs,
        iters_per_epoch=len(train_loader),
        slow_start_epochs=config.final_model.slow_start_epochs,
        slow_start_lr=config.final_model.slow_start_lr,
        end_lr=config.final_model.end_lr,
    )

    scaler = None if not config.is_amp else amp.GradScaler()

    # Train the model, and get the predicted logits
    for epoch in range(1, config.final_model.epochs + 1):

        logger.info(f"Train Epoch: {epoch}")

        train_log = final_model_epoch(
            config,
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
        )
        add_logs(tb_log, "final_model/train", train_log, global_step=epoch)

        if epoch % config.final_model.test_frequency == 0:

            logger.info(f"Test Epoch: {epoch}")
            test_log = validate(config, model, test_loader, criterion, device)

            add_logs(tb_log, "final_model/test", test_log, global_step=epoch)

    if config.data.dataset == "webvision":
        logger.info(f"Final Model Imagenet Test")
        test_log = validate_imagenet(config, model, criterion, device)
        add_logs(tb_log, "final_model/test_ilsvrc2012", test_log, global_step=0)

    if config.data.dataset == "cifar10" and config.data.noise_type == "asym-0.4":
        logger.info(f"Final Model Asym 0.4 Test")
        test_log = validate_asym(config, model, criterion, device)

    return model


def split_dataset(config, ds, model, device, tb_log):

    eval_dataset = get_split_dataset_datasets(
        config.data.dataset, ds, config.bootstrapping.eval_aug
    )
    eval_loader = get_split_dataset_loader(config, eval_dataset)

    clean_ds, noisy_ds = select_samples(config, model, ds, eval_loader, device)

    return clean_ds, noisy_ds


def create_final_dataset(config, clean_ds, noisy_ds, model, noisy_weak_loader, device):

    import copy
    from utils.pseudo_labelling import get_logits, get_predictions

    logits = get_logits(
        model,
        noisy_weak_loader,
        noisy_ds.indices.shape[0],
        config.data.num_classes,
        device,
        config.ds_partition.guessing_label_iterations,
    )

    # Make our selection based on the predictions
    predictions, confidences, uncertainties = get_predictions(logits)

    output_ds = copy.deepcopy(clean_ds)
    output_ds.learned_labels[noisy_ds.indices] = predictions
    output_ds.indices = np.union1d(output_ds.indices, noisy_ds.indices)

    output_ds.noise_distribution = None

    return output_ds


def eval_with_aug(config, model, dataset, device):

    from utils.pseudo_labelling import get_logits_labels_eval, get_predictions

    test_aug_dataset = get_test_aug_datasets(dataset)
    test_aug_loader = get_test_aug_loader(config, test_aug_dataset)

    print(len(test_aug_dataset))
    logits, labels = get_logits_labels_eval(
        model,
        test_aug_loader,
        len(test_aug_dataset),
        config.data.num_classes,
        device,
        config.ds_partition.guessing_label_iterations,
    )

    predictions, confidences, uncertainties = get_predictions(logits)

    correct = predictions == labels
    acc = np.sum(correct) / correct.shape[0]
    logger.info(f"Eval With Aug with Dataset {dataset}: {acc}")
    avg_acc1, avg_acc5 = metric.accuracy(torch.tensor(np.mean(logits, axis=1)), torch.tensor(labels), topk=(1, 5))
    logger.info(f"\tTop 1: {avg_acc1}, Top 5: {avg_acc5}")


def eval_with_aug_and_nl_prediction(config, model_a, model_b, device):

    from utils.pseudo_labelling import get_logits_labels_combined, get_predictions

    test_aug_dataset = get_test_aug_datasets(config.data.dataset)
    test_aug_loader = get_test_aug_loader(config, test_aug_dataset)

    logits, labels = get_logits_labels_combined(
        model_a,
        model_b,
        test_aug_loader,
        len(test_aug_dataset),
        config.data.num_classes,
        device,
        config.ds_partition.guessing_label_iterations,
    )

    predictions, confidences, uncertainties = get_predictions(logits)

    correct = predictions == labels
    acc = np.sum(correct) / correct.shape[0]
    logger.info(f"Eval With Aug and NL Prediction: {acc}")


def lerp(a, b, t):
    return a + (b-a) * t

def main(config):

    # Set the process name and start the tensorboard file
    setproctitle.setproctitle("BS - New Pipeline")
    tb_log = SummaryWriter(log_dir=os.path.join(config.save_dir, "tensorboard"))

    # Get the device and set the seed
    device = utils.set_device(config)
    if config.seed is not None:
        utils.set_seed(config.seed)

    # Load the datasets to learn on
    ds = load_dataset(config.data.dataset, config.data.noise_type)
    ds.learned_labels = np.argmax(ds.noisy_label_sets[0], axis=1)
    config.data.total_samples = ds.noisy_label_sets[0].shape[0]
    config.data.num_classes = ds.noisy_label_sets[0].shape[1]
    log_dataset_info.log_dataset(ds, os.path.join(config.save_dir, "starting"))

    # Create the model
    model = create_model(
        config,
        semantic_type=config.model.semantic_type,
        num_classes=config.data.num_classes,
        label_size=config.data.num_classes,
        load_pretrained=config.model.load_pretrained,
    )
    model = model.to(device)

    # Run bootstrapping training
    model = bootstrapping(config, ds, model, device, tb_log)
    bootstrapping_model = copy.deepcopy(model)
    eval_with_aug(config, model, config.data.dataset, device)
    if config.data.dataset == "webvision":
        eval_with_aug(config, model, "ilsvrc2012", device)
    eval_with_aug_and_nl_prediction(config, bootstrapping_model, model, device)
    eval_with_aug_and_nl_prediction(config, model, model, device)

    # Find the initial clean and noisy sets (using null labels rather than noisy ones)
    clean_ds, noisy_ds = split_dataset_2(
        config,
        ds,
        model,
        config.ds_partition.forced_frac,
        config.ds_partition.confidence_threshold,
        device,
        tb_log,
        os.path.join(config.save_dir, "dataset_split", "bootstrapping"),
        use_noisy_labels=False,
    )

    for round_no in range(config.ssl.training_rounds):
        logger.info(f"Training Round {round_no}")
        final_ds, model = ssl_round(
            config, round_no, clean_ds, noisy_ds, model, device, tb_log
        )
        eval_with_aug(config, model, config.data.dataset, device)
        if config.data.dataset == "webvision":
            eval_with_aug(config, model, "ilsvrc2012", device)
        eval_with_aug_and_nl_prediction(config, bootstrapping_model, model, device)
        eval_with_aug_and_nl_prediction(config, model, model, device)

    model = final_model(config, final_ds, model, device, tb_log)
    eval_with_aug(config, model, config.data.dataset, device)
    if config.data.dataset == "webvision":
        eval_with_aug(config, model, "ilsvrc2012", device)
    eval_with_aug_and_nl_prediction(config, bootstrapping_model, model, device)
    eval_with_aug_and_nl_prediction(config, model, model, device)

    tb_log.flush()
    tb_log.close()

    torch.cuda.empty_cache()


if __name__ == "__main__":

    # Load the config and create the results folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_name = sys.argv[1]
    config = workspace.create_workspace(current_dir, config_file_name)

    # Run the experiment
    main(config)
import os
import shutil
import time
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.multiprocessing
import apex
from apex.parallel.LARC import LARC
import torch.distributed as dist
import helper.train_utils as tut
from helper.my_enums import ChannelModal,EvalType,ModalMergeAlgo
import my_ds.ds_server as mydss

# torch.multiprocessing.set_sharing_strategy('file_system')
from helper.mutils import   get_model
from helper.opt_aug import parse_arguments
import helper.mutils as mut
from helper.io_utilts import prep_logfiles

from helper.utils import (
    initialize_exp,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    init_signal_handler,
    # get_loss,

)
logger = getLogger()

# global variables
sk_schedule = None
group = None
global sk_counter
sk_counter = 0

def fw_pass(optimizer,model,inputs,queue,use_the_queue,epoch):
    loss, use_the_queue, bs = tut.compute_ot_loss(args, dist, model, inputs, queue, use_the_queue, epoch)
    loss /= len(args.temporal_for_assign)

    if args.use_protreg == True:
        if args.distributed == True:
            ploss, _ = prototype_loss(model.module.prototypes.weight)
        else:
            ploss, _ = prototype_loss(model.prototypes.weight)
        if args.use_precomp_prot != True:
            loss = ploss + loss
        ploss_item=ploss.item()
    else:
        ploss_item=0
    optimizer.zero_grad()
    loss.backward()

    if epoch < args.freeze_prototypes_epoch or args.use_precomp_prot == True:
        for name, p in model.named_parameters():
            if "prototypes" in name:
                p.grad = None

    optimizer.step()
    return loss.item(),ploss_item,bs

def train_one_epoch(train_loader, model, optimizer, epoch, lr_schedule, queue):
    batch_time = AverageMeter(args.distributed)
    data_time = AverageMeter(args.distributed)
    losses = AverageMeter(args.distributed)
    plosses = AverageMeter(args.distributed)
    model.train()
    use_the_queue = False
    end = time.time()
    for it, inputs in enumerate(train_loader):

        ctime = time.time()
        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            if args.distributed == False:
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.prototypes.weight.copy_(w)
            else:
                w = model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.module.prototypes.weight.copy_(w)

        data_time.update(ctime - end)
        loss_item,ploss_item,bs=fw_pass(optimizer, model, inputs, queue, use_the_queue, epoch)
        losses.update(loss_item, bs)
        if args.use_protreg == True:
            plosses.update(ploss_item, bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if (it % args.per_iter_show  == 0 or it %(len(train_loader)-1)==0) and (it>0 or (epoch==0 and it ==0)):
            data_time.synchronize_between_processes()
            batch_time.synchronize_between_processes()
            losses.synchronize_between_processes()
            if args.use_protreg == True:
                plosses.synchronize_between_processes()

        if args.rank == 0 and (it % args.per_iter_show  == 0 or it %(len(train_loader)-1)==0) and (it>0 or (epoch==0 and it ==0)):
            logger.info(
                "Ep [{0}]: [{1}][{2}]\t"
                "Tm {batch_time.val:.3f} ({batch_time.sync_avg:.3f})\t"
                "Dt {data_time.val:.3f} ({data_time.sync_avg:.3f})\t"
                "Lss {loss.val:.4f} ({loss.sync_avg:.4f})\t"
                "Lss {plosses.val:.4f} ({plosses.sync_avg:.4f})\t"
                "Lr: {lr:.4f}".format(epoch,
                    it,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses, plosses=plosses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.sync_avg), queue

def prototype_loss(prototypes):
    product = torch.matmul(prototypes, prototypes.t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()

def main():

    global args
    args = parse_arguments()
    args.evaluation_type=EvalType.PreTrain.value
    args.device_count=torch.cuda.device_count()

    init_distributed_mode(args)
    init_signal_handler(args.rank)
    fix_random_seeds(args.seed)
    args = prep_logfiles(args,logmode='train')
    # Load model
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    if args.rank==0:
        logger.info("Log Path: {0}".format( args.log_path))

    model = get_model(logger,args)
    return_lst=mydss.get_dataset_withmodality(args,modality=['train'],eval_mode=False)
    train_loader=return_lst[0]
    args.nclass = len(train_loader.dataset.class_to_idx)
    if args.rank == 0:
        logger.info('Data Loading Completed: nclass: {0}, nsamples: {1}'.format(args.nclass,len(train_loader)*args.world_size*args.batch_size))

    ds_szie = len(train_loader)

    model, optimizer, lr_scheduler = tut.setup_optimization_finetune(args, model, logger, ds_szie)

    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        que_oth=torch.load(queue_path)
        queue = que_oth["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    start_epoch=0
    if args.resume_training==True:
        #(args,ckpt_path,model,optimizer,amp):
        start_epoch=mut.resume_training_model(args,
            ckpt_path=args.resume_model_path,
            model=model,
            optimizer=optimizer
        )

    for epoch in range(start_epoch, args.curr_final_epochs):
        if args.rank == 0:
            logger.info("============ Starting epoch %i ... ============" % epoch)
        if args.distributed == True:
            train_loader.sampler.set_epoch(epoch)

        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            if args.video_channels == ChannelModal.rgb_flow.value:
                if args.video_modal_merge_algo == ModalMergeAlgo.PriorRgbFlow.value:
                    queue = torch.zeros(
                        2,  len(args.temporal_for_assign),
                        args.queue_length // args.world_size,
                        args.emedding_dim,
                    ).cuda()
                elif args.video_modal_merge_algo == ModalMergeAlgo.MixRgbFlow.value:
                    queue = torch.zeros(
                        len(args.temporal_for_assign)*2,
                        args.queue_length // args.world_size,
                        args.emedding_dim,
                    ).cuda()
            else:
                queue = torch.zeros(
                    len(args.temporal_for_assign),
                    args.queue_length // args.world_size,
                    args.emedding_dim,
                ).cuda()

        scores, queue = train_one_epoch(train_loader, model, optimizer, epoch, lr_scheduler, queue)
        if args.rank == 0:
            training_stats.update(scores)
        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_checkpoints, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_checkpoints, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)


if __name__ == "__main__":
    main()

import torch.nn.functional as F
import numpy as np
import math,copy
from helper.my_enums import ChannelModal,EvalType,ModalMergeAlgo
from helper.scheduler import WarmupMultiStepLR
import torch
import torch.nn as nn
import apex
from apex.parallel.LARC import LARC

def get_scheduled_LR(ds_szie,args):

    base_lr = args.base_lr * math.sqrt(args.batch_size * args.world_size)
    warmup_lr_schedule = np.linspace(args.start_warmup, base_lr, ds_szie * args.warmup_epochs)

    iters = np.arange(ds_szie * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (base_lr- args.final_lr) * (1 + \
                                                                                           math.cos(math.pi * t / (ds_szie* (args.epochs - args.warmup_epochs))))
                                   for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    if args.fix_lr==True:
        lr_schedule[:]=args.final_lr
    return lr_schedule


def get_wicc_val(ds_szie,args):
    base_lr = args.base_lr
    lr_schedule=[]
    for e in range(args.epochs):
        curr_lr =base_lr
        for milestone in args.lr_milestones:
            curr_lr *= 0.1 if e >= milestone else 1.
        lr_schedule.extend([curr_lr]*ds_szie)
    lr_scheduler = np.asarray(lr_schedule)
    return lr_scheduler


def get_original_scheluder(ds_szie,args,optimizer):
    warmup_iters = args.lr_warmup_epochs * ds_szie
    lr_milestones = [ds_szie * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)
    return lr_scheduler


def wrap_ddp(args,model):
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    return model



def setup_optimization_finetune(args,model,logger,ds_szie):
    if args.evaluation_type == EvalType.LinCls.value:  # linear classifier training
        nfrozen=0
        for name, param in model.named_parameters():
            if 'video_network' in name or  'video_network_flow' in name:
                param.requires_grad = False
                nfrozen+=1
        print('We are frozing {0} parameters'.format(nfrozen))

        if args.distributed == True:
            if args.sync_bn == "pytorch":
                model.linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(model.linear_classifier)
                # print('convert_sync_batchnorm======================',model)
            elif args.sync_bn == "apex":
                process_group = None
                if args.world_size // 8 > 0:
                    process_group = apex.parallel.create_syncbn_process_group(args.world_size // 8)
                model.linear_classifier = apex.parallel.convert_syncbn_model(model.linear_classifier, process_group=process_group)
        model=model.cuda()
        selected_params = model.linear_classifier.parameters()
    elif args.evaluation_type == EvalType.FT.value:  # Fine Tuning or pretraining
        if args.distributed == True:
            if args.sync_bn == "pytorch":
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            elif args.sync_bn == "apex":
                process_group = None
                if args.world_size // 8 > 0:
                    process_group = apex.parallel.create_syncbn_process_group(args.world_size // 8)
                model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        model = model.cuda()
        if args.use_vicc_opt == True:
            selected_params = model.named_parameters()
        else:
            selected_params = model.parameters()
        print('Print FT or Pre-Trainining....')
    elif args.evaluation_type ==EvalType.PreTrain.value :  # Fine Tuning or pretraining
        if args.distributed == True:
            if args.sync_bn == "pytorch":
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            elif args.sync_bn == "apex":
                process_group = None
                if args.world_size // 8 > 0:
                    process_group = apex.parallel.create_syncbn_process_group(args.world_size // 8)
                model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        model = model.cuda()
        selected_params = model.parameters()
        print('Print FT or Pre-Trainining....')

    if args.use_vicc_opt== True:
        if args.evaluation_type ==EvalType.FT.value:
            tmp_selected_params = []
            for name, param in selected_params:
                if 'linear_classifier' not in name:
                    tmp_selected_params.append({'params': param, 'lr': args.base_lr / args.backbone_ratio})
                else:
                    tmp_selected_params.append({'params': param})
            selected_params=tmp_selected_params

        if args.use_adam==True:
            optimizer = torch.optim.Adam(
                selected_params,
                lr=args.base_lr,
                weight_decay=args.weight_decay,
            )
            print('Using Adam optimzer...')
        else:
            optimizer = torch.optim.SGD(
                selected_params,
                lr=args.base_lr,
                nesterov=False,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        lr_scheduler = get_wicc_val(ds_szie, args)
    else:
        lr_scheduler = get_scheduled_LR(ds_szie, args)
        optimizer = torch.optim.SGD(
            selected_params,
            lr=args.base_lr,
            nesterov=False,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

        if args.use_LARC == True:
            optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler[0]


            # init mixed precision
    if args.rank == 0:
        logger.info("Building optimizer done: Eval Type: {0}".format(args.evaluation_type))

    if args.distributed==True:
        model=wrap_ddp(args,model)

    return model,optimizer,lr_scheduler

def merge_rgb_flow(bs, embedding, output, start_idx,args):
    new_nmb_temporal_samples = copy.deepcopy(args.nmb_temporal_samples)
    new_temporal_for_assign = list(range(len(args.temporal_for_assign) * 2))
    output, output_flow = output
    embedding, embedding_flow = embedding
    for i, tempo_id in enumerate(args.temporal_for_assign):
        end_idx=start_idx + args.nmb_temporal_samples[tempo_id] * bs
        out_cut = output[start_idx:end_idx]
        out_flow_cut = output_flow[start_idx:end_idx]
        emb_cut = embedding[start_idx:end_idx]
        emb_flow_cut = embedding_flow[start_idx:end_idx]
        if start_idx == 0:
            concat_output = torch.cat((out_cut, out_flow_cut))
            concat_embedding = torch.cat((emb_cut, emb_flow_cut))
        else:
            concat_output = torch.cat((concat_output, out_cut))
            concat_output = torch.cat((concat_output, out_flow_cut))
            concat_embedding = torch.cat((concat_embedding, emb_cut))
            concat_embedding = torch.cat((concat_embedding, emb_flow_cut))
        start_idx = end_idx
        new_nmb_temporal_samples.append(args.nmb_temporal_samples[tempo_id])
        # print('done...', concat_output.shape, out_cut.shape, out_flow_cut.shape)
    # print('after_zero_mean: ',concat_output.mean(-1))
    return concat_output,concat_embedding, new_temporal_for_assign


@torch.no_grad()
def distributed_sinkhorn(dist,args,out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if args.distributed==True:
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if args.distributed == True:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


@torch.no_grad()
def distributed_sinkhorn_withprior(dist,args,out,prior_P,type_of_modal=''):
    B = out.t().shape[1] * args.world_size  # number of samples to assign
    K = out.t().shape[0]  # how many prototypes
    if prior_P!=None:
        denominator = (args.epsilon) + (args.epsilon2)
        prior_withreg = -torch.log(prior_P/B) * (args.epsilon2)
        out_tmp = (out + prior_withreg) / denominator
    else:
        out_tmp=out / args.epsilon
    Q = torch.exp(out_tmp).t()  # Q is K-by-B for consistency with notations from our paper

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if args.distributed==True:
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if args.distributed == True:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()
import time
def compute_ot_loss(args,dist,model,inputs, queue, use_the_queue,epoch):
    video, _, _, _ = inputs
    if args.video_channels == ChannelModal.rgb_flow.value:
        bs = video[0][0].size(0)
    else:
        bs = video[0].size(0)
    embedding, output = model(video)
    if args.video_channels ==ChannelModal.rgb_flow.value:
        start_idx = 0
        total_samples = np.sum(args.nmb_temporal_samples)
        new_temporal_for_assign = copy.deepcopy(args.temporal_for_assign)
        if args.video_modal_merge_algo ==ModalMergeAlgo.MixRgbFlow.value:
            output, embedding, new_temporal_for_assign = merge_rgb_flow(bs, embedding, output, start_idx, args)
            total_samples = total_samples * 2
    else:
        new_temporal_for_assign = args.temporal_for_assign
        total_samples = np.sum(args.nmb_temporal_samples)

    # print('output:',torch.mean(output))
    loss = 0
    for i, tempo_id in enumerate(new_temporal_for_assign):
        with torch.no_grad():
            if args.video_modal_merge_algo == ModalMergeAlgo.PriorRgbFlow.value and args.video_channels == ChannelModal.rgb_flow.value:

                out = output[0][bs * tempo_id: bs * (tempo_id + 1)].detach()
                out_flow = output[1][bs * tempo_id: bs * (tempo_id + 1)].detach()

                if queue is not None:
                    if use_the_queue or not torch.all(queue[0, i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[0, i],
                            model.module.prototypes.weight.t()
                        ), out))

                        out_flow = torch.cat((torch.mm(
                            queue[1, i],
                            model.module.prototypes.weight.t()
                        ), out_flow))

                    # fill the queue
                    queue[0, i, bs:] = queue[0, i, :-bs].clone()
                    queue[0, i, :bs] = embedding[0][tempo_id * bs: (tempo_id + 1) * bs]
                    queue[1, i, bs:] = queue[1, i, :-bs].clone()
                    queue[1, i, :bs] = embedding[1][tempo_id * bs: (tempo_id + 1) * bs]
                # print('Running....', epoch<args.video_modal_merge_start, epoch,args.video_modal_merge_start)
                if epoch<args.video_modal_merge_start:
                    q_flow = distributed_sinkhorn_withprior(dist, args, out_flow, prior_P=None)[-bs:]
                    q = distributed_sinkhorn_withprior(dist, args, out, prior_P=None)[-bs:]
                else:

                    q_flow_tmp = distributed_sinkhorn_withprior(dist, args, out_flow, prior_P=None)
                    q_rgb_tmp = distributed_sinkhorn_withprior(dist, args, out, prior_P=None)
                    if args.sinkhorn_cycle == 1:
                        q = distributed_sinkhorn_withprior(dist, args, out, prior_P=q_flow_tmp, type_of_modal='rgb')[-bs:]
                        q_flow = distributed_sinkhorn_withprior(dist, args, out_flow, prior_P=q_rgb_tmp,type_of_modal='flow')[-bs:]
                    elif args.sinkhorn_cycle ==2:
                        q_rgb_tmp_2 = distributed_sinkhorn_withprior(dist, args, out, prior_P=q_flow_tmp,type_of_modal='rgb')
                        q_flow_tmp_2 = distributed_sinkhorn_withprior(dist, args, out_flow, prior_P=q_rgb_tmp,type_of_modal='flow')
                        q = distributed_sinkhorn_withprior(dist, args, out, prior_P=q_flow_tmp_2,type_of_modal='rgb')[-bs:]
                        q_flow = distributed_sinkhorn_withprior(dist, args, out_flow, prior_P=q_rgb_tmp_2,type_of_modal='flow')[-bs:]
                    else:
                        raise ValueError('A very specific bad thing happened. It is about sinkhorn_cycle')
            else:
                out = output[bs * tempo_id: bs * (tempo_id + 1)].detach()
                # print('out_mean" ',out.mean(),tempo_id,bs)
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[tempo_id * bs: (tempo_id + 1) * bs]

                q = distributed_sinkhorn_withprior(dist, args, out,None)
                q = q[-bs:]
        if args.video_modal_merge_algo == ModalMergeAlgo.PriorRgbFlow.value and args.video_channels == ChannelModal.rgb_flow.value:
            subloss_rgb = compute_ce(args, bs, total_samples, output[0], q, tempo_id) / (total_samples - 1)
            subloss_flow = compute_ce(args, bs, total_samples, output[1], q_flow, tempo_id) / (total_samples - 1)
            subloss = (subloss_rgb + subloss_flow) / 2
            # print(subloss_rgb,subloss_flow)
        else:
            subloss = compute_ce(args,bs, total_samples, output, q, tempo_id) / (total_samples - 1)

        loss += subloss
    loss /= len(new_temporal_for_assign)
    return loss,use_the_queue,bs

def compute_ce(args,bs,  total_samples, output, q, tempo_id):
    subloss = 0
    for v in np.delete(np.arange(total_samples), tempo_id):
        x = output[bs * v: bs * (v + 1)] / args.temperature
        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
    return subloss
import os
import numpy as np
import torch
from helper.my_enums import ChannelModal,EvalType
from my_models.model_aug import AVModel
import copy
#
#


def load_checkpint(args,ckpt_path):
    if os.path.exists(ckpt_path):
        ckpt_dict = torch.load(ckpt_path, map_location="cuda:" + str(args.gpu_to_work_on))
        return ckpt_dict
    else:
        print('File not exist: {0}'.format(ckpt_path))


def load_model_parameters(args,ckpt_channel,curr_model, loaded_state_dic):
    curr_self_state = curr_model.state_dict()
    cnt_params_set=0
    # print(curr_self_state.keys())
    for name in loaded_state_dic:
        param = loaded_state_dic[name]
        if 'module.' in name:
            name = name.replace('module.', '')
        if 'base.' in name:
            name = name.replace('base.', '')

        if name in curr_self_state.keys():
            curr_self_state[name].copy_(param)
            cnt_params_set+=1
        else:
            if args.rank==0:
                print("didnt load ", name)
    if args.rank == 0:
        print('Total Parameters Loaded and Set: {0}, Total params on Model {1}'.format(cnt_params_set,len(curr_self_state.keys())))
        assert cnt_params_set>0



def get_model(logger,args):
    model = AVModel(
        args=args
    )
    if args.rank == 0:
        logger.info(model)
        logger.info('Model Building {0}'.format(model.count_params()))

    if args.use_precomp_prot==True and args.evaluation_type==EvalType.PreTrain.value:
        with torch.no_grad():
            w=torch.as_tensor(np.load(args.use_precomp_prot_path)).cuda()
            model.prototypes.weight.copy_(w)
            if args.rank == 0:
                logger.info('Pre-computed prototypes loaded: {0}'.format(args.use_precomp_prot_path))



    return model


def load_model_parameters_resume(args,curr_model, loaded_state_dic):
    curr_self_state = curr_model.state_dict()
    cnt_params_set=0
    # print(curr_self_state.keys())
    for name in loaded_state_dic:
        param = loaded_state_dic[name]
        if name in curr_self_state.keys():
            curr_self_state[name].copy_(param)
            cnt_params_set+=1
        else:
            if args.rank==0:
                print("didnt load ", name)
    if args.rank == 0:
        print('Total Paramset Set: {0}, Total params on Model {1}'.format(cnt_params_set,len(curr_self_state.keys())))



def resume_training_model(args,ckpt_path,model,optimizer):
    if args.rank == 0:
        print('Resuming From:', ckpt_path)
    assert os.path.exists(ckpt_path)
    ckpt_dic = load_checkpint(args,ckpt_path)
    loaded_state_dic = ckpt_dic['state_dict']
    optimizer.load_state_dict(ckpt_dic['optimizer'])
    sdic = model.state_dict()
    if args.rank == 0:
        print(args.rank, 'Random Loading Sum preweights: ', sum([torch.sum(sdic[ky]) for ky in sdic.keys()]))
    load_model_parameters_resume(args, model, loaded_state_dic)
    if args.rank == 0:
        print(args.rank, 'Random Loading Sum afterweights: ', sum([torch.sum(sdic[ky]) for ky in sdic.keys()]))
    return  ckpt_dic['epoch']


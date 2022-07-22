# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from helper.my_enums import ChannelModal,EvalType,ModalMergeType
import my_models.LinearEvalModels as LEM
from my_models.mcommons import (get_video_feature_extractor,get_video_dim,MultiPrototypes)

class AVModel(nn.Module):
    def __init__(
        self,
        args
    ):
        super(AVModel, self).__init__()

        # Save proprties
        self.return_features = False
        self.pre_protype_normalize=args.pre_protype_normalize
        self.share_proj_head=args.share_proj_head
        self.video_channels=args.video_channels
        self.video_channels_val=args.video_channels_val
        self.video_modal_merge=args.video_modal_merge
        self.video_modal_merge_val=args.video_modal_merge_val
        self.evaluation_type = args.evaluation_type
        self.video_network = get_video_feature_extractor(
            args.vid_base_arch,
            pool_type=args.pool_type,
            pretrained=args.pretrained,
            rank=args.rank)

        # print(self.evaluation_type , EvalType.PreTrain.value , args.pre_train_model.split('/')[-4],ChannelModal.rgb_flow.value)
        if self.video_channels == ChannelModal.rgb_flow.value or \
                (self.evaluation_type != EvalType.PreTrain.value and args.pre_train_model.split('/')[-4]==ChannelModal.rgb_flow.value):
            self.video_network_flow = get_video_feature_extractor(
                args.vid_base_arch,
                pool_type=args.pool_type,
                pretrained=args.pretrained,
                rank=args.rank)



        if self.evaluation_type in [EvalType.LinCls.value,EvalType.FT.value]:
            self.linear_classifier =LEM.RegLog(args.nclass, args.vid_base_arch,
                                       use_lincls_use_bn=args.use_lincls_use_bn,
                                       use_lincls_l2_norm=args.use_lincls_l2_norm,lincls_drop=args.lincls_drop)

        elif self.evaluation_type in [EvalType.PreTrain.value,EvalType.KNN_emb_prot.value,
                                      EvalType.KNN_all.value]:
            self.init_head(args)
        # elif self.evaluation_type in [3]:
        #     if args.rank==0:
        #         print('No Head, No Linear Layer')

    def init_head(self,args):
        self.encoder_dim = get_video_dim(args.vid_base_arch)
        if self.video_channels == ChannelModal.rgb_flow.value:
            if self.video_modal_merge == ModalMergeType.Concat.value:
                self.encoder_dim=self.encoder_dim*2 #we are concataneting our two features.

        self.hidden_mlp = args.hidden_mlp
        self.emedding_dim = args.emedding_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_mlp),
            nn.BatchNorm1d(self.hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_mlp, self.hidden_mlp),
            nn.BatchNorm1d(self.hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_mlp, self.emedding_dim),
        )
        if self.share_proj_head==False and  self.video_channels == ChannelModal.rgb_flow.value:
            self.projection_head_flow = nn.Sequential(
                nn.Linear(self.encoder_dim, self.hidden_mlp),
                nn.BatchNorm1d(self.hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_mlp, self.hidden_mlp),
                nn.BatchNorm1d(self.hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_mlp, self.emedding_dim),
            )
        # prototype layer
        self.prototypes = None
        if isinstance(args.nmb_prototypes, list):
            self.prototypes = MultiPrototypes(self.emedding_dim, args.nmb_prototypes)
        elif args.nmb_prototypes > 0:
            self.prototypes = nn.Linear(self.emedding_dim, args.nmb_prototypes, bias=False)

    def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_prots(self,x):
        if self.pre_protype_normalize:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x,self.prototypes(x)

    def forward_head(self, emb,modality='rgb'):
        if self.evaluation_type in [EvalType.PreTrain.value,EvalType.KNN_emb_prot.value,
                                      EvalType.KNN_all.value]:
            if modality=='flow' and self.share_proj_head==False and  self.video_channels == ChannelModal.rgb_flow.value:
                x = self.projection_head_flow(emb)
            else:
                x = self.projection_head(emb)

            x, prots=self.forward_prots(x)
            if self.evaluation_type in [EvalType.PreTrain.value]:
                return x, prots
            if self.evaluation_type in [EvalType.KNN_emb_prot.value]:
                return emb,prots
            if self.evaluation_type in [EvalType.KNN_all.value]:
                return emb, x, prots

        elif self.evaluation_type in [EvalType.LinCls.value,EvalType.FT.value]:
            return self.linear_classifier(emb)
        elif self.evaluation_type in [EvalType.KNN_emb.value]:
            return emb


    def forward(self, inps, whichhead=0):
        if self.video_channels==ChannelModal.rgb_flow.value:
            inps_rgb=[inp[0] for inp in inps]
            inps_flow=[inp[1] for inp in inps]
            inps=inps_rgb

        if not isinstance(inps, list):
            inps = [inps]
        if self.video_channels == ChannelModal.rgb_flow.value:
            if not isinstance(inps_flow, list):
                inps_flow = [inps_flow]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-3] for inp in inps]),
            return_counts=True,
        )[1], 0)

        start_idx = 0
        # print('+'*10,idx_crops,' ln',len(inps), 'shp:',inps[0].shape)
        for end_idx in idx_crops:
            curr_inp = torch.cat(inps[start_idx: end_idx]).cuda(non_blocking=True)
            # print('curr_inp:',curr_inp.shape)
            if self.video_channels == ChannelModal.rgb_flow.value:
                curr_inp_flow = torch.cat(inps_flow[start_idx: end_idx]).cuda(non_blocking=True)


            if self.evaluation_type in [EvalType.LinCls.value]: #if we are doing linear eval....
                with torch.no_grad():
                    _out = self.video_network(curr_inp).squeeze()
                    if self.video_channels == ChannelModal.rgb_flow.value:
                        _out_flow = self.video_network_flow(curr_inp_flow).squeeze()
            else:
                _out = self.video_network(curr_inp).squeeze()
                if self.video_channels == ChannelModal.rgb_flow.value:
                    _out_flow = self.video_network_flow(curr_inp_flow).squeeze()
            # print('----' * 10, _out.shape,curr_inp.shape)
            if len(_out.shape)==1:
                _out=_out.unsqueeze(0)
                if self.video_channels == ChannelModal.rgb_flow.value:
                    _out_flow = _out_flow.unsqueeze(0)

            if start_idx == 0:
                output = _out
                if self.video_channels == ChannelModal.rgb_flow.value:
                    output_flow = _out_flow
            else:
                output = torch.cat((output, _out))
                if self.video_channels == ChannelModal.rgb_flow.value:
                    output_flow = torch.cat((output_flow, _out_flow))
            start_idx = end_idx
        if self.video_channels == ChannelModal.rgb_flow.value:
            if self.evaluation_type in [EvalType.PreTrain.value]:  # if we are doing linear
                if self.video_modal_merge==ModalMergeType.Avg.value:
                    # print(output.shape,output_flow.shape)
                    if self.share_proj_head == False:
                        merge_output=(self.projection_head(output) + self.projection_head_flow(output_flow)) / 2
                        _,final_output=self.forward_prots(merge_output)
                        return merge_output,final_output
                    else:
                        # print('Merging', output.shape, output_flow.shape, output.mean(), output_flow.mean())
                        final_output =(output+output_flow)/2
                        return self.forward_head(final_output)
                elif self.video_modal_merge == ModalMergeType.ProcSeperate.value:
                    # print(output.shape,output_flow.shape,torch.cat((output,output_flow),1).shape)
                    emb,final_pred=self.forward_head(output,modality='rgb')
                    emb_flow,final_pred_flow=self.forward_head(output_flow,modality='flow')
                    return [emb,emb_flow],[final_pred,final_pred_flow]
            elif self.evaluation_type in [EvalType.LinCls.value,EvalType.FT.value]:  # if we are doing linear
                if self.video_modal_merge_val==ModalMergeType.Avg.value: #take average
                    if self.video_channels_val == ChannelModal.rgb_flow.value:
                        final_output = (output + output_flow) / 2
                    elif self.video_channels_val == ChannelModal.rgb.value:
                        final_output = output
                    elif self.video_channels_val == ChannelModal.flow.value:
                        final_output = output_flow
                    return self.forward_head(final_output)

                # elif self.video_modal_merge_val == ModalMergeType.ProcSeperate.value: # FW pass seperately
                #     # print(output.shape,output_flow.shape,torch.cat((output,output_flow),1).shape)
                #     emb, final_pred = self.forward_head(output)
                #     emb_flow, final_pred_flow = self.forward_head(output_flow)
                #     return [emb, emb_flow], [final_pred, final_pred_flow]
            elif self.evaluation_type in EvalType.KNN_FullOpt.value:  # if we are doing knn
                if self.video_modal_merge_val==ModalMergeType.Avg.value: #take average
                    if self.video_channels_val==ChannelModal.rgb_flow.value:
                        final_output =(output+output_flow)/2
                    elif self.video_channels_val==ChannelModal.rgb.value:
                        final_output =output
                    elif self.video_channels_val == ChannelModal.flow.value:
                        final_output =output_flow
                    return self.forward_head(final_output)
                elif self.video_modal_merge_val == ModalMergeType.ProcSeperate.value: # FW pass seperately
                    # print(output.shape,output_flow.shape,torch.cat((output,output_flow),1).shape)
                    emb, final_pred = self.forward_head(output,'rgb')
                    emb_flow, final_pred_flow = self.forward_head(output_flow,'flow')
                    return [emb, emb_flow], [final_pred, final_pred_flow]
        else:
            return self.forward_head(output)



# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

QUEEN_NUM = 8
QUEEN_LENGTH = 5

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        # self.D_img = None
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.bceLoss_func = nn.BCEWithLogitsLoss()

        ##############################   2023.02.24  ####################################################

        self.register_buffer("NF",torch.zeros([QUEEN_NUM, QUEEN_LENGTH, 1024]))  #[8,5,1024]
        self.register_buffer("NS", torch.zeros([QUEEN_NUM, QUEEN_LENGTH, 1]))  #[8,5,1]
        self.register_buffer("queue_ptr", torch.zeros([QUEEN_NUM, 1], dtype=torch.int64)) #[8,1]

        ##################################################################################################


    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # "dis_loss_ratio": cfg.xxx,
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t
    
    ###################################  2023.02.24  ##############################################
    @torch.no_grad()
    def confidence_momentum_update(self,negative_feat, cls_max_index, score):
        #negative_feat.shape[954,1024] score.shape[954,1]
        negative_feat = negative_feat.unsqueeze(1) #negative_feat.shape[954,1,1024] negative_feat[i].shape[1,1024]
        score = score.unsqueeze(1) #score.shape[954,1,1] score[i].shape[1,1]

        # cls_max_index范围[0,8), 因为预测为背景的8在提取FP时已经筛掉, shape[954]
        for i in range(score.shape[0]):
            c = cls_max_index[i]
            ptr = int(self.queue_ptr[c])
            # queen not full
            if (ptr < QUEEN_LENGTH):
                self.NF[c][ptr],self.NS[c][ptr] = negative_feat[i], score[i]  # torch.size([1,1024]) torch.size([1,1])
                ptr += 1
                self.queue_ptr[c][0] = ptr
            else:  #negative_feat[i].shape[1,1024] self.NF[c][0:5].shape[5,1024]
                similarity = torch.cosine_similarity(negative_feat[i].unsqueeze(1), self.NF[c][0:5].unsqueeze(0), dim=2) #similarity.shape[1,5]
                mmax = torch.argmax(similarity, dim=1) #mmax.shape:[1]
                fm, sm = self.NF[c][mmax].clone().detach(),self.NS[c][mmax].clone().detach()  # NF[c][mmax].shape:[1,1024] NS[c][mmax].shape:[1,1]
                self.NF[c][mmax] = torch.mm(torch.div(sm,(sm + score[i])) , fm) + torch.mm(torch.div(score[i],(sm + score[i])) , negative_feat[i]) #shape[1,1024]
                self.NS[c][mmax] = torch.mm(torch.div(sm,(sm + score[i])) , sm) + torch.mm(torch.div(score[i],(sm + score[i])) , score[i]) #shape[1,1]


    
    def mem_loss(self,nf_feature,nf_cls):
        #cmp loss
        #unq loss
        # nf_features [954,1024] nf_cls [954]
        #前提：当每个bank里的mj满足5个之后，bank开始更新，才使用该loss
        #且负特征个数不为0
        losses = {} #字典，存储cmp,unq
        l2_loss = nn.MSELoss() #l2 loss
        sim_max_feat = []
        mj_feat = []
        anchor,positive,negative = [],[],[]
        for i in range(QUEEN_NUM):
            inds = ( nf_cls == i )  #选出第i类的index
            if inds.any():  #如果存在第i类的负特征
                spec_feat = nf_feature[inds]  #加入第i类有11个,[11,1024]
                mj = self.NF[i]  # 第i个mem bank [5,1024]
                similarity = torch.cosine_similarity(mj.unsqueeze(1), spec_feat.unsqueeze(0), dim=2)  # [5,11]

                #############  只有一个负特征可优化  #############################
                #如果第i类只有一个负特征，只有cmp,没有unq
                if len(spec_feat) == 1:
                    index = torch.max(similarity,dim=1).indices  #[5]
                    sim_max_feat.append(spec_feat[index]) #[5,1024]
                    mj_feat.append(mj)
                else:
                #找第二大相似：
                #如果存在两个以上nf，说明可以获得第二相似，存储此时的mj,gskp,gskn,既有cmp又有unq
                    anchor.append(mj)
                    temp, idx = torch.topk(similarity, 2, dim=1)  #temp,idx [5,2] 获得前2大值的下标
                    idx1, idx2 = idx[:,0], idx[:,1]  #获得第一,二大值的下标 [5]
                    positive.append(spec_feat[idx1])
                    negative.append(spec_feat[idx2])
                    #cmp
                    sim_max_feat.append(spec_feat[idx1]) #[5,1024]
                    mj_feat.append(mj)

        Gk = torch.cat([sim for sim in sim_max_feat], dim=0) if len(sim_max_feat) else torch.empty(0) #[40,1024] 可导
        Mk = torch.cat([ff for ff in mj_feat], dim=0) if len(mj_feat) else torch.empty(0) #[40,1024]  不可导
        cmp_loss = l2_loss(Gk,Mk)

        #unq loss
        Ancho = torch.cat([aho for aho in anchor], dim=0) if len(anchor) else torch.empty(0) #[xx,1024] 不可导
        pos = torch.cat([ps for ps in positive], dim=0) if len(positive) else torch.empty(0) #[xx,1024],可导
        neg = torch.cat([ng for ng in negative], dim=0) if len(negative) else torch.empty(0) #[xx,1024],可导

        # 如果inds全为空,loss会是nan
        # sim_max_feat和mj_feat要么同时为空，要么形状一样
        # mj不用梯度反向传播更新
        Ancho1 = Ancho.clone().detach()
        triplet_loss = nn.TripletMarginLoss(margin=0.005, p=2) #默认情况，可改
        unq_loss = triplet_loss(Ancho1, pos, neg)

        losses['cmp_loss'] = cmp_loss * 0.1
        losses['unq_loss'] = unq_loss * 0.01

        return losses
    
    def nicl_loss(self,pseudo_feat,ROI_predictions):

        # pseudo_feat[2000,1024] ROI_predictions[0]  [2000,9]
        pseudo_feat = pseudo_feat.unsqueeze(1)  # pseudo_feat.shape[2000,1,1024]
        pseudo_score = ROI_predictions[0][:,:8]  # pseudo_score [2000,8] 去掉最后一列对背景的预测
        cls_max = torch.argmax(pseudo_score, 1)  # cls_max.shape[2000]
        sim_matrix = torch.index_select(self.NF, dim=0, index=cls_max)  # sim_matrix.shape[2000,5,1024]
        sim = torch.cosine_similarity(pseudo_feat, sim_matrix, dim=2)  # sim.shape[2000,5]
        sim_values = torch.max(sim, 1).values.unsqueeze(1)  # sim_values.shape[2000,1]
        mmax = torch.argmax(sim, 1).view(cls_max.shape[0], 1, 1)  # mmax.shape[2000,1,1]
        ns = torch.index_select(self.NS, dim=0, index=cls_max)  # ns.shape[2000,5,1]
        max_ns = torch.gather(ns, 1, mmax).squeeze(2)  # max_ns.shape[2000,1]
        loss = 0.3 * max_ns * sim_values  # loss.shape[2000,1]

        return loss.mean()

    ################################################################################################

    

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1

        if branch == "domain":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)
            
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)

            # import pdb
            # pdb.set_trace()
           
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_t = self.backbone(images_t.tensor)
            
            features_t = grad_reverse(features_t[self.dis_type])
            # features_t = grad_reverse(features_t['p2'])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            # import pdb
            # pdb.set_trace()

            #

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            losses["loss_D_img_t"] = loss_D_img_t
            return losses, [], [], None

        
        # self.D_img.eval()
        
        images = self.preprocess_image(batched_inputs)
        

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        
        # TODO: remove the usage of if else here. This needs to be re-organized

        if branch.startswith("warm_up_supervised"):

            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_s"] = loss_D_img_s*0.001

            return losses, [], [], None

            #############################################################################################

        elif branch == "supervised":

            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            #print("新加内容")
            ##############################################################################
            
            loss_nice, nf_feature, nf_score, nf_cls = self.roi_heads._forward_fp(
                features,
                proposals_rpn,
                targets=gt_instances,
                branch=branch,)
            
            #print('loss_nice')
            #print(loss_nice)
            
            # update memory bank
            with torch.no_grad():
                if min(nf_feature.shape) != 0:
                    nf = nf_feature.detach()
                    ns = nf_score.detach()
                    ncls = nf_cls.detach()
                    self.confidence_momentum_update(nf, ncls, ns)

            ###############################################################################

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_s"] = loss_D_img_s*0.001

            ########################   2023.02.26   ###################################################
            if not torch.isnan(loss_nice):
                losses['loss_nice'] = loss_nice * 0.01

            # 若队列满足 5 个mj
            # 若存在小于8个负实例
            
            booltensor = torch.zeros([QUEEN_NUM,1], dtype=torch.int64).cuda()+5
            if self.queue_ptr.equal(booltensor):
                mem_losses = self.mem_loss( nf_feature, nf_cls )
                if not torch.isnan(mem_losses['cmp_loss']):
                    losses['loss_cmp'] = mem_losses['cmp_loss']
                if not torch.isnan(mem_losses['unq_loss']):
                    losses['loss_unq'] = mem_losses['unq_loss']
            

            ###########################################################################################
            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_nicl":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            ###########################2023.03.06################################

            box_features = self.roi_heads._forward_boxfeature(features,proposals_rpn)  #[2000,1024]

            loss_nicl = self.nicl_loss(box_features,ROI_predictions)
            record_nicl = {'loss_nicl':loss_nicl}

            #########################################################################

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            #return {}, proposals_rpn, proposals_roih, ROI_predictions
            return record_nicl, proposals_rpn, proposals_roih, ROI_predictions
        
        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )


            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None



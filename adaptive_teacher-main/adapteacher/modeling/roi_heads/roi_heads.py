# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from adapteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler

tao = 0.1

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        #debug
        predictions = self.box_predictor(box_features)
        del box_features

        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
    

    #############################################################################################

    def NegativeInstance(self,cls_scores,gt_classes,box_features):

        cls_scores = F.softmax(cls_scores,dim=1)   #对cls_score做softmax,进行归一化,防止loss nan

        pred_classes = cls_scores.argmax(dim=1) #[1024]
        maxscore = torch.max(cls_scores, 1).values #[1024]
        bg_class_ind = cls_scores.shape[1] - 1  #背景坐标8

        #找出并去掉pred里的背景，因为pred背景是FN和TN，不需要
        fg_inds = (pred_classes >= 0) & (pred_classes < bg_class_ind)
        fg_gt_classes, fg_pred_classes, fg_maxscore = gt_classes[fg_inds], pred_classes[fg_inds], maxscore[fg_inds]

        #box_features第一次筛选 [1024,1024]
        inds1 = torch.nonzero(fg_inds).squeeze(dim=1)
        temp_feat = torch.index_select(box_features, dim=0, index=inds1)

        #TP是pred和gt相同,FP是pred和gt不同,并进行一次分数过滤0.1
        fp_inds = (fg_gt_classes != fg_pred_classes) & (fg_maxscore >= tao)
        fp_gt_classes, fp_pred_classes, fp_maxscore = fg_gt_classes[fp_inds], fg_pred_classes[fp_inds], fg_maxscore[fp_inds] # torch.Size([954])

        #box_features第二次筛选
        inds2 = torch.nonzero(fp_inds).squeeze(dim=1)
        fp_feat = torch.index_select(temp_feat, dim=0, index=inds2)  # torch.Size([954, 1024])

        fp_maxscore = fp_maxscore.view(fp_maxscore.shape[0], 1) #[954,1]

        return fp_feat, fp_pred_classes, fp_maxscore

    def nice_losses(self,fpscore):
        loss = -torch.log(1 - fpscore) #torch.Size([954])
        return loss.mean()

    #如果训练刚开始没有高质量的负实例，loss实现需要低质量分数
    def SpecialNegativeInstance(self,cls_scores,gt_classes):

        cls_scores = F.softmax(cls_scores,dim=1)   #对cls_score做softmax,进行归一化,防止loss nan

        #如果是因为tao的原因没有高质量负实例

        pred_classes = cls_scores.argmax(dim=1)  # [1024]
        bg_class_ind = cls_scores.shape[1] - 1  #背景坐标8

        #找出并去掉pred里的背景，因为pred背景是FN和TN，不需要
        fg_inds = (pred_classes >= 0) & (pred_classes < bg_class_ind)
        if fg_inds.any():  #如果有True
            maxscore = torch.max(cls_scores, 1).values #[1024]
            fg_gt_classes, fg_pred_classes, fg_maxscore = gt_classes[fg_inds], pred_classes[fg_inds], maxscore[fg_inds]

            #TP是pred和gt相同,FP是pred和gt不同
            fp_inds = (fg_gt_classes != fg_pred_classes)
            fp_maxscore = fg_maxscore[fp_inds]
            #special_score = max(fp_maxscore).unsqueeze(0)

        #如果不是tao的原因，是网络预测全为背景
        #存在没有fp的情况，此时fp_maxscore shape:torch.Size([0])
        else:
           fn_cls_scores = cls_scores[:,:8]  #[1024,8] 去掉最后一列预测背景的分数
           id1 = (gt_classes != bg_class_ind)
           pr_classes, mscore = fn_cls_scores.argmax(dim=1), torch.max(fn_cls_scores, 1).values #[1024]
           fg_gt_classes, fg_pred_classes, fg_maxscore = gt_classes[id1], pr_classes[id1], mscore[id1]  #[40]

           id2 = (fg_gt_classes != fg_pred_classes)
           fp_maxscore = fg_maxscore[id2]

        return fp_maxscore



    def _forward_fp( 
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        branch="",):

        #获得proposals
        assert targets
        # 1000 --> 512
        proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
        )

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  #box_features.shape=[1024,512,7,7]
        #1024 = 512 * 2   proposal个数 * batch_size
        box_features = self.box_head(box_features) #box_features.shape = [1024,1024]  [proposal个数 * batch_size , C]
        predictions = self.box_predictor(box_features)  #predictions[0] = score , predictions[1] = 坐标偏移量

        #2023.2.22 FP收集函数,返回负特征，负特征分数，对应的类
        cls_scores, _ = predictions
        gt_classes = ( torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0) ) #[1024]

        nf_feature, nf_cls, nf_score = self.NegativeInstance(cls_scores,gt_classes,box_features)

        #增加loss 筛选高质量负实例
        #先判断是否存在高质量负实例
        if nf_score.shape[0] == 0:
            fp_maxscore = self.SpecialNegativeInstance(cls_scores,gt_classes)
            if fp_maxscore.shape[0] == 0:  #网络学习太差,没有FP存在
                loss_nice = self.nice_losses(fp_maxscore)  #loss_nice is nan
            else:
                special_score = max(fp_maxscore).unsqueeze(0)
                loss_nice = self.nice_losses(special_score)
        else:
            loss_nice = self.nice_losses(nf_score)

        return loss_nice, nf_feature, nf_score, nf_cls
    

        #2023.03.06
    #获得伪标签ROI box feature [2014,1024]
    def _forward_boxfeature(self,features: Dict[str, torch.Tensor],proposals: List[Instances] ):

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features,[x.proposal_boxes for x in proposals])  # box_features.shape=[1024,512,7,7]
        # 1024 = 512 * 2   proposal个数 * batch_size
        box_features = self.box_head(box_features)  # box_features.shape = [1024,1024]  [proposal个数 * batch_size , C]

        return box_features
        

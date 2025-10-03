import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanodet.util import (
    bbox2distance,
    distance2bbox,
    multi_apply,
    overlay_bbox_cv,
    warp_rboxes,
)

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import Integral, reduce_mean


class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
        use_rotated_boxes (bool): Enable rotated bounding box prediction.
            Default: False.
        rot_wh_loss_weight (float): Loss weight for rotated width and height.
            Default: 1.0.
        rot_angle_loss_weight (float): Loss weight for rotated angle regression.
            Default: 1.0.
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        use_rotated_boxes=False,
        rot_wh_loss_weight=1.0,
        rot_angle_loss_weight=1.0,
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.distribution_project = Integral(self.reg_max)

        self.use_rotated_boxes = use_rotated_boxes
        self.rot_wh_loss_weight = rot_wh_loss_weight
        self.rot_angle_loss_weight = rot_angle_loss_weight
        self.rot_dim = 4 if self.use_rotated_boxes else 0

        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.num_classes + 4 * (self.reg_max + 1),
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )
        if self.use_rotated_boxes:
            self.rot_convs = nn.ModuleList(
                [
                    nn.Conv2d(self.feat_channels, self.rot_dim, 1, padding=0)
                    for _ in self.strides
                ]
            )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return cls_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")
        if self.use_rotated_boxes:
            for conv in self.rot_convs:
                normal_init(conv, std=0.01)

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        rot_layers = self.rot_convs if self.use_rotated_boxes else [None] * len(
            self.strides
        )
        for feat, cls_convs, gfl_cls, rot_conv in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
            rot_layers,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            if self.use_rotated_boxes:
                rot_pred = rot_conv(feat)
                output = torch.cat([output, rot_pred], dim=1)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        device = preds.device
        batch_size = preds.shape[0]
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]

        gt_bboxes_ignore = gt_meta["gt_bboxes_ignore"]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(batch_size)]
        gt_rbboxes = gt_meta.get("gt_rbboxes")
        if gt_rbboxes is None:
            gt_rbboxes = [None for _ in range(batch_size)]
        gt_rbboxes_ignore = gt_meta.get("gt_rbboxes_ignore")
        if gt_rbboxes_ignore is None:
            gt_rbboxes_ignore = [None for _ in range(batch_size)]

        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        if self.use_rotated_boxes:
            cls_preds, reg_preds, rot_preds = preds.split(
                [
                    self.num_classes,
                    4 * (self.reg_max + 1),
                    self.rot_dim,
                ],
                dim=-1,
            )
        else:
            cls_preds, reg_preds = preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            rot_preds = None
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            # use auxiliary head to assign
            if self.use_rotated_boxes:
                aux_cls_preds, aux_reg_preds, aux_rot_preds = aux_preds.split(
                    [
                        self.num_classes,
                        4 * (self.reg_max + 1),
                        self.rot_dim,
                    ],
                    dim=-1,
                )
            else:
                aux_cls_preds, aux_reg_preds = aux_preds.split(
                    [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
                )
                aux_rot_preds = None
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
                gt_rbboxes,
            )
        else:
            # use self prediction to assign
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
                gt_rbboxes,
            )

        loss, loss_states = self._get_loss_from_assign(
            cls_preds,
            reg_preds,
            decoded_bboxes,
            batch_assign_res,
            center_priors,
            rot_preds,
        )

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds,
                aux_reg_preds,
                aux_decoded_bboxes,
                batch_assign_res,
                center_priors,
                aux_rot_preds,
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    def _get_loss_from_assign(
        self,
        cls_preds,
        reg_preds,
        decoded_bboxes,
        assign,
        center_priors,
        rot_preds=None,
    ):
        device = cls_preds.device
        if self.use_rotated_boxes:
            (
                labels,
                label_scores,
                label_weights,
                bbox_targets,
                dist_targets,
                num_pos,
                wh_targets,
                angle_targets,
                angle_weights,
            ) = assign
        else:
            (
                labels,
                label_scores,
                label_weights,
                bbox_targets,
                dist_targets,
                num_pos,
            ) = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        label_weights = torch.cat(label_weights, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        center_priors = center_priors.reshape(-1, 4)
        loss_qfl = self.loss_qfl(
            cls_preds,
            (labels, label_scores),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        loss_rot_wh = None
        loss_rot_angle = None
        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )

            if self.use_rotated_boxes:
                if rot_preds is None:
                    zero = reg_preds.sum() * 0
                    loss_rot_wh = zero
                    loss_rot_angle = zero
                else:
                    rot_preds = rot_preds.reshape(-1, self.rot_dim)
                    wh_targets = torch.cat(wh_targets, dim=0)
                    angle_targets = torch.cat(angle_targets, dim=0)
                    angle_weights = torch.cat(angle_weights, dim=0)

                    pos_rot_preds = rot_preds[pos_inds]
                    pos_wh_targets = wh_targets[pos_inds]
                    pos_angle_targets = angle_targets[pos_inds]
                    pos_angle_weights = angle_weights[pos_inds]

                    valid_rot_inds = torch.nonzero(
                        pos_angle_weights > 0, as_tuple=False
                    ).squeeze(1)
                    if len(valid_rot_inds) > 0:
                        rot_weight = weight_targets[valid_rot_inds]
                        wh_loss = F.smooth_l1_loss(
                            pos_rot_preds[valid_rot_inds, :2],
                            pos_wh_targets[valid_rot_inds],
                            reduction="none",
                        ).sum(dim=1)
                        loss_rot_wh = (wh_loss * rot_weight).sum() / bbox_avg_factor

                        angle_pred = F.normalize(
                            pos_rot_preds[valid_rot_inds, 2:], dim=1
                        )
                        angle_loss = F.smooth_l1_loss(
                            angle_pred,
                            pos_angle_targets[valid_rot_inds],
                            reduction="none",
                        ).sum(dim=1)
                        loss_rot_angle = (angle_loss * rot_weight).sum() / bbox_avg_factor
                    else:
                        zero = pos_rot_preds.sum() * 0
                        loss_rot_wh = zero
                        loss_rot_angle = zero
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0
            if self.use_rotated_boxes:
                zero = reg_preds.sum() * 0
                loss_rot_wh = zero
                loss_rot_angle = zero

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        if self.use_rotated_boxes:
            if loss_rot_wh is None:
                zero = reg_preds.sum() * 0
                loss_rot_wh = zero
                loss_rot_angle = zero
            loss_rot_wh = loss_rot_wh * self.rot_wh_loss_weight
            loss_rot_angle = loss_rot_angle * self.rot_angle_loss_weight
            loss = loss + loss_rot_wh + loss_rot_angle
            loss_states["loss_rot_wh"] = loss_rot_wh
            loss_states["loss_rot_angle"] = loss_rot_angle
        return loss, loss_states

    @torch.no_grad()
    def target_assign_single_img(
        self,
        cls_preds,
        center_priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_rbboxes=None,
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_rbboxes (Tensor, optional): Rotated ground truth bboxes of one
                image in (cx, cy, w, h, angle) format.
        """

        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = torch.from_numpy(gt_bboxes_ignore).to(device)
            gt_bboxes_ignore = gt_bboxes_ignore.to(decoded_bboxes.dtype)
        if self.use_rotated_boxes and gt_rbboxes is not None:
            gt_rbboxes = torch.from_numpy(gt_rbboxes).to(device)
            gt_rbboxes = gt_rbboxes.to(decoded_bboxes.dtype)
        else:
            gt_rbboxes = None

        assign_result = self.assigner.assign(
            cls_preds,
            center_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )

        num_priors = center_priors.size(0)
        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_weights = center_priors.new_zeros(num_priors, dtype=torch.float)
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)
        if self.use_rotated_boxes:
            wh_targets = center_priors.new_zeros((num_priors, 2))
            angle_targets = center_priors.new_zeros((num_priors, 2))
            angle_weights = center_priors.new_zeros(num_priors)

        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
            if self.use_rotated_boxes and gt_rbboxes is not None and gt_rbboxes.numel() > 0:
                assigned_rbboxes = gt_rbboxes[pos_assigned_gt_inds]
                stride = center_priors[pos_inds, 2:4]
                wh_targets[pos_inds] = torch.log(
                    torch.clamp(assigned_rbboxes[:, 2:4] / stride, min=1e-6)
                )
                angle_rad = assigned_rbboxes[:, 4] * math.pi / 180.0
                angle_targets[pos_inds, 0] = torch.sin(angle_rad)
                angle_targets[pos_inds, 1] = torch.cos(angle_rad)
                angle_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if self.use_rotated_boxes:
            return (
                labels,
                label_scores,
                label_weights,
                bbox_targets,
                dist_targets,
                num_pos_per_img,
                wh_targets,
                angle_targets,
                angle_weights,
            )
        return (
            labels,
            label_scores,
            label_weights,
            bbox_targets,
            dist_targets,
            num_pos_per_img,
        )

    def sample(self, assign_result, gt_bboxes):
        """Sample positive and negative bboxes."""
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
        """
        if self.use_rotated_boxes:
            cls_scores, bbox_preds, rot_preds = preds.split(
                [
                    self.num_classes,
                    4 * (self.reg_max + 1),
                    self.rot_dim,
                ],
                dim=-1,
            )
        else:
            cls_scores, bbox_preds = preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            rot_preds = None
        result_list = self.get_bboxes(cls_scores, bbox_preds, rot_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            if self.use_rotated_boxes:
                warped = warp_rboxes(
                    det_bboxes[:, :5], np.linalg.inv(warp_matrix), img_width, img_height
                )
                det_bboxes = np.concatenate(
                    [warped.astype(np.float32), det_bboxes[:, 5:6].astype(np.float32)],
                    axis=1,
                )
            else:
                det_bboxes[:, :4] = warp_boxes(
                    det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
                )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                if self.use_rotated_boxes:
                    det_result[i] = det_bboxes[inds].astype(np.float32).tolist()
                else:
                    det_result[i] = np.concatenate(
                        [
                            det_bboxes[inds, :4].astype(np.float32),
                            det_bboxes[inds, 4:5].astype(np.float32),
                        ],
                        axis=1,
                    ).tolist()
            det_results[img_id] = det_result
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, rot_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            rot_preds (Tensor, optional): Shape (num_imgs, num_points, rot_dim)
                when ``use_rotated_boxes`` is enabled.
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        if self.use_rotated_boxes and rot_preds is not None:
            rot_preds = rot_preds.reshape(b, -1, self.rot_dim)
            wh_log = rot_preds[..., :2]
            angle_vec = F.normalize(rot_preds[..., 2:], dim=-1)
            wh = torch.exp(wh_log) * center_priors[..., 2:4]
            angles = torch.atan2(angle_vec[..., 0], angle_vec[..., 1])
            angles = angles * 180.0 / math.pi
            rboxes = torch.cat([center_priors[..., :2], wh, angles.unsqueeze(-1)], dim=-1)
        else:
            rboxes = None
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            if self.use_rotated_boxes and rboxes is not None:
                results = multiclass_nms_rotated(
                    rboxes[i],
                    score,
                    score_thr=0.05,
                    nms_cfg=dict(type="nms", iou_threshold=0.6),
                    max_num=100,
                )
            else:
                results = multiclass_nms(
                    bbox,
                    score,
                    score_thr=0.05,
                    nms_cfg=dict(type="nms", iou_threshold=0.6),
                    max_num=100,
                )
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    def _forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            cls_pred, reg_pred = output.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=1
            )
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)

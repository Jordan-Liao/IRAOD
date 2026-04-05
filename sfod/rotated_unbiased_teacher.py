# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Re-implementation: Unbiased teacher for semi-supervised object detection

There are several differences with official implementation:
1. we only use the strong-augmentation version of labeled data rather than the strong-augmentation and weak-augmentation version of labeled data.
"""
import math
import numpy as np
import torch
import os

import cv2
import mmcv
from mmcv.runner.dist_utils import get_dist_info

from mmdet.utils import get_root_logger
from mmdet.models.builder import DETECTORS
from mmrotate.core.bbox import rbbox_overlaps

from .rotated_semi_two_stage import SemiTwoStageDetector
from mmrotate.core.visualization import imshow_det_rbboxes

@DETECTORS.register_module()
class UnbiasedTeacher(SemiTwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 # ema model
                 ema_config=None,
                 ema_ckpt=None,
                 # ut config
                 cfg=dict(),
                 ):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, neck=neck, pretrained=pretrained,
                         ema_config=ema_config, ema_ckpt=ema_ckpt)
        self.debug = cfg.get('debug', False)
        self.vis_dir = cfg.get('vis_dir', None)
        self.num_classes = self.roi_head.bbox_head.num_classes
        self.cur_iter = 0
        
        # hyper-parameter
        self.score_thr = cfg.get('score_thr', 0.7)
        self.weight_u = cfg.get('weight_u', 1.0)
        # Safe defaults: keep supervised branch on and allow bbox regression on pseudo labels.
        # (Both being off can easily collapse training to near-random mAP.)
        self.weight_l = cfg.get('weight_l', 1.0)
        self.use_bbox_reg = cfg.get('use_bbox_reg', True)
        self.momentum = cfg.get('momentum', 0.998)

        # --- burn-in: disable unsupervised loss for the first N epochs ---
        self.burn_in_epochs = cfg.get('burn_in_epochs', 0)

        # --- threshold annealing (Exp B) ---
        self.thr_schedule = cfg.get('thr_schedule', 'fixed')  # 'fixed' or 'linear'
        self.score_thr_start = cfg.get('score_thr_start', self.score_thr)
        self.score_thr_end = cfg.get('score_thr_end', self.score_thr)

        # --- per-class threshold (Exp E): score_thr can be list/tuple ---
        if isinstance(self.score_thr, (list, tuple)):
            self.class_score_thr = list(self.score_thr)
            self.score_thr = max(self.class_score_thr)  # scalar fallback for logging
        else:
            self.class_score_thr = None

        # --- Round 3: save initial per-class thresholds for annealing ---
        self.class_score_thr_start = list(self.class_score_thr) if self.class_score_thr is not None else None
        # Per-class annealing end thresholds (Exp L)
        self.class_score_thr_end = cfg.get('class_score_thr_end', None)
        if self.class_score_thr_end is not None:
            self.class_score_thr_end = list(self.class_score_thr_end)

        # --- Round 3: wu epoch-based schedule (Exp M) ---
        # dict like {4: 0, 6: 0.25, 8: 0.5, 12: 0.25}
        self.weight_u_schedule = cfg.get('weight_u_schedule', None)

        # --- Round 3: separate bbox weight for unsup (Exp O) ---
        self.weight_u_bbox = cfg.get('weight_u_bbox', None)

        # --- EMA cosine schedule: momentum ramps from ema_momentum_start to ema_momentum_end ---
        self.ema_schedule = cfg.get('ema_schedule', 'fixed')  # 'fixed' or 'cosine'
        self.ema_momentum_end = cfg.get('ema_momentum_end', self.momentum)
        self.max_iters = cfg.get('max_iters', 47304)  # total training iters for cosine schedule

        # analysis
        self.image_num = 0
        self.pseudo_num = np.zeros(self.num_classes)
        self.pseudo_num_tp = np.zeros(self.num_classes)
        self.pseudo_num_gt = np.zeros(self.num_classes)

    def set_epoch(self, epoch): 
        self.roi_head.cur_epoch = epoch 
        self.roi_head.bbox_head.cur_epoch = epoch
        self.cur_epoch = epoch
        # Round 3: update global epoch for ConditionalDTRandCrop (Exp N)
        try:
            from sfod.dense_teacher_rand_aug import set_global_epoch
            set_global_epoch(epoch)
        except ImportError:
            pass
        
    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1,
    ):
        device = img.device
        self.image_num += len(img_metas_unlabeled)
        self.cur_iter += 1

        # --- Compute dynamic EMA momentum ---
        if self.ema_schedule == 'cosine':
            progress = min(self.cur_iter / max(self.max_iters, 1), 1.0)
            cur_momentum = self.momentum + (self.ema_momentum_end - self.momentum) * (1.0 - math.cos(math.pi * progress)) / 2.0
        else:
            cur_momentum = self.momentum

        self.update_ema_model(cur_momentum)

        # --- threshold annealing: update score_thr based on training progress ---
        if self.thr_schedule == 'linear':
            progress = min(self.cur_iter / max(self.max_iters, 1), 1.0)
            if self.class_score_thr is not None and self.class_score_thr_end is not None:
                # Per-class independent annealing (Exp L)
                self.class_score_thr = [
                    s + (e - s) * progress
                    for s, e in zip(self.class_score_thr_start, self.class_score_thr_end)
                ]
                self.score_thr = max(self.class_score_thr)
            else:
                new_thr = self.score_thr_start + (self.score_thr_end - self.score_thr_start) * progress
                if self.class_score_thr is not None and self.class_score_thr_start is not None:
                    ratio = new_thr / max(self.score_thr_start, 1e-6)
                    self.class_score_thr = [t * ratio for t in self.class_score_thr_start]
                self.score_thr = new_thr

        self.analysis()

        # --- Burn-in + scheduling: determine effective unsupervised weight ---
        cur_epoch = getattr(self, 'cur_epoch', 0)
        if cur_epoch < self.burn_in_epochs:
            effective_weight_u = 0.0
        elif self.weight_u_schedule is not None:
            # Epoch-based wu schedule (Exp M): find last matching epoch
            effective_weight_u = self.weight_u
            for ep_thr in sorted(self.weight_u_schedule.keys()):
                if cur_epoch >= ep_thr:
                    effective_weight_u = self.weight_u_schedule[ep_thr]
        else:
            effective_weight_u = self.weight_u

        # # ---------------------label data---------------------
        losses = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        losses = self.parse_loss(losses)
        for key, val in losses.items():
            if key.find('loss') == -1:
                continue
            else:
                losses[key] = self.weight_l * val
        # # -------------------unlabeled data-------------------
        bbox_transform = []
        bbox_results = self.inference_unlabeled(
            img_unlabeled, img_metas_unlabeled, rescale=True
        )
        gt_bboxes_pred, gt_labels_pred = self.create_pseudo_results(
            img_unlabeled_1, bbox_results, bbox_transform, device,
            gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled  # for analysis
        )
 
        losses_unlabeled = self.forward_train(img_unlabeled_1, img_metas_unlabeled_1,
                                              gt_bboxes_pred, gt_labels_pred)
        losses_unlabeled = self.parse_loss(losses_unlabeled)

        # [FIX] Sanitize NaN losses from degenerate pseudo-label bboxes
        # NaN in loss_bbox_unlabeled can corrupt BatchNorm running stats
        # during forward pass, causing irreversible NaN death spiral.
        for _k, _v in losses_unlabeled.items():
            if isinstance(_v, torch.Tensor) and not torch.isfinite(_v).all():
                losses_unlabeled[_k] = torch.zeros_like(_v)

        # --- Check if all pseudo labels are empty (pseudo_num == 0) ---
        total_pseudo = sum(len(b) for b in gt_bboxes_pred)

        # Compute effective bbox weight (Exp O: cls/reg split)
        if self.weight_u_bbox is not None:
            effective_weight_u_bbox = 0.0 if cur_epoch < self.burn_in_epochs else self.weight_u_bbox
        else:
            effective_weight_u_bbox = effective_weight_u

        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            if total_pseudo == 0:
                # No pseudo labels at all: zero out unsupervised loss to avoid noise
                losses_unlabeled[key] = 0.0 * val
            elif key.find('bbox') != -1:
                losses_unlabeled[key] = effective_weight_u_bbox * val if self.use_bbox_reg else 0 * val
            else:
                losses_unlabeled[key] = effective_weight_u * val
        losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})

        pseudo_sum = self.pseudo_num.sum()
        extra_info = {
            'pseudo_num': torch.Tensor([pseudo_sum / max(self.image_num, 1)]).to(device),
            'pseudo_num(acc)': torch.Tensor([self.pseudo_num_tp.sum() / max(pseudo_sum, 1e-10)]).to(device)
        }
        losses.update(extra_info)

        # [DEEP FIX] Ensure ALL ranks have identical loss keys.
        # The supervised forward_train may return different keys when gt is empty.
        # We enforce a canonical set of keys that MUST exist on every rank.
        device = next(iter(losses.values())).device if losses else torch.device('cpu')
        canonical_keys = [
            # Supervised loss keys (from RPN + ROI head)
            'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox',
            # Unlabeled loss keys
            'loss_rpn_cls_unlabeled', 'loss_rpn_bbox_unlabeled',
            'loss_cls_unlabeled', 'acc_unlabeled', 'loss_bbox_unlabeled',
            # Pseudo label stats
            'pseudo_num', 'pseudo_num(acc)',
        ]
        for k in canonical_keys:
            if k not in losses:
                losses[k] = torch.tensor(0.0, device=device, requires_grad=False)

        return losses
    
    def create_pseudo_results(self, img, bbox_results, box_transform, device,
                              gt_bboxes=None, gt_labels=None, img_metas=None):
        """using dynamic score to create pseudo results"""
        gt_bboxes_pred, gt_labels_pred = [], []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None
        for b, result in enumerate(bbox_results):
            bboxes, labels = [], []
            if use_gt:
                gt_bbox, gt_label = gt_bboxes[b].cpu().numpy(), gt_labels[b].cpu().numpy()
                scale_factor = img_metas[b]['scale_factor']
                gt_bbox_scale = gt_bbox.copy()
                gt_bbox_scale[:,:4] = gt_bbox[:,:4] / scale_factor
            for cls, r in enumerate(result):
                label = cls * np.ones_like(r[:, 0], dtype=np.uint8)
                # per-class threshold support (Exp E)
                if self.class_score_thr is not None:
                    thr = self.class_score_thr[cls]
                else:
                    thr = self.score_thr
                flag = r[:, -1] >= thr
                # print(flag)
                bboxes.append(r[flag][:, :-1])
                labels.append(label[flag])
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes[-1]) > 0:
                    overlap = rbbox_overlaps(torch.tensor(bboxes[-1]), torch.tensor(gt_bbox_scale[gt_label == cls]))
                    self.pseudo_num_tp[cls] += (torch.max(overlap,dim=1)[0] > 0.5).sum()
                self.pseudo_num_gt[cls] += (gt_label == cls).sum()
                self.pseudo_num[cls] += len(bboxes[-1])
            bboxes = np.concatenate(bboxes)
            labels = np.concatenate(labels)
            # [FIX-1] Filter degenerate pseudo bboxes (w/h too small).
            # Rotated bbox format: [cx, cy, w, h, angle].
            # log(w/anchor_w) in bbox regression targets → log(0) = -inf → NaN loss.
            if len(bboxes) > 0 and bboxes.ndim == 2 and bboxes.shape[1] >= 4:
                min_size = 1.0  # pixels
                valid = (bboxes[:, 2] > min_size) & (bboxes[:, 3] > min_size)
                # Also reject NaN/Inf coordinates
                valid &= np.isfinite(bboxes).all(axis=1)
                bboxes = bboxes[valid]
                labels = labels[valid]
            gt_bboxes_pred.append(torch.from_numpy(bboxes).float().to(device))
            gt_labels_pred.append(torch.from_numpy(labels).long().to(device))
        return gt_bboxes_pred, gt_labels_pred

    def analysis(self):
        if self.cur_iter % 500 == 0 and get_dist_info()[0] == 0:
            logger = get_root_logger()
            info = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                             in zip(self.CLASSES, self.pseudo_num, self.pseudo_num_tp)])
            info_gt = ' '.join([f'{a}' for a in self.pseudo_num_gt])
            logger.info(f'pseudo pos: {info}')
            logger.info(f'pseudo gt: {info_gt}')
            
    def show_result(self,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=4,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]
        imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            # class_names=None,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
            thickness=4,
            font_size=20,
            bbox_color=PALETTE,
            text_color=(200, 200, 200))

        if not (show or out_file):
            return img

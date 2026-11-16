import argparse
import os
import sys
import time
import random
import math
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from models import vision_transformer as vits
try:
    from models import vision_transformer2 as vits2
except ImportError:
    vits2 = None
from models.resnet_backbone import build_resnet50, set_trainable_from_grad
from models.mapper import Mapper, load_mapper, ResNet50ClipMap
from expclip.backbone import ExpCLIPBackbone
from expclip.caption_anchor import prototype_anchor_loss


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0), logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1))
    return Le


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr * (args.batch_size/128), momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * (args.batch_size/128) * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    # inductive
    best_test_acc_lab = 0
    best_test_acc_all = 0.0
    best_test_epoch = -1

    # caption text anchor state: current prototypes + per-sample GCD pseudo-labels
    text_protos = None
    pseudo_labels = {}
    if args.text_anchor_builder is not None and args.text_anchor_weight > 0:
        text_protos = args.text_anchor_builder.build_prototypes(pseudo_labels=None)
        args.logger.info(
            f'Caption text anchor: initial prototypes built '
            f'({text_protos.shape[0]} classes incl. unknown, dim={text_protos.shape[1]}).')

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            data_time = time.perf_counter() - start

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out, student_ood, pseudo_out, ood_feat, feat_norm = student(images)
                teacher_out = student_out.detach()

                # 跨分支对齐用: 保存 [2B, 256] 原始特征 (在 reshape 之前)。
                # gcd_proj_raw = x_proj (GCD 分支投影, 未归一化); ood_feat_raw = z_ood (OOD 分支, 已归一化)。
                gcd_proj_raw = student_proj
                ood_feat_raw = ood_feat

                pstr = ''
                loss = 0
                # ---------------------------------- SimGCD ----------------------------------
                # supervised GCD loss
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                loss += args.sup_weight * cls_loss
                pstr += f'cls_loss: {cls_loss.item():.4f} '

                # unsupervised GCD loss
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                loss += (1 - args.sup_weight) * cluster_loss
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                # OOD 特征空间监督对比: 已知类内聚、类间分离 (让未知类更难被吸进已知簇)
                ood_feat = torch.cat([f[mask_lab].unsqueeze(1) for f in ood_feat.chunk(2)], dim=1)
                ood_feat = torch.nn.functional.normalize(ood_feat, dim=-1)
                ood_sup_con_loss = SupConLoss()(ood_feat, labels=sup_con_labels)
                loss += args.ood_supcon_weight * ood_sup_con_loss
                pstr += f'ood_sup_con_loss: {ood_sup_con_loss.item():.4f} '

                # ---------------------------------- 跨分支特征空间对齐 (OOD z_ood <-> GCD x_proj) ----------------------------------
                # 目标: 让 OOD 分支 (unknown 判别强) 与 GCD 分支 (聚类强) 落在同一几何空间, 打通两个分支。
                # - supcon: 跨分支监督对比 (已知类同类的 z_ood 与 x_proj 互为正样本, 异类分离)
                if args.ood_gcd_align_weight > 0:
                    align_loss = 0.0
                    if args.ood_gcd_align_mode in ('none'):
                        pass
                    if args.ood_gcd_align_mode in ('supcon'):
                        # 每个已知类样本取 view1 的 x_proj 与 z_ood 作为一对表示
                        gcd_v1 = gcd_proj_raw.chunk(2)[0]   # [B, 256]
                        ood_v1 = ood_feat_raw.chunk(2)[0]   # [B, 256]
                        cross_feat = torch.stack(
                            [torch.nn.functional.normalize(gcd_v1[mask_lab], dim=-1),
                             ood_v1[mask_lab]], dim=1)       # [num_lab, 2, 256]
                        align_supcon = SupConLoss()(cross_feat, labels=sup_con_labels)
                        align_loss = align_loss + align_supcon
                        pstr += f'align_supcon: {align_supcon.item():.4f} '
                    loss += args.ood_gcd_align_weight * align_loss

                # ---------------------------------- Semantic Distribution Learning ----------------------------------
                # reference: https://github.com/VisionLearningGroup/OP_Match
                student_ood = student_ood / 0.1
                logits_ood = torch.cat([f[mask_lab] for f in student_ood.chunk(2)], dim=0)
                logits_ood_u = torch.cat([f[~mask_lab] for f in student_ood.chunk(2)], dim=0)
                logits_open_u1, logits_open_u2 = logits_ood_u.chunk(2)
                # SDL Loss for labeled samples
                sup_sdl_loss = ova_loss(logits_ood, sup_labels)
                pstr += f'sup_sdl_loss: {sup_sdl_loss.item():.4f} '

                # SDL Loss for unlabeled samples
                # entropy minimization
                L_oem = ova_ent(logits_open_u1) / 2.
                L_oem += ova_ent(logits_open_u2) / 2.
                # Soft consistency regularization
                logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
                logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
                logits_open_u1 = F.softmax(logits_open_u1, 1)
                logits_open_u2 = F.softmax(logits_open_u2, 1)
                unsup_sdl_loss = 0.1 * L_oem + torch.mean(torch.sum(torch.sum(torch.abs(logits_open_u1 - logits_open_u2) ** 2, 1), 1))
                pstr += f'unsup_sdl_loss: {unsup_sdl_loss.item():.4f} '
                loss += args.sdl_loss_weight * (sup_sdl_loss + unsup_sdl_loss)

                # distribution certainty score from OVA classifier
                ova_scores = F.softmax(student_ood.view(student_ood.size(0), 2, -1), 1).detach()
                pred_close = ova_scores[:, 1, :].data.max(1)[1]
                tmp_range = torch.arange(0, ova_scores.size(0)).long().cuda()
                unk_score = ova_scores[tmp_range, 0, pred_close]
                unk_score_unlabelled = torch.cat([f[~mask_lab] for f in unk_score.chunk(2)], dim=0)
                ood_cer_score = torch.abs(2*unk_score_unlabelled - 1)

                # ---------------------------------- Auxiliary Debiased Learning ----------------------------------
                sup_logits_pseudo = torch.cat([f[mask_lab] for f in (pseudo_out / args.pseudo_temp).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                sup_adl_loss = nn.CrossEntropyLoss()(sup_logits_pseudo, sup_labels)
                pstr += f'sup_adl_loss: {sup_adl_loss.item():.4f} '
                loss += args.adl_loss_weight * (1 - args.pl_loss_weight) * sup_adl_loss

                unsup_logits_pseudo = torch.cat([f[~mask_lab] for f in (pseudo_out / args.pseudo_temp).chunk(2)], dim=0)
                # softmax score
                unsup_logits = torch.cat([f[~mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                pseudo_label = torch.softmax(unsup_logits.detach(), dim=-1)
                # logit
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                # weighting the loss using distribution certainty score
                unsup_adl_loss = (F.cross_entropy(unsup_logits_pseudo, targets_u, reduction='none') * mask * ood_cer_score).mean()
                pstr += f'unsup_adl_loss: {unsup_adl_loss.item():.4f} '
                loss += args.adl_loss_weight * args.pl_loss_weight * unsup_adl_loss

                # ---------------------------------- Caption text prototype anchor ----------------------------------
                # Softly pull ALL cluster centers (known + unknown) toward the
                # caption-driven text prototypes.
                if args.text_anchor_weight > 0 and text_protos is not None:
                    weight_all = student[1].last_layer.weight_v
                    anchor_loss = prototype_anchor_loss(weight_all, text_protos)
                    loss += args.text_anchor_weight * anchor_loss
                    pstr += f'anchor_loss: {anchor_loss.item():.4f} '

                # Accumulate GCD pseudo-labels for unlabeled samples (view1 argmax),
                # used to progressively refresh the caption text prototypes.
                # Confidence-filtered with the same temperature & threshold as the
                # ADL pseudo-label branch, so only reliable pseudo-labels anchor text.
                if args.text_anchor_builder is not None and args.text_anchor_weight > 0:
                    with torch.no_grad():
                        logits_v1 = student_out.chunk(2)[0][~mask_lab]  # [num_unlab, C]
                        probs_v1 = logits_v1.div(0.1).softmax(dim=-1)
                        max_prob_v1, pred_v1 = probs_v1.max(dim=-1)
                        keep = max_prob_v1.ge(args.threshold)
                        uq_np = uq_idxs.cpu().numpy()
                        unlab_mask_np = (~mask_lab).cpu().numpy()
                        uq_unlab = uq_np[unlab_mask_np]
                        for u, p, ok in zip(uq_unlab, pred_v1.cpu().numpy(), keep.cpu().numpy()):
                            if ok:
                                pseudo_labels[int(u)] = int(p)

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            whole_time = time.perf_counter() - start
            start = time.perf_counter()
            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t time {:.3f} data_time {:.3f} loss {:.3f}\t {}'.format(epoch, batch_idx, len(train_loader), whole_time, data_time, loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        # Step schedule
        exp_lr_scheduler.step()
        torch.save(student.state_dict(), args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        # Refresh caption text prototypes every N epochs using accumulated pseudo-labels.
        if args.text_anchor_builder is not None and args.text_anchor_weight > 0:
            if epoch > 0 and epoch % args.text_anchor_update_freq == 0:
                text_protos = args.text_anchor_builder.build_prototypes(pseudo_labels=pseudo_labels)
                args.logger.info(
                    f'Updated caption text prototypes at epoch {epoch} '
                    f'({len(pseudo_labels)} unlabeled samples assigned pseudo-labels).')

        if epoch:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc, all_acc2, old_acc2, new_acc2, auroc_gcd, auroc_ood, ood_acc, ood_recall, ood_spec, ood_best_acc, ood_best_thr = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test, all_acc_test2, old_acc_test2, new_acc_test2, auroc_gcd_test, auroc_ood_test, ood_acc_test, ood_recall_test, ood_spec_test, ood_best_acc_test, ood_best_thr_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

            # ---------- 三个分支单独输出 ----------
            # 分支1: GCD 主聚类分支 (logits_gcd / student_out)
            args.logger.info('[Branch1 GCD     ] Train ACC: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            args.logger.info('[Branch1 GCD     ] Test  ACC: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
            args.logger.info('[Branch1 GCD     ] AUROC (main-head unknown prob): Train {:.4f} | Test {:.4f}'.format(auroc_gcd, auroc_gcd_test))

            # 分支2: OOD 分支 / 语义分布学习 (logits_ood / student_ood)
            args.logger.info('[Branch2 OOD     ] AUROC (OVA unknown score): Train {:.4f} | Test {:.4f}'.format(auroc_ood, auroc_ood_test))
            args.logger.info('[Branch2 OOD     ] Det Acc@0.5 (known/unknown): Train {:.4f} | Test {:.4f}'.format(ood_acc, ood_acc_test))
            args.logger.info('[Branch2 OOD     ] Det Recall@0.5 (unknown): Train {:.4f} | Test {:.4f}'.format(ood_recall, ood_recall_test))
            args.logger.info('[Branch2 OOD     ] Det Spec@0.5 (known): Train {:.4f} | Test {:.4f}'.format(ood_spec, ood_spec_test))
            args.logger.info('[Branch2 OOD     ] Det Best Acc (Youden thr Train {:.3f} | Test {:.3f}): Train {:.4f} | Test {:.4f}'.format(ood_best_thr, ood_best_thr_test, ood_best_acc, ood_best_acc_test))

            # 分支3: 去偏辅助分支 (logits_deb / pseudo_out)
            args.logger.info('[Branch3 Debiased] Train ACC: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc2, old_acc2, new_acc2))
            args.logger.info('[Branch3 Debiased] Test  ACC: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test2, old_acc_test2, new_acc_test2))

            if old_acc_test > best_test_acc_lab:
                args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
                args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

                torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                # inductive
                best_test_acc_lab = old_acc_test
                # transductive
                best_train_acc_lab = old_acc
                best_train_acc_ubl = new_acc
                best_train_acc_all = all_acc

                args.logger.info(f'Exp Name: {args.exp_name}')
                args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')
'''
            # 按 disjoint test 的 All 准确率选 best model (而非 Old)，且跳过 warmup 假峰
            # (warmup_teacher_temp_epochs 内 Old 会虚高, 见 epoch1 Old=0.6899 的假峰)。
            if all_acc_test > best_test_acc_all and epoch >= args.warmup_teacher_temp_epochs:
                best_test_acc_all = all_acc_test
                best_test_epoch = epoch
                args.logger.info(f'Best ACC on disjoint test set (All): {all_acc_test:.4f} @epoch {epoch} (Old {old_acc_test:.4f} | New {new_acc_test:.4f})...')
                args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

                torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                # inductive
                best_test_acc_lab = old_acc_test
                # transductive
                best_train_acc_lab = old_acc
                best_train_acc_ubl = new_acc
                best_train_acc_all = all_acc

                args.logger.info(f'Exp Name: {args.exp_name}')
                args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')
'''

def compute_auroc(scores, targets, num_labeled_classes):
    """Binary AUROC for known vs unknown discrimination.

    scores: 1-D array of 'unknown' scores (higher => more likely unknown).
    targets: remapped labels (known: 0..num_labeled_classes-1, unknown: >= num_labeled_classes).
    """
    y_true = (targets >= num_labeled_classes).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def compute_ood_detection(ood_scores, targets, num_known):
    """OOD 分支的筛选指标: 用 unknown score 对样本做 known/unknown 二分类。

    返回 (acc_05, recall_05, spec_05, best_acc, best_thr):
    - acc_05:     固定阈值 0.5 下的二分类准确率
    - recall_05:  阈值 0.5 下未知类召回率 (筛出的未知 / 全部未知)
    - spec_05:    阈值 0.5 下已知类特异率 (保留的已知 / 全部已知)
    - best_acc:   Youden's J 最优阈值下的二分类准确率
    - best_thr:   Youden's J 最优阈值
    """
    y_true = (targets >= num_known).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.5

    y_pred = (ood_scores >= 0.5).astype(int)
    acc_05 = float((y_pred == y_true).mean())
    recall_05 = float(y_pred[y_true == 1].mean())
    spec_05 = float((1 - y_pred)[y_true == 0].mean())

    fpr, tpr, thr = roc_curve(y_true, ood_scores)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    best_thr = float(thr[idx])
    best_acc = float(((ood_scores >= best_thr).astype(int) == y_true).mean())

    return acc_05, recall_05, spec_05, best_acc, best_thr


def test(model, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    preds_pseudo = []
    gcd_probs, ood_scores = [], []
    mask = np.array([])
    num_known = len(args.train_classes)
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            x_proj, logits, logits_ood, logits_pseudo, _, _ = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            preds_pseudo.append(logits_pseudo.argmax(1).cpu().numpy())
            gcd_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            # OVA (one-vs-all) unknown score from OOD head: prob of 'not belong' for the most likely known class
            ova = torch.softmax(logits_ood.view(logits_ood.size(0), 2, -1), dim=1)
            pred_close = ova[:, 1, :].argmax(dim=1)
            unk = ova[torch.arange(ova.size(0)), 0, pred_close].cpu().numpy()
            ood_scores.append(unk)
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(num_known) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    preds_pseudo = np.concatenate(preds_pseudo)
    gcd_probs = np.concatenate(gcd_probs, axis=0)
    ood_scores = np.concatenate(ood_scores)

    # AUROC: main-head unknown score = sum of softmax prob over unknown class dims
    unk_score_gcd = gcd_probs[:, num_known:].sum(axis=1)
    auroc_gcd = compute_auroc(unk_score_gcd, targets, num_known)
    auroc_ood = compute_auroc(ood_scores, targets, num_known)

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=f'{save_name} [GCD]', args=args)
    all_acc2, old_acc2, new_acc2 = log_accs_from_preds(y_true=targets, y_pred=preds_pseudo, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=f'{save_name} [Debiased]', args=args)

    # OOD 分支筛选指标: 用 unknown score 二分类 known/unknown 的准确率等
    ood_acc, ood_recall, ood_spec, ood_best_acc, ood_best_thr = compute_ood_detection(ood_scores, targets, num_known)

    return all_acc, old_acc, new_acc, all_acc2, old_acc2, new_acc2, auroc_gcd, auroc_ood, ood_acc, ood_recall, ood_spec, ood_best_acc, ood_best_thr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2b'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='rafdb', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19, rafdb')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--unknown_classes', nargs='+', type=int, default=None,
                        help='Class ids treated as unknown (open-set), e.g. "--unknown_classes 4 5 6". '
                             'Only used for datasets without a predefined split (rafdb); the remaining classes become known.')

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='simgcd', type=str)
    parser.add_argument('--class_num', default=0, type=int)
    parser.add_argument('--dino', type=str, default='v1')
    parser.add_argument('--backbone', type=str, default='vit', choices=['vit', 'resnet50', 'expclip', 'resnet50_clipmap'],
                        help='Backbone for DebGCD: vit (DINO ViT-B), resnet50, expclip (frozen CLIP + Exp-CLIP head), or resnet50_clipmap (resnet50 + frozen 2048->4096 mapper).')

    parser.add_argument('--mapper_path', type=str,
                        default='dev_outputs/mapper_resnet50_to_expclip.pth',
                        help='Path to the trained resnet50->expclip mapper (used with --backbone resnet50_clipmap).')
    # Exp-CLIP backbone options
    parser.add_argument('--clip_name', type=str, default='ViT-B/16',
                        help='CLIP model name used by the expclip backbone.')
    parser.add_argument('--expclip_head_path', type=str,
                        default='expclip/checkpoint/ExpCLIP_B16_model.pth',
                        help='Path to the Exp-CLIP projection head checkpoint.')
    parser.add_argument('--text_prototypes', action='store_true', default=False,
                        help='Initialize known-class cluster centers with CLIP text prototypes.')
    parser.add_argument('--text_prompt_style', type=str, default='single', choices=['single', 'llm'],
                        help='Text prototype style: "single" (one template prompt per class) or "llm" (rich LLM-style facial-action descriptions).')
    parser.add_argument('--proto_mapper_path', type=str,
                        default='dev_outputs/mapper_expclip_to_resnet50.pth',
                        help='Path to the Exp-CLIP -> ResNet50 mapper used to map CLIP text prototypes into the ResNet50 space.')
    # Dynamic caption-driven text prototypes + prototype anchor loss (Qwen2-VL captions)
    parser.add_argument('--caption_path', type=str,
                        default='dev_outputs/rafdb_desc.jsonl',
                        help='Path to the label-agnostic image description jsonl ({"img","desc"}). '
                             'Descriptions are split-independent; the "This is <slot>." slot is filled dynamically.')
    parser.add_argument('--text_anchor_weight', type=float, default=0.0,
                        help='Weight of the prototype anchor loss (soft-pulls known-class centers toward caption text prototypes). 0 disables the loss (ablation).')
    parser.add_argument('--text_anchor_update_freq', type=int, default=10,
                        help='Recompute caption text prototypes every N epochs using the latest GCD pseudo-labels (progressive label injection).')
    parser.add_argument('--text_anchor_init', action='store_true', default=False,
                        help='Initialize known-class cluster centers with caption text prototypes (labeled samples only) before training.')
    parser.add_argument('--seed', default=0, type=int)

    # auxiliary debiased classifier
    parser.add_argument('--adl_loss_weight', type=float, default=1.0)
    parser.add_argument('--pl_loss_weight', type=float, default=0.5)
    parser.add_argument('--threshold', default=0.95, type=float, help='debiasing threshold')
    parser.add_argument('--pseudo_temp', default=0.1, type=float)
    parser.add_argument('--save_all', action='store_true', default=False)

    # distribution detector
    parser.add_argument('--sdl_loss_weight', type=float, default=0.01)
    parser.add_argument('--ood_supcon_weight', type=float, default=0.5,
                        help='Weight for the supervised contrastive loss on the OOD feature space.')
    parser.add_argument('--ood_gcd_align_weight', type=float, default=0.5,
                        help='Weight for cross-branch feature-space alignment between OOD (z_ood) and GCD (x_proj) branches. Set 0 to disable.')
    parser.add_argument('--ood_gcd_align_mode', type=str, default='supcon', choices=['none', 'supcon'],
                        help='Alignment loss type: mse (per-sample cosine), supcon (cross-branch supervised contrast on known classes), both.')
    parser.add_argument('--num_ood_layers', default=5, type=int)

    # evaluation
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str, default='debgcd_models/dinov1/scars/model.pt')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON config file. Values inside act as defaults; explicit CLI args override them.')

    # ----------------------
    # INIT
    # ----------------------
    # 预解析 config 路径: 用 JSON 里的值作为默认值, 命令行显式参数仍可覆盖
    _pre_args, _ = parser.parse_known_args()
    if _pre_args.config is not None:
        with open(_pre_args.config, 'r') as _f:
            _cfg = json.load(_f)
        parser.set_defaults(**_cfg)

    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    if not args.class_num:
        args.num_unlabeled_classes = len(args.unlabeled_classes)
    else:
        args.num_unlabeled_classes = args.class_num - args.num_labeled_classes

    init_experiment(args, runner_name=[f'DebGCD_{args.dataset_name}'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # SET SEED
    # ----------------------
    set_random_seed(args.seed)

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    if args.backbone == 'vit':
        # DINO version
        if args.dino == 'v1':
            backbone = vits.__dict__['vit_base']()
        elif args.dino == 'v2':
            backbone = vits2.__dict__['vit_base']()
            args.warmup_model_dir = args.warmup_model_dir.replace('dino_vitb16', 'dinov2_vitb14_reg4')
        else:
            raise AttributeError('Unsupported DINO version')
        args.feat_dim = 768
    elif args.backbone == 'resnet50':
        backbone = build_resnet50(pretrained=(args.warmup_model_dir is None))
        args.feat_dim = 2048
    elif args.backbone == 'resnet50_clipmap':
        resnet = build_resnet50(pretrained=(args.warmup_model_dir is None))
        if args.warmup_model_dir is not None:
            args.logger.info(f'Loading ResNet50 weights from {args.warmup_model_dir}')
            resnet.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
        mapper = load_mapper(args.mapper_path, device='cpu')
        args.logger.info(f'Loaded resnet50->expclip mapper from {args.mapper_path} (frozen)')
        backbone = ResNet50ClipMap(resnet, mapper)
        args.feat_dim = 4096
    elif args.backbone == 'expclip':
        backbone = ExpCLIPBackbone(clip_name=args.clip_name, head_path=args.expclip_head_path)
        args.feat_dim = backbone.feat_dim
    else:
        raise AttributeError('Unsupported backbone')

    if args.backbone not in ('expclip', 'resnet50_clipmap') and args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: image size fixed to 224
    args.image_size = 224
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    if args.backbone == 'expclip':
        # CLIP + projection head stay frozen; only the DebGCD head is trained.
        for p in backbone.parameters():
            p.requires_grad = False
    else:
        set_trainable_from_grad(backbone, args.backbone, args.grad_from_block)

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    # expclip expects CLIP normalization, not ImageNet's.
    if args.backbone == 'expclip' and args.transform == 'imagenet':
        args.logger.info('expclip backbone: switching transform from "imagenet" to "clip"')
        args.transform = 'clip'

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    from model import DebGCDHead
    projector = DebGCDHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim , ood_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers , noodlayers=args.num_ood_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TEXT PROTOTYPES (LLM-style descriptions -> known-class cluster centers)
    # ----------------------
    if args.text_prototypes:
        from expclip.text_prototypes import build_text_prototypes, inject_text_prototypes
        known_classes = list(args.train_classes)

        if hasattr(backbone, 'encode_text'):
            # expclip backbone: its own text encoder directly gives feat_dim-dim prototypes.
            text_protos = build_text_prototypes(backbone, known_classes, style=args.text_prompt_style)
        else:
            # resnet50 / resnet50_clipmap: build prototypes in the CLIP text space,
            # then (if dims mismatch) pull them into the backbone feature dim via a
            # frozen Exp-CLIP -> ResNet50 mapper. Clustering stays in ResNet50 space.
            from expclip.backbone import ExpCLIPBackbone
            text_backbone = ExpCLIPBackbone(clip_name=args.clip_name, head_path=args.expclip_head_path,
                                            device=device)
            text_protos = build_text_prototypes(text_backbone, known_classes, style=args.text_prompt_style)  # [K, 4096]
            if text_protos.shape[1] != args.feat_dim:
                from models.mapper import load_mapper
                proto_mapper = load_mapper(args.proto_mapper_path, device=device)
                proto_mapper.eval()
                with torch.no_grad():
                    text_protos = proto_mapper(text_protos)  # [K, feat_dim], L2-normalized
                args.logger.info(
                    f'Mapped text prototypes 4096 -> {args.feat_dim} via {args.proto_mapper_path}')

        inject_text_prototypes(projector, text_protos)
        args.logger.info(
            f'Injected {text_protos.shape[0]} text prototypes (style={args.text_prompt_style}, '
            f'dim={text_protos.shape[1]}) into the GCD clustering head.')

    # ----------------------
    # DYNAMIC CAPTION TEXT PROTOTYPES (Qwen2-VL captions -> prototype anchor loss)
    # ----------------------
    args.text_anchor_builder = None
    if args.text_anchor_weight > 0 or args.text_anchor_init:
        from expclip.caption_anchor import (
            CaptionTextPrototypeBuilder, load_desc, build_uqidx_to_desc_label,
        )

        # Text encoder: expclip already has one; otherwise build a frozen CLIP text
        # encoder and (if the dims differ) pull features into the backbone space.
        if hasattr(backbone, 'encode_text'):
            text_backbone = backbone
            proto_mapper = None
        else:
            from expclip.backbone import ExpCLIPBackbone
            text_backbone = ExpCLIPBackbone(clip_name=args.clip_name,
                                            head_path=args.expclip_head_path,
                                            device=device)
            proto_mapper = None
            if text_backbone.feat_dim != args.feat_dim:
                from models.mapper import load_mapper
                proto_mapper = load_mapper(args.proto_mapper_path, device=device)
                proto_mapper.eval()

        descs = load_desc(args.caption_path)
        from config import rafdb_root
        train_label_file = os.path.join(rafdb_root, 'RAF-DB_train', 'EmoLabel', 'train_label.txt')
        # Keep ground-truth labels available to text prototypes only for the
        # samples designated as labeled; unlabeled samples require GCD pseudo-labels.
        labelled_uq = set(int(u) for u in train_dataset.labelled_dataset.uq_idxs)
        uq_to_desc, uq_to_label, num_train_samples = build_uqidx_to_desc_label(
            train_label_file, descs, labelled_uq)
        args.logger.info(
            f'Loaded {len(descs)} image descriptions; {len(uq_to_desc)} matched to '
            f'training uq_idxs (num_train_samples={num_train_samples}).')

        # Global uq_idx of labeled samples (real labels, known classes only).
        args.text_anchor_builder = CaptionTextPrototypeBuilder(
            text_backbone=text_backbone,
            mapper=proto_mapper,
            known_classes=list(args.train_classes),
            unlabeled_classes=list(args.unlabeled_classes),
            labelled_uq_idxs=labelled_uq,
            uq_to_desc=uq_to_desc,
            uq_to_label=uq_to_label,
            feat_dim=args.feat_dim,
            device=device,
        )

        if args.text_anchor_init:
            from expclip.text_prototypes import inject_text_prototypes as _inject
            init_protos = args.text_anchor_builder.build_prototypes(pseudo_labels=None)
            _inject(projector, init_protos)
            args.logger.info(
                f'Initialized known-class centers with caption text prototypes '
                f'({init_protos.shape[0]} classes, dim={init_protos.shape[1]}).')

    # ----------------------
    # TRAIN
    # ----------------------
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_path, map_location='cpu'))
        model = model.to(device)
        all_acc, old_acc, new_acc, _, _, _, auroc_gcd, auroc_ood, *_ = test(model, test_loader_unlabelled, epoch=0, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        args.logger.info('Train AUROC: Main {:.4f} | OOD {:.4f}'.format(auroc_gcd, auroc_ood))
    else:
        train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
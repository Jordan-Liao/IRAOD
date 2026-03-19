#!/usr/bin/env python3
"""P0032: SARCLIP LoRA fine-tuning + P0033: L_ent entropy minimization.
Supports single-GPU and multi-GPU (DDP) training.

Usage (single GPU):
    python -m lora_finetune.lora_sarclip_train ...

Usage (multi-GPU):
    python -m torch.distributed.launch --nproc_per_node=5 \
        -m lora_finetune.lora_sarclip_train ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path (needed when launched via torchrun)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T


def _ensure_sarclip(sarclip_dir: str):
    d = Path(sarclip_dir).resolve()
    if (d / 'sar_clip').is_dir():
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    else:
        raise FileNotFoundError(f'SARCLIP not found at {d}')


class RSARPatchDataset(Dataset):
    """Load RSAR training images with DOTA-format annotations,
    crop object patches for contrastive LoRA fine-tuning."""

    def __init__(self, img_dir, ann_dir, classes, transform=None,
                 max_patches_per_image=50, return_score=False):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.classes = list(classes)
        self.cls2id = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.max_patches = max_patches_per_image
        self.return_score = return_score

        self.patches = []
        for ann_file in sorted(self.ann_dir.glob('*.txt')):
            img_name = ann_file.stem + '.png'
            img_path = self.img_dir / img_name
            if not img_path.exists():
                for ext in ['.jpg', '.jpeg', '.tif', '.bmp']:
                    alt = self.img_dir / (ann_file.stem + ext)
                    if alt.exists():
                        img_path = alt
                        break
            if not img_path.exists():
                continue

            with open(ann_file) as f:
                lines = f.readlines()

            count = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(x) for x in parts[:8]]
                    cls_name = parts[8]
                    difficulty = int(parts[9]) if len(parts) > 9 else 0
                except (ValueError, IndexError):
                    continue
                if cls_name not in self.cls2id:
                    continue
                xs = coords[0::2]
                ys = coords[1::2]
                x1, y1 = max(0, min(xs)), max(0, min(ys))
                x2, y2 = max(xs), max(ys)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                score = 0.9 if difficulty == 0 else 0.3
                self.patches.append((str(img_path), (x1, y1, x2, y2),
                                     self.cls2id[cls_name], score))
                count += 1
                if count >= self.max_patches:
                    break

        if _is_main_process():
            print(f'[RSARPatchDataset] {len(self.patches)} patches from {img_dir}')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, (x1, y1, x2, y2), label, score = self.patches[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = x2 - x1, y2 - y1
        ex = 0.2
        x1 = max(0, x1 - w * ex)
        y1 = max(0, y1 - h * ex)
        x2 = x2 + w * ex
        y2 = y2 + h * ex
        patch = img.crop((int(x1), int(y1), int(x2), int(y2)))
        if self.transform:
            patch = self.transform(patch)
        if self.return_score:
            return patch, label, score
        return patch, label


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """H(p) = -sum(p * log(p))"""
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    ent = -(p * log_p).sum(dim=-1)
    return ent


def _is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _setup_distributed():
    """Initialize distributed training if launched with torch.distributed.launch."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    elif 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return local_rank, rank, world_size
    else:
        return 0, 0, 1


def train(args):
    local_rank, rank, world_size = _setup_distributed()
    distributed = world_size > 1

    _ensure_sarclip(args.sarclip_dir)
    import sar_clip

    device = torch.device(f'cuda:{local_rank}')

    # Load SARCLIP model
    model = sar_clip.create_model_with_args(
        args.sarclip_model,
        pretrained=args.sarclip_pretrained,
        precision='fp32',
        device=str(device),
        cache_dir=None,
        output_dict=True,
    )

    # Inject LoRA
    from tools.lora_utils import LoraConfig, inject_lora, lora_state_dict

    def vision_filter(name, _m):
        return name.startswith('visual.')

    cfg = LoraConfig(r=args.lora_r, alpha=args.lora_alpha, dropout=args.dropout)
    n_replaced = inject_lora(model, config=cfg, module_filter=vision_filter)
    if _is_main_process():
        print(f'[LoRA] Injected into {n_replaced} linear layers')

    # Freeze all except LoRA params
    for name, p in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if _is_main_process():
        print(f'[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

    # Build text classifier
    tokenizer = sar_clip.get_tokenizer(args.sarclip_model, cache_dir=None)
    templates = [
        'an SAR image of a {}',
        'this SAR patch shows a {}',
        'a satellite SAR view of a {}',
    ]
    with torch.no_grad():
        all_text_feats = []
        for t in templates:
            prompts = [t.format(c) for c in args.classes]
            tokens = tokenizer(prompts).to(device)
            out = model(text=tokens)
            tf = out['text_features'] if isinstance(out, dict) else out[1]
            tf = tf / tf.norm(dim=-1, keepdim=True)
            all_text_feats.append(tf)
        text_features = torch.stack(all_text_feats).mean(0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.detach()  # [C, D]

    # Wrap model in DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # Dataset
    use_ent = args.ent_weight > 0
    transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = RSARPatchDataset(args.data_root, args.ann_file, args.classes,
                               transform, return_score=use_ent)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                        num_workers=4, pin_memory=True, drop_last=True,
                        sampler=sampler)

    # Optimizer
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader))

    model.train()
    os.makedirs(args.output_dir, exist_ok=True)
    tau = 100.0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss_val = 0
        total_ce_val = 0
        total_ent_val = 0
        n_batches = 0

        for batch in loader:
            if use_ent:
                images, labels, scores = batch
                scores = scores.float().to(device)
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            # Forward through visual encoder
            out = model(image=images)
            img_feats = out['image_features'] if isinstance(out, dict) else out[0]
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # Contrastive logits: [B, C]
            logits = tau * img_feats @ text_features.T

            # Cross-entropy loss
            loss_ce = F.cross_entropy(logits, labels)

            # P0033: L_ent only on low-confidence samples
            loss_ent_val = 0.0
            if use_ent:
                ent = entropy_loss(logits)  # [B]
                low_conf_mask = (scores < args.ent_score_thr).float()
                n_low = low_conf_mask.sum()
                if n_low > 0:
                    loss_ent = (ent * low_conf_mask).sum() / n_low
                    loss_ent_val = loss_ent.item()
                else:
                    loss_ent = torch.tensor(0.0, device=device)
                loss = loss_ce + args.ent_weight * loss_ent
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss_val += loss.item()
            total_ce_val += loss_ce.item()
            total_ent_val += loss_ent_val
            n_batches += 1

        avg_loss = total_loss_val / max(n_batches, 1)
        avg_ce = total_ce_val / max(n_batches, 1)
        avg_ent = total_ent_val / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]['lr']
        if _is_main_process():
            print(f'Epoch [{epoch+1}/{args.epochs}] loss={avg_loss:.4f} ce={avg_ce:.4f} '
                  f'ent={avg_ent:.4f} lr={lr_now:.6f}')

        # Save checkpoint (rank 0 only)
        if _is_main_process():
            raw_model = model.module if distributed else model
            ckpt = {
                'state_dict': lora_state_dict(raw_model),
                'meta': {
                    'r': args.lora_r,
                    'alpha': args.lora_alpha,
                    'dropout': args.dropout,
                    'target': 'vision',
                    'epoch': epoch + 1,
                    'sarclip_model': args.sarclip_model,
                    'classes': list(args.classes),
                    'ent_weight': args.ent_weight,
                    'ent_score_thr': args.ent_score_thr,
                },
            }
            save_path = os.path.join(args.output_dir, f'lora_epoch_{epoch+1}.pth')
            torch.save(ckpt, save_path)

        if distributed:
            dist.barrier()

    if _is_main_process():
        final_path = os.path.join(args.output_dir, 'lora_final.pth')
        torch.save(ckpt, final_path)
        print(f'[LoRA] Training done. Final checkpoint: {final_path}')

    if distributed:
        dist.destroy_process_group()
    return 0


def main():
    parser = argparse.ArgumentParser(description='SARCLIP LoRA fine-tuning (P0032+P0033)')
    parser.add_argument('--sarclip-dir', default='third_party/SARCLIP')
    parser.add_argument('--sarclip-model', default='RN50')
    parser.add_argument('--sarclip-pretrained', default=None)
    parser.add_argument('--data-root', required=True, help='Image directory')
    parser.add_argument('--ann-file', required=True, help='Annotation directory (DOTA format)')
    parser.add_argument('--classes', nargs='+',
                        default=['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--lora-r', '--r', dest='lora_r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', '--alpha', dest='lora_alpha', type=float,
                        default=16.0, help='LoRA alpha')
    parser.add_argument('--dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--ent-weight', type=float, default=0.0,
                        help='P0033: entropy loss weight (0=disabled)')
    parser.add_argument('--ent-score-thr', type=float, default=0.5,
                        help='P0033: only apply L_ent to samples with score < thr')
    parser.add_argument('--output-dir', default='work_dirs/sarclip_lora')
    # DDP args (auto-set by torch.distributed.launch)
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    return train(args)


if __name__ == '__main__':
    raise SystemExit(main())

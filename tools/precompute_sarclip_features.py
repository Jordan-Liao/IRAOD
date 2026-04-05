#!/usr/bin/env python3
"""Pre-compute SARCLIP ViT-L-14 spatial features for all RSAR images."""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party/SARCLIP"))

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--corrupts", nargs="+", default=["clean"])
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()

@torch.no_grad()
def extract_patch_features(vit, images_tensor):
    """Extract 16x16 patch token grid BEFORE global pooling. Returns [B, 16, 16, output_dim]."""
    x = vit.conv1(images_tensor)
    grid_h, grid_w = x.shape[2], x.shape[3]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + vit.positional_embedding.to(x.dtype)
    x = vit.patch_dropout(x)
    x = vit.ln_pre(x)
    x = vit.transformer(x)
    x = vit.ln_post(x)
    patch_tokens = x[:, 1:, :]  # remove CLS
    if vit.proj is not None:
        patch_tokens = patch_tokens @ vit.proj
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    return patch_tokens.reshape(-1, grid_h, grid_w, patch_tokens.shape[-1])

def main():
    args = parse_args()
    device = torch.device("cuda")

    import sar_clip
    model = sar_clip.create_model_with_args(
        "ViT-L-14",
        pretrained=os.environ.get("SARCLIP_PRETRAINED", "weights/sarclip/ViT-L-14/vit_l_14_model.safetensors"),
        precision="amp", device=str(device), output_dict=True,
    )
    model.eval()

    # LoRA
    lora_path = os.environ.get("SARCLIP_LORA", "").strip()
    if lora_path and os.path.exists(lora_path):
        from tools.lora_utils import LoraConfig, inject_lora
        cfg = LoraConfig(r=4, alpha=4); inject_lora(model, config=cfg, module_filter=lambda n, m: "visual" in n and isinstance(m, torch.nn.Linear))
        state = torch.load(lora_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[LoRA] Loaded {lora_path}")

    # Preprocess: simple resize+normalize (same as CLIP)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    print(f"Pre-computing features → {args.out_dir}")
    total = 0
    start = time.time()

    for corrupt in args.corrupts:
        for split in args.splits:
            img_dir = os.path.join(args.data_root, split, "images" if corrupt == "clean" else f"images-{corrupt}")
            if not os.path.isdir(img_dir):
                print(f"  SKIP {corrupt}/{split}")
                continue

            out_dir = os.path.join(args.out_dir, corrupt, split)
            os.makedirs(out_dir, exist_ok=True)

            files = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in EXTS])
            existing = len([f for f in os.listdir(out_dir) if f.endswith(".npz")])
            if existing >= len(files):
                print(f"  SKIP {corrupt}/{split}: {existing} already done")
                continue

            print(f"  {corrupt}/{split}: {len(files)} images")
            batch_t, batch_n = [], []

            for i, fname in enumerate(files):
                out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".npz")
                if os.path.exists(out_path):
                    continue
                try:
                    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
                    batch_t.append(preprocess(img))
                    batch_n.append(out_path)
                except:
                    continue

                if len(batch_t) >= args.batch_size:
                    t = torch.stack(batch_t).to(device)
                    with torch.cuda.amp.autocast():
                        feats = extract_patch_features(model.visual, t)
                    feats = feats.cpu().numpy().astype(np.float16)
                    for j, p in enumerate(batch_n):
                        np.savez_compressed(p, features=feats[j])
                    total += len(batch_t)
                    batch_t.clear(); batch_n.clear()
                    if (i+1) % 5000 == 0:
                        print(f"    [{i+1}/{len(files)}] {total/(time.time()-start):.0f} img/s")

            if batch_t:
                t = torch.stack(batch_t).to(device)
                with torch.cuda.amp.autocast():
                    feats = extract_patch_features(model.visual, t)
                feats = feats.cpu().numpy().astype(np.float16)
                for j, p in enumerate(batch_n):
                    np.savez_compressed(p, features=feats[j])
                total += len(batch_t)

    elapsed = time.time() - start
    print(f"\nDone: {total} images, {elapsed/60:.1f} min, {total/max(elapsed,1):.0f} img/s")

if __name__ == "__main__":
    main()

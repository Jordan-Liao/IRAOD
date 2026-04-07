# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image


# from torchvision import transforms as trans
# import os
#
#
# CLASSES = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
# save_img = False
#
# def obb2xyxy(rbboxes):
#     w = rbboxes[:, 2::5]
#     h = rbboxes[:, 3::5]
#     a = rbboxes[:, 4::5]
#     cosa = np.abs(np.cos(a))
#     sina = np.abs(np.sin(a))
#     hbbox_w = cosa * w + sina * h
#     hbbox_h = sina * w + cosa * h
#     dx = rbboxes[..., 0]
#     dy = rbboxes[..., 1]
#     dw = hbbox_w.reshape(-1)
#     dh = hbbox_h.reshape(-1)
#     x1 = dx - dw / 2
#     y1 = dy - dh / 2
#     x2 = dx + dw / 2
#     y2 = dy + dh / 2
#     xyxy_array = np.stack((x1, y1, x2, y2), -1)
#
#     return xyxy_array
#
#
# class CGA:
#     # def __init__(self, class_names, model='RN50x64', templates = 'an aerial image of a {}'):
#     def __init__(self, class_names, model='RN50x64', templates='an SAR image of a {}'):
#         super().__init__()
#         self.save_path = '_clip_img'
#         self.device = torch.cuda.current_device()
#
#         self.expand_ratio = 0.4
#
#         # CLIP configs
#         import clip
#         self.class_names = class_names
#
#         self.clip, self.preprocess = clip.load(model, device=self.device)
#         self.prompts = clip.tokenize([
#             templates.format(cls_name)
#             for cls_name in self.class_names
#         ]).to(self.device)
#         with torch.no_grad():
#             self.text_features = self.clip.encode_text(self.prompts)
#             self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
#
#         print("[CGA] init classes =", self.class_names)#print
#
#     def load_image_by_box(self, img_path, boxes, scores, labels):
#         image = Image.open(img_path).convert("RGB")
#         image_list = []
#         probs_list = []
#         ori_image_list = []
#         for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
#             x1, y1, x2, y2 = box
#             h, w = y2 - y1, x2 - x1
#             x1 = max(0, x1 - w * self.expand_ratio)
#             y1 = max(0, y1 - h * self.expand_ratio)
#             x2 = x2 + w * self.expand_ratio
#             y2 = y2 + h * self.expand_ratio
#             sub_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
#             if save_img:
#                 label_ = CLASSES[label]
#                 sub_image.save(os.path.join(self.save_path, f"sub_image_{i}_{score}_{label_}.jpg"))
#
#             ori_image_list.append(sub_image)
#             sub_image = self.preprocess(sub_image).to(self.device)
#             image_list.append(sub_image)
#         return torch.stack(image_list), ori_image_list
#
#     @torch.no_grad()
#     def __call__(self, img_path, boxes, scores, labels):
#         images,ori_image_list = self.load_image_by_box(img_path, boxes, scores, labels)
#         with torch.no_grad():
#             image_features = self.clip.encode_image(images)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#
#         logits_per_image = (100*image_features @ self.text_features.T).softmax(dim=-1).cpu().numpy()
#         return logits_per_image, ori_image_list
#
# class TestMixins:
#     def __init__(self):
#         self.cga = None
#
#     def refine_test(self, results, img_metas):
#         print("[CGA] refine_test on", img_metas[0]['filename'])#print
#
#         if not hasattr(self, 'cga'):
#
#             self.cga= CGA(CLASSES, model='RN50x64')
#             self.exclude_ids = [7,8,11]
#
#         boxes_list, scores_list, labels_list = [], [], []
#         for cls_id, result in enumerate(results[0]):
#             if len(result) == 0:
#                 continue
#
#             result_ = obb2xyxy(result)
#
#             boxes_list.append(result_[:, :4])
#             scores_list.append(result[:, -1])
#
#             labels_list.append([cls_id] * len(result))
#         if len(boxes_list) == 0:
#             return results
#
#         boxes_list = np.concatenate(boxes_list, axis=0)
#
#         scores_list = np.concatenate(scores_list, axis=0)
#         labels_list = np.concatenate(labels_list, axis=0)
#
#         logits, images = self.cga(img_metas[0]['filename'], boxes_list, scores_list, labels_list)
#
#         for i, prob in enumerate(logits):
#
#             if labels_list[i] != np.argmax(prob):
#                 if labels_list[i] not in self.exclude_ids:
#                     scores_list[i] = scores_list[i] * 0.7 + prob[labels_list[i]] * 0.3
#             else:
#                 pass
#         j = 0
#         for i in range(len(results[0])):
#             num_dets = len(results[0][i])
#             if num_dets == 0:
#                 continue
#             for k in range(num_dets):
#                 results[0][i][k, -1] = scores_list[j]
#                 j += 1
#
#         return results
#

import os

# --- Pre-computed feature support ---
_PRECOMPUTED_FEATURES_DIR = os.environ.get("CGA_PRECOMPUTED_DIR", "").strip()
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import sys

save_img = False


def _parse_templates(value: str) -> tuple[str, ...]:
    parts = [p.strip() for p in value.split("|")]
    return tuple([p for p in parts if p])


def _templates_from_env(*, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get("CGA_TEMPLATES", "").strip()
    if raw:
        return _parse_templates(raw)
    return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _str_env(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default


def obb2xyxy(rbboxes):
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = np.abs(np.cos(a))
    sina = np.abs(np.sin(a))
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    xyxy_array = np.stack((x1, y1, x2, y2), -1)
    return xyxy_array


class CGA:
    """
    用 SARCLIP 做“类别引导的重打分”：
    - 文本侧：用 SARCLIP 的 tokenizer + 文本编码器得到 text_features（归一化）
    - 图像侧：对每个候选框裁剪成 patch，预处理后喂给 SARCLIP 得到 image_features（归一化）
    - 打分：logits = softmax( tau * image_features @ text_features.T )
    """
    def __init__(
        self,
        class_names,
        model="RN50",
        pretrained: str | None = None,
        precision="amp",
        templates=("an SAR image of a {}", "this SAR patch shows a {}"),
        tau=100.0,
        expand_ratio=0.4,
        force_grayscale=False,
    ):
        super().__init__()
        # Lazily add SARCLIP checkout to PYTHONPATH if present.
        repo_root = Path(__file__).resolve().parents[1]
        sarclip_dir = Path(os.environ.get("SARCLIP_DIR", repo_root / "third_party" / "SARCLIP"))
        if "sar_clip" not in sys.modules:
            try:
                import sar_clip  # type: ignore
            except Exception:
                if (sarclip_dir / "sar_clip").is_dir():
                    sys.path.insert(0, str(sarclip_dir))
                import sar_clip  # type: ignore
        else:
            import sar_clip  # type: ignore

        if pretrained is None:
            pretrained = os.environ.get("SARCLIP_PRETRAINED", "").strip() or None
        if pretrained is None:
            # Best-effort default under repo-local weights/
            model_name_norm = str(model).replace("/", "-")
            default_dir = repo_root / "weights" / "sarclip" / model_name_norm
            base = {
                "RN50": "rn50_model",
                "RN101": "rn101_model",
                "ViT-B-16": "vit_b_16_model",
                "ViT-B-32": "vit_b_32_model",
                "ViT-L-14": "vit_l_14_model",
            }.get(model_name_norm)
            if base is not None:
                can_load_safetensors = hasattr(torch, "frombuffer")
                if can_load_safetensors:
                    try:
                        import safetensors.torch  # type: ignore  # noqa: F401
                    except Exception:
                        can_load_safetensors = False

                exts = [".safetensors", ".pth", ".pt"] if can_load_safetensors else [".pth", ".pt", ".safetensors"]
                for ext in exts:
                    cand = default_dir / f"{base}{ext}"
                    if cand.is_file():
                        pretrained = str(cand)
                        break

        if pretrained is not None and not Path(pretrained).is_file():
            print(f"[CGA/SARCLIP] WARNING: pretrained not found, fallback to random init: {pretrained}")
            pretrained = None

        self.class_names = list(class_names)
        self.device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
        self.save_path = '_clip_img'
        self.expand_ratio = float(expand_ratio)
        self.tau = float(tau)
        self.force_grayscale = bool(force_grayscale)

        # 1) 构建 SARCLIP 模型
        self.clip = sar_clip.create_model_with_args(
            model,
            pretrained=pretrained,
            precision=precision,          # 'amp' 在 torch1.7 下OK
            device=str(self.device),
            cache_dir=None,
            output_dict=True              # forward 返回 dict，包含 'image_features'
        )
        self.clip.eval()

        # Optional: load LoRA adapter on top of SARCLIP (for interference-robust scoring).
        lora_path = os.environ.get("SARCLIP_LORA", "").strip() or None
        if lora_path:
            p = Path(lora_path).expanduser()
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            if not p.is_file():
                print(f"[CGA/SARCLIP] WARNING: SARCLIP_LORA not found: {p} (ignored)")
            else:
                try:
                    from tools.lora_utils import LoraConfig, inject_lora, load_lora_state_dict

                    ckpt = torch.load(str(p), map_location="cpu")
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}
                        state = ckpt["state_dict"]
                    else:
                        meta = {}
                        state = ckpt

                    r = int(meta.get("r", 8))
                    alpha = float(meta.get("alpha", 16.0))
                    dropout = float(meta.get("dropout", 0.0))
                    target = str(meta.get("target", "vision")).strip().lower()

                    def _filter(full_name: str, _m: torch.nn.Linear) -> bool:
                        if target in ("vision", "visual"):
                            return full_name.startswith("visual.")
                        if target in ("vision+text", "both", "all", "visual+text"):
                            return full_name.startswith("visual.") or full_name.startswith("transformer.") or full_name.startswith("text.")
                        return True

                    replaced = inject_lora(self.clip, config=LoraConfig(r=r, alpha=alpha, dropout=dropout), module_filter=_filter)
                    load_lora_state_dict(self.clip, state)
                    self.clip.eval()
                    print(f"[CGA/SARCLIP] LoRA loaded: {p} target={target} replaced_linears={replaced}")
                except Exception as e:
                    print(f"[CGA/SARCLIP] WARNING: failed to load LoRA ({p}): {e}")

        # 2) tokenizer & zero-shot 文本分类器（把多个模板的文本特征做平均&归一化）
        self.tokenizer = sar_clip.get_tokenizer(model, cache_dir=None)

        # build_zero_shot_classifier 返回 [embed_dim, num_classes] 的张量（已归一化）
        self.classifier = sar_clip.build_zero_shot_classifier(
            self.clip,
            tokenizer=self.tokenizer,
            classnames=self.class_names,
            templates=[lambda c, t=t: t.format(c) for t in templates],
            num_classes_per_batch=None,
            device=self.device,
            use_tqdm=False,
        )  # shape: [D, C]
        # 保险起见再归一化一次
        self.classifier = self.classifier / self.classifier.norm(dim=0, keepdim=True)

        # 3) 简单的图像预处理（与 CLIP 近似；SARCLIP 自带 pipeline 也可以，但这里走轻量版）
        #    - resize 224, ToTensor, 归一化到 [-1, 1]（mean=0.5,std=0.5）
        from torchvision import transforms as T
        # 兼容老 torchvision 无 InterpolationMode 的情况（你已经做过兜底）
        try:
            from torchvision.transforms import InterpolationMode
            _b = dict(interpolation=InterpolationMode.BICUBIC)
        except Exception:
            _b = {}

        self.preprocess = T.Compose([
            T.Resize((224, 224), **_b),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print("[CGA/SARCLIP] init classes =", self.class_names)
        print("[CGA/SARCLIP] model =", model, "| tau =", self.tau, "| expand_ratio =", self.expand_ratio)

    def _crop_patches(self, img_path, boxes, scores, labels):
        # 若是 SAR 单通道，可选转 'L' 再堆 3 通道；否则直接 RGB
        # 你也可以把 force_grayscale=True 来强制灰度
        mode = 'L' if self.force_grayscale else 'RGB'
        image = Image.open(img_path).convert(mode)

        image_list = []
        ori_image_list = []

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            x1 = max(0, x1 - w * self.expand_ratio)
            y1 = max(0, y1 - h * self.expand_ratio)
            x2 = x2 + w * self.expand_ratio
            y2 = y2 + h * self.expand_ratio

            sub_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
            if self.force_grayscale and sub_image.mode == 'L':
                # 堆成 3 通道（与模型预期一致）
                arr = np.array(sub_image)  # HxW
                arr = np.repeat(arr[..., None], 3, axis=2)  # HxWx3
                sub_image = Image.fromarray(arr)

            if save_img:
                label_ = (
                    self.class_names[int(label)]
                    if isinstance(label, (int, np.integer)) and 0 <= int(label) < len(self.class_names)
                    else str(label)
                )
                os.makedirs(self.save_path, exist_ok=True)
                sub_image.save(os.path.join(self.save_path, f"sub_image_{i}_{score:.3f}_{label_}.jpg"))

            ori_image_list.append(sub_image)
            tensor = self.preprocess(sub_image).to(self.device)  # 3x224x224
            image_list.append(tensor)

        if len(image_list) == 0:
            return None, None
        return torch.stack(image_list, dim=0), ori_image_list

    _precomputed_miss_count = 0
    _precomputed_hit_count = 0
    _precomputed_feat_dir_resolved = None

    @torch.no_grad()
    def _load_precomputed_features(self, img_path):
        """Load pre-computed spatial feature map for an image."""
        feat_dir = _PRECOMPUTED_FEATURES_DIR
        if not feat_dir:
            return None

        # Resolve to absolute path once
        if self.__class__._precomputed_feat_dir_resolved is None:
            if os.path.isabs(feat_dir):
                self.__class__._precomputed_feat_dir_resolved = feat_dir
            else:
                # Try relative to project root (parent of sfod/)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.__class__._precomputed_feat_dir_resolved = os.path.join(project_root, feat_dir)
        feat_dir = self.__class__._precomputed_feat_dir_resolved

        # Parse img_path to find corruption type, split, and filename
        # Handles both absolute and relative paths
        # e.g. /xxx/dataset/RSAR/train/images-chaff/0000002.png
        # e.g. /xxx/dataset/RSAR/val/images/0000002.png
        parts = img_path.replace("\\", "/").split("/")
        corrupt = "clean"
        split = "train"
        fname = ""
        for i, p in enumerate(parts):
            if p.startswith("images"):
                corrupt = p.replace("images-", "") if "-" in p else "clean"
                # split is the directory before "images*"
                split = parts[i - 1] if i > 0 else "train"
                # filename is the next part
                fname = parts[i + 1] if i + 1 < len(parts) else ""
                break

        if not fname:
            return None

        npz_name = os.path.splitext(fname)[0] + ".npz"
        feat_path = os.path.join(feat_dir, corrupt, split, npz_name)

        if not os.path.exists(feat_path):
            cls = self.__class__
            cls._precomputed_miss_count += 1
            if cls._precomputed_miss_count <= 3:
                print(f"[CGA/precomputed] MISS ({cls._precomputed_miss_count}): {feat_path}")
                print(f"  img_path={img_path}, parsed: corrupt={corrupt}, split={split}, fname={fname}")
            return None

        self.__class__._precomputed_hit_count += 1
        data = np.load(feat_path)
        return data["features"]  # [grid_h, grid_w, dim], float16

    def _roi_pool_features(self, feature_map, boxes, img_size):
        """ROI average pool from pre-computed feature map for each bbox.

        Args:
            feature_map: [grid_h, grid_w, dim] spatial features
            boxes: [N, 4] xyxy bboxes in image coordinates
            img_size: (height, width) of original image

        Returns:
            [N, dim] pooled features (L2 normalized)
        """
        grid_h, grid_w, dim = feature_map.shape
        img_h, img_w = img_size

        # Scale boxes to grid coordinates
        scale_x = grid_w / img_w
        scale_y = grid_h / img_h

        features_list = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Expand by expand_ratio (same as _crop_patches)
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, x1 - bw * self.expand_ratio)
            y1 = max(0, y1 - bh * self.expand_ratio)
            x2 = min(img_w, x2 + bw * self.expand_ratio)
            y2 = min(img_h, y2 + bh * self.expand_ratio)

            # Map to grid
            gx1 = max(0, int(x1 * scale_x))
            gy1 = max(0, int(y1 * scale_y))
            gx2 = min(grid_w, int(np.ceil(x2 * scale_x)))
            gy2 = min(grid_h, int(np.ceil(y2 * scale_y)))

            # Ensure at least 1x1 region
            if gx2 <= gx1: gx2 = min(gx1 + 1, grid_w)
            if gy2 <= gy1: gy2 = min(gy1 + 1, grid_h)

            # Average pool over the region
            region = feature_map[gy1:gy2, gx1:gx2, :]  # [h, w, dim]
            pooled = region.reshape(-1, dim).mean(axis=0)  # [dim]

            # L2 normalize
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm

            features_list.append(pooled)

        return np.stack(features_list, axis=0).astype(np.float32)  # [N, dim]

    @torch.no_grad()
    def __call__(self, img_path, boxes, scores, labels):
        # Try pre-computed features first (instant, no GPU needed)
        feature_map = self._load_precomputed_features(img_path)
        if feature_map is not None:
            if len(boxes) == 0:
                return np.empty((0, len(self.class_names))), []

            # Get image size for coordinate mapping
            from PIL import Image as _PILImage
            with _PILImage.open(img_path) as _img:
                img_w, img_h = _img.size

            # ROI pool from pre-computed feature map
            image_features = self._roi_pool_features(feature_map, boxes, (img_h, img_w))

            # Compute logits (same as original)
            classifier = self.classifier.detach().cpu().numpy()
            logits_raw = self.tau * (image_features @ classifier)
            # Softmax
            logits_exp = np.exp(logits_raw - logits_raw.max(axis=-1, keepdims=True))
            logits = logits_exp / logits_exp.sum(axis=-1, keepdims=True)

            return logits, []

        # Fallback: original forward pass (slow, runs ViT-L-14)
        images, ori_image_list = self._crop_patches(img_path, boxes, scores, labels)
        if images is None:
            return np.empty((0, len(self.class_names))), []

        out = self.clip(image=images)
        image_features = out['image_features'] if isinstance(out, dict) else out[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = (self.tau * (image_features @ self.classifier)).softmax(dim=-1)
        return logits.detach().cpu().numpy(), ori_image_list


class TestMixins:
    def __init__(self):
        self.cga = None
        self._cga_sig = None

    def refine_test(self, results, img_metas):
        # 在 Teacher simple_test 中被调用，见 sfod/rotated_semi_two_stage.py
        # 这里仅做“后处理重打分”，不改变检测框位置
        class_names = getattr(self, "CLASSES", None)
        if not class_names:
            # Fallback to numbered classes to avoid crashes.
            class_names = [f"class_{i}" for i in range(len(results[0]))]

        scorer = _str_env("CGA_SCORER", "clip").lower()
        tau = _float_env("CGA_TAU", 100.0)
        expand_ratio = _float_env("CGA_EXPAND_RATIO", 0.4)

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        # Rebuild when environment changes (important for experiments).
        sig = {
            "scorer": scorer,
            "classes": tuple(class_names),
            "templates": _templates_from_env(
                default=(
                    ("a SAR image of a {}",)
                    if scorer == "sarclip"
                    else ("an aerial image of a {}",)
                )
            ),
            "tau": tau,
            "expand_ratio": expand_ratio,
            "sarclip_model": _str_env("SARCLIP_MODEL", "RN50"),
            "sarclip_pretrained": os.environ.get("SARCLIP_PRETRAINED", "").strip() or None,
            "sarclip_lora": os.environ.get("SARCLIP_LORA", "").strip() or None,
            "clip_model": _str_env("CLIP_MODEL", "RN50"),
        }

        if not hasattr(self, "cga") or self.cga is None or getattr(self, "_cga_sig", None) != sig:
            self._cga_sig = sig
            self.cga = None

            if scorer in ("none", "off", "disable", "disabled", ""):
                self.exclude_ids = []
            elif scorer == "sarclip":
                try:
                    self.cga = CGA(
                        class_names=class_names,
                        model=sig["sarclip_model"],
                        pretrained=sig["sarclip_pretrained"],
                        precision="amp",
                        templates=sig["templates"],
                        tau=tau,
                        expand_ratio=expand_ratio,
                        force_grayscale=False,
                    )
                    self.exclude_ids = []
                except Exception as e:
                    print(f"[CGA] WARNING: SARCLIP backend init failed, fallback to CLIP. err={e}")
                    self.cga = _ClipCGA(
                        class_names=list(class_names),
                        model=sig["clip_model"],
                        templates=sig["templates"],
                        tau=tau,
                        expand_ratio=expand_ratio,
                        device=device,
                    )
                    self.exclude_ids = []
            else:
                self.cga = _ClipCGA(
                    class_names=list(class_names),
                    model=sig["clip_model"],
                    templates=sig["templates"],
                    tau=tau,
                    expand_ratio=expand_ratio,
                    device=device,
                )
                self.exclude_ids = []

        boxes_list, scores_list, labels_list = [], [], []
        det_records = []
        for cls_id, result in enumerate(results[0]):
            if len(result) == 0:
                continue
            # mmrotate 的输出是 OBB（cx,cy,w,h,theta,...,score），先转为 AABB
            result_xyxy = obb2xyxy(result)
            boxes_list.append(result_xyxy[:, :4])
            scores_list.append(result[:, -1])
            labels_list.append([cls_id] * len(result))
            det_records.extend([det.copy() for det in result])

        if len(boxes_list) == 0:
            return results

        boxes_list = np.concatenate(boxes_list, axis=0)
        scores_list = np.concatenate(scores_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        if self.cga is None:
            return results

        img_path = img_metas[0]["filename"]

        cache_dir = os.environ.get("CGA_CACHE_DIR", "").strip()
        disable_cache = _str_env("CGA_DISABLE_CACHE", "").lower() in ("1", "true", "yes", "y")
        cache_verbose = _str_env("CGA_CACHE_VERBOSE", "").lower() in ("1", "true", "yes", "y")

        if cache_dir and not disable_cache and scorer == "sarclip":
            try:
                from .scorers.disk_cache import DiskCache

                cache = DiskCache(cache_dir)
                # Cache key is based on the actual crop coordinates used by CGA._crop_patches
                # (expanded by expand_ratio and cast to int), so small float jitter in boxes
                # won't destroy cache hit rate.
                crop_xyxy = boxes_list.astype(np.float32).copy()
                w = crop_xyxy[:, 2] - crop_xyxy[:, 0]
                h = crop_xyxy[:, 3] - crop_xyxy[:, 1]
                crop_xyxy[:, 0] = np.maximum(0.0, crop_xyxy[:, 0] - w * float(expand_ratio))
                crop_xyxy[:, 1] = np.maximum(0.0, crop_xyxy[:, 1] - h * float(expand_ratio))
                crop_xyxy[:, 2] = crop_xyxy[:, 2] + w * float(expand_ratio)
                crop_xyxy[:, 3] = crop_xyxy[:, 3] + h * float(expand_ratio)
                crop_xyxy = crop_xyxy.astype(np.int32)

                key_obj = {
                    "kind": "cga_logits_v1",
                    "scorer": scorer,
                    "model": sig.get("sarclip_model"),
                    "pretrained": sig.get("sarclip_pretrained"),
                    "templates": list(sig.get("templates") or []),
                    "tau": tau,
                    "expand_ratio": expand_ratio,
                    "classes": list(class_names),
                    "image": str(img_path),
                    "crop_xyxy": crop_xyxy.tolist(),
                }

                def _compute_logits():
                    logits_arr, _ = self.cga(img_path, boxes_list, scores_list, labels_list)
                    return logits_arr.tolist()

                entry, hit = cache.get_or_compute(key_obj, _compute_logits)
                if cache_verbose:
                    print(f"[CGA/cache] hit={hit} dir={cache_dir} n={len(boxes_list)}")
                logits = np.asarray(entry["value"], dtype=np.float32)
            except Exception as e:
                print(f"[CGA/cache] WARNING: cache failed, fallback to direct compute. err={e}")
                logits, _ = self.cga(img_path, boxes_list, scores_list, labels_list)
        else:
            logits, _ = self.cga(img_path, boxes_list, scores_list, labels_list)

        lambda_clip = float(np.clip(_float_env("CGA_LAMBDA", 0.2), 0.0, 1.0))
        det_dim = next((result.shape[1] for result in results[0] if len(result) > 0), 6)
        refined_by_class: list[list[np.ndarray]] = [[] for _ in range(len(class_names))]

        for i, prob in enumerate(logits):
            orig_label = int(labels_list[i])
            clip_pred = int(np.argmax(prob))
            det = det_records[i].copy()
            teacher_score = float(det[-1])

            if orig_label == clip_pred or lambda_clip <= 0.0:
                new_label = orig_label
                new_score = teacher_score
            else:
                teacher_dist = np.zeros(len(class_names), dtype=np.float32)
                teacher_dist[orig_label] = 1.0
                blended = (1.0 - lambda_clip) * teacher_dist + lambda_clip * prob.astype(np.float32)
                new_label = int(np.argmax(blended))
                new_score = float(max(0.0, teacher_score * float(blended[new_label])))

            det[-1] = new_score
            refined_by_class[new_label].append(det)

        results[0] = [
            np.stack(items, axis=0).astype(np.float32) if items else np.zeros((0, det_dim), dtype=np.float32)
            for items in refined_by_class
        ]

        return results


class _ClipCGA:
    def __init__(
        self,
        *,
        class_names: list[str],
        model: str,
        templates: tuple[str, ...],
        tau: float,
        expand_ratio: float,
        device: torch.device,
    ):
        import clip  # type: ignore

        self._clip = clip
        self.class_names = list(class_names)
        self.device = device
        self.expand_ratio = float(expand_ratio)
        self.tau = float(tau)

        self.model, self.preprocess = clip.load(model, device=self.device)
        self.model.eval()

        if not templates:
            templates = ("an image of a {}",)

        prompts: list[str] = []
        for c in self.class_names:
            for t in templates:
                prompts.append(t.format(c))

        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            t = len(templates)
            text_features = text_features.view(len(self.class_names), t, -1).mean(dim=1)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [C, D]

    def _crop_patches(self, img_path, boxes, scores, labels):
        image = Image.open(img_path).convert("RGB")
        image_list = []
        ori_image_list = []

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            x1 = max(0, x1 - w * self.expand_ratio)
            y1 = max(0, y1 - h * self.expand_ratio)
            x2 = x2 + w * self.expand_ratio
            y2 = y2 + h * self.expand_ratio

            sub_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
            if save_img:
                os.makedirs("_clip_img", exist_ok=True)
                sub_image.save(os.path.join("_clip_img", f"sub_image_{i}_{float(score):.3f}_{int(label)}.jpg"))

            ori_image_list.append(sub_image)
            image_list.append(self.preprocess(sub_image).to(self.device))

        if not image_list:
            return None, None
        return torch.stack(image_list, dim=0), ori_image_list

    @torch.no_grad()
    def __call__(self, img_path, boxes, scores, labels):
        images, ori_image_list = self._crop_patches(img_path, boxes, scores, labels)
        if images is None:
            return np.empty((0, len(self.class_names))), []

        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (self.tau * (image_features @ self.text_features.T)).softmax(dim=-1)
        return logits.detach().cpu().numpy(), ori_image_list

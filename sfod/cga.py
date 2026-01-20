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
import numpy as np
import torch
from PIL import Image

# ---------- 你的数据集类别（与 cfg & bbox_head.num_classes 保持一致） ----------
CLASSES = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
save_img = False


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
        model='RN50',
        # 把下面路径改为你的 SARCLIP RN50/ViT-B-32 权重（*.safetensors 或 *.pt）
        pretrained='/home/storageSDA1/Dataset/SARCLIP/RN50/rn50_model.safetensors',
        precision='amp',
        templates=('an SAR image of a {}', 'this SAR patch shows a {}'),
        tau=100.0,
        expand_ratio=0.4,
        force_grayscale=False
    ):
        super().__init__()
        import sar_clip  # 只依赖 sar_clip 的模型 & tokenizer & zero_shot_classifier，不用 data

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
                label_ = CLASSES[label]
                os.makedirs(self.save_path, exist_ok=True)
                sub_image.save(os.path.join(self.save_path, f"sub_image_{i}_{score:.3f}_{label_}.jpg"))

            ori_image_list.append(sub_image)
            tensor = self.preprocess(sub_image).to(self.device)  # 3x224x224
            image_list.append(tensor)

        if len(image_list) == 0:
            return None, None
        return torch.stack(image_list, dim=0), ori_image_list

    @torch.no_grad()
    def __call__(self, img_path, boxes, scores, labels):
        images, ori_image_list = self._crop_patches(img_path, boxes, scores, labels)
        if images is None:
            # 没有候选框
            return np.empty((0, len(self.class_names))), []

        # 前向（SARCLIP 输出 dict；取 image_features）
        out = self.clip(image=images)
        image_features = out['image_features'] if isinstance(out, dict) else out[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [N, D]

        # logits: [N, C]
        logits = (self.tau * (image_features @ self.classifier)).softmax(dim=-1)
        return logits.detach().cpu().numpy(), ori_image_list


class TestMixins:
    def __init__(self):
        self.cga = None

    def refine_test(self, results, img_metas):
        # 在 Teacher simple_test 中被调用，见 sfod/rotated_semi_two_stage.py
        # 这里仅做“后处理重打分”，不改变检测框位置
        if not hasattr(self, 'cga') or self.cga is None:
            # ★★ 按需修改 pretrained 路径 / 模型名 ★★
            self.cga = CGA(
                class_names=CLASSES,
                model='RN50',
                pretrained='/home/storageSDA1/Dataset/SARCLIP/RN50/rn50_model.safetensors',
                precision='amp',
                templates=(
                    'an SAR image of a {}',
                    'this SAR patch shows a {}',
                ),
                tau=100.0,
                expand_ratio=0.4,
                force_grayscale=False
            )
            # 注意：原仓库里 exclude_ids 针对 DIOR 的某些类。你是 RSAR 六类，这里置空。
            self.exclude_ids = []

        boxes_list, scores_list, labels_list = [], [], []
        for cls_id, result in enumerate(results[0]):
            if len(result) == 0:
                continue
            # mmrotate 的输出是 OBB（cx,cy,w,h,theta,...,score），先转为 AABB
            result_xyxy = obb2xyxy(result)
            boxes_list.append(result_xyxy[:, :4])
            scores_list.append(result[:, -1])
            labels_list.append([cls_id] * len(result))

        if len(boxes_list) == 0:
            return results

        boxes_list = np.concatenate(boxes_list, axis=0)
        scores_list = np.concatenate(scores_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        logits, _ = self.cga(img_metas[0]['filename'], boxes_list, scores_list, labels_list)

        # 与原逻辑一致：若 CLIP/SARCLIP 预测与原标签不一致，按一定权重融合
        for i, prob in enumerate(logits):
            pred = np.argmax(prob)
            if labels_list[i] != pred:
                if labels_list[i] not in self.exclude_ids:
                    scores_list[i] = scores_list[i] * 0.7 + float(prob[labels_list[i]]) * 0.3

        j = 0
        for i in range(len(results[0])):
            num_dets = len(results[0][i])
            if num_dets == 0:
                continue
            for _k in range(num_dets):
                results[0][i][_k, -1] = scores_list[j]
                j += 1

        return results

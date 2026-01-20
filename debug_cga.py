# /home/storageSDA1/liaojr/SFOD-RS/debug_cga_one.py
# 单图调试：调用 sfod.cga 里的 SARCLIP 版 CGA 做重打分

import numpy as np
from sfod.cga import CGA   # 确保已替换为 SARCLIP 版本的 cga.py

import sys
sys.path.insert(0, "/home/storageSDA1/liaojr/SARCLIP")  # 指向包含 sar_clip/ 的目录
import sar_clip
print("sar_clip at:", sar_clip.__file__)


# 1) 类别（务必与 cfg & bbox_head.num_classes 顺序一致）
CLASSES = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
cls_to_id = {c: i for i, c in enumerate(CLASSES)}

# 2) 两行 DOTA 标注（四点+类别+diff）
poly_lines = [
    "61.0 252.0 208.0 99.0 226.0 116.0 78.0 269.0 harbor 0",
    "81.0 7.0 118.0 41.0 33.0 131.0 -4.0 97.0 harbor 0"
]

def poly_line_to_xyxy_and_label(line):
    parts = line.strip().split()
    coords = list(map(float, parts[:8]))   # x1 y1 x2 y2 x3 y3 x4 y4
    cls = parts[8]
    xs = coords[0::2]; ys = coords[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)  # 取AABB包围框
    return [x1, y1, x2, y2], cls_to_id[cls]

# 3) 组装 CGA 入参
boxes, scores, labels = [], [], []
for line in poly_lines:
    b, lid = poly_line_to_xyxy_and_label(line)
    boxes.append(b)
    labels.append(lid)
    scores.append(0.9)  # 初始分数，CGA会重打分

boxes  = np.array(boxes,  dtype=np.float32)
scores = np.array(scores, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# 4) 图片路径（与标注对应）
img_path = "/home/storageSDA1/liaojr/dataset/RSAR/train/images/0000002.png"

# 5) 初始化 SARCLIP 版 CGA 并前向
#    把 PRETRAINED_PATH 改成你的 SARCLIP RN50/ViT-B-32 权重路径
PRETRAINED_PATH = "/home/storageSDA1/Dataset/SARCLIP/RN50/rn50_model.safetensors"

cga = CGA(
    class_names=CLASSES,
    model='RN50',
    pretrained=PRETRAINED_PATH,
    precision='amp',
    templates=('an SAR image of a {}', 'this SAR patch shows a {}'),
    tau=100.0,            # 温度，可试 50~150
    expand_ratio=0.4,     # 框扩张，可试 0.2~0.6
    force_grayscale=False # 若是单通道SAR且想强制灰度->3通道，可设 True
)

logits, patches = cga(img_path, boxes, scores, labels)

print("logits shape:", logits.shape)     # [N, num_classes]
print("top-1 idx:", logits.argmax(axis=1))
print("prob for GT cls:", [float(logits[i, labels[i]]) for i in range(len(labels))])

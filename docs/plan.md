下面给你一个“可落地”的完整任务设计：按里程碑拆分、每一步的检验方法（能跑/能对/能复现）、最后的实验清单与可视化/辅助材料清单、以及最终验收期望结果。默认约束是：**尽量不动你当前老训练栈（torch=1.7.1, mmcv=1.7.2, mmdet/mmrotate 老版）**，同时让项目 **DIOR 能跑 + RSAR 能跑 + 未来 RSAR 干扰集能无痛加进来**。IRAOD 的 README 本身就宣称支持 DIOR 与 RSAR，并给出了 RSAR 的数据组织与训练/测试入口。([GitHub][1])

---

## 0. 总目标、关键风险与推荐技术路线

### 0.1 总目标（项目最终形态）

1. **同一代码仓 + 同一训练入口**（train.py/test.py）能在：

   * DIOR（含 corruption）上跑通并复现 IRAOD baseline/自训练结果；
   * RSAR（原始数据集）上跑通并验证；
   * RSAR-Interference（你后续加入的干扰版本）通过一个配置项（如 `corrupt=interference_xxx`）切换即可训练/测试。
2. **CGA 打分模块可插拔**：

   * DIOR：用现有 CLIP-CGA；
   * RSAR：用 SARCLIP 做同样的“图文相似度打分/课程筛选”更合理（域内 VLM）。
3. **可复现与可分析**：每个实验都有配置、日志、权重、评估结果、可视化与统计图。

### 0.2 关键风险点（你提到的两大问题）

* **风险 A：SARCLIP 的 `nn.MultiheadAttention(batch_first=True)` 与 torch 1.7.1 不兼容**，直接报错。你描述的现象本质上就是 batch_first 参数在老版本 MultiheadAttention 里不存在。([GitHub][2])
* **风险 B：RSAR 图片格式混杂（jpg/bmp/png），dataset 读取可能有硬编码扩展名的问题**。虽然 IRAOD README 的 RSAR 结构说明里明确写了 images 可包含 `*.jpg, *.bmp, *.png`，但代码实现是否真正“扩展名无关”仍需验证。([GitHub][1])

### 0.3 推荐路线（按风险最小优先）

我建议你把方案分成“主路线 + 备选路线”，这样项目推进不会卡死：

**主路线（优先）**：

* **不升级整个 mmcv/mmdet/mmrotate 训练栈**；
* 只对 **SARCLIP 做一个极小的兼容性补丁**（去掉 batch_first + 前后 transpose），让它能在 torch 1.7.1 下跑通；
* IRAOD 内部把 CGA 的 encoder 做成可插拔（CLIP/SARCLIP）。

**备选路线（稳妥兜底）**：双环境离线打分

* 检测训练环境保持老栈不动；
* 单独建一个新环境（torch≥1.11/2.x）只跑 SARCLIP “离线打分导出”（对每张图或每个 corruption 版本预先算好分数，存成 json/pkl）；
* 训练时只读分数文件，不在训练进程里 import SARCLIP。

> 这条路线“工程更干净/风险更低”，但前提是 CGA 的打分对象可预先确定（例如对固定 corruption 版本打分）。如果 CGA 对“训练中随机增强后的图”实时打分，那离线方案就要做缓存/服务化（后面也给你预案）。

**不建议（除非必须）**：整体升级 mmlab 新栈
mmrotate 与 mmcv/mmdet 版本耦合很强（官方也明确给过兼容区间），整体升级会牵一串依赖，风险最大。([GitHub][3])

---

## 1. 里程碑式任务拆解（每步含检验方法）

### Milestone 1：基线复现与工程“可跑”验证（DIOR & RSAR）

**目标**：先不改任何核心算法，确认 IRAOD 训练/测试入口、日志、eval、可视化都能跑通。

#### Step 1.1 环境冻结与可复现记录

* **做什么**

  * 用 IRAOD README 的方式建立环境（torch=1.7.1 等）。([GitHub][1])
  * 固定 CUDA/cuDNN、python 版本、pip freeze 导出。
* **检验方法**

  * `python -c "import torch; print(torch.__version__); import mmcv; print(mmcv.__version__)"`
  * 能 import mmrotate/mmdet 相关模块；
  * 把 `pip freeze > env_lock.txt` 存档。
* **产物**

  * `env_lock.txt`
  * `system_info.md`（GPU/driver/CUDA 版本）

#### Step 1.2 DIOR 数据集路径与 dataloader sanity

* **做什么**

  * 按 IRAOD README 的 DIOR 目录组织数据（JPEGImages/ImageSets/Corruption 等）。([GitHub][1])
  * 写一个 `tools/sanity_check_dior.py`：

    * 随机抽 20 张图；
    * 读取标注；
    * 可视化旋转框叠加输出到 `work_dirs/sanity/dior_vis/`。
* **检验方法**

  * 脚本运行不报错；
  * 叠加框位置合理（不漂、不旋转错位、不出现离谱坐标）。
* **产物**

  * `dior_sanity_report.json`（统计：图片数、标注数、空标注数、异常框数）
  * 可视化样例图（20 张）

#### Step 1.3 DIOR：跑一个“超小 smoke train”

* **做什么**

  * 用 IRAOD README 给的训练脚本跑起来（可以先把 epoch/iter 改很小，比如 200 iter）。([GitHub][1])
* **检验方法**

  * loss 正常下降或至少稳定；
  * eval 能跑、mAP 不为 NaN；
  * 训练日志可被解析（tensorboard/纯 log）。
* **产物**

  * `work_dirs/exp_smoke_dior/`（含 log、latest.pth、eval 结果）

#### Step 1.4 RSAR：数据结构 + dataloader sanity

* **做什么**

  * 按 IRAOD README 的 RSAR 数据结构准备：`train|val|test/annfiles + images(可含jpg/bmp/png)`。([GitHub][1])
  * 写 `tools/sanity_check_rsar.py`：

    * 扫描 annfiles；
    * 对每个 annfile 找对应 image（重点检查扩展名问题，见后续 Milestone 2）；
    * 抽样可视化。
* **检验方法**

  * 输出统计：

    * ann 文件数 = 可找到的 image 数；
    * 缺失对齐的样本数=0（或定位出具体文件名）。
* **产物**

  * `rsar_sanity_report.json`
  * `work_dirs/sanity/rsar_vis/` 样例

#### Step 1.5 RSAR：跑一个“超小 smoke test”

* **做什么**

  * 先用现有 RSAR 配置跑测试（或训 200 iter 再测）。IRAOD README 给了 RSAR 的 train/test 命令示例。([GitHub][1])
* **检验方法**

  * `test.py --show-dir vis_rsar` 能生成可视化输出；([GitHub][1])
  * mAP 输出正常。
* **产物**

  * `work_dirs/exp_smoke_rsar/`
  * `vis_rsar/` 若干张结果图

---

### Milestone 2：解决 RSAR 多格式图片读取（jpg/bmp/png）“工程级”适配

**目标**：不管 RSAR 图片后续怎么混合格式、甚至未来你加干扰版本图片，dataloader 都能稳定找到对应文件。

#### Step 2.1 先做“事实核验”：代码里是否硬编码扩展名

* **做什么**

  * 全仓搜索：`grep -R "\.jpg" dataset -n`、`grep -R "JPEGImages" -n`、`grep -R "img_suffix" -n`。
* **检验方法**

  * 找到 RSAR dataset 类的 `load_annotations()` 或 “annfile->img_path” 的映射逻辑。
* **产物**

  * `notes/dataloader_extension_audit.md`（记录哪些位置需要改）

#### Step 2.2 推荐实现：扩展名无关的 resolve 机制

给你一个“最抗干扰”的实现策略（建议）：

* **做法 A（优先）**：建立 index（更快）

  * 在 dataset 初始化时扫描 `images/` 目录，把所有文件名（不带扩展名）映射到真实路径：
    `id -> /path/to/xxx.jpg|png|bmp`
  * 读取 annfile 时，只用 annfile 基名去查 map。
* **做法 B（简单但慢）**：每次 resolve 用 glob

  * `glob(f"{img_dir}/{img_id}.*")`，并限定允许后缀集合 `{'.jpg','.png','.bmp'}`。

**检验方法（必须有自动化）**

* 写 `tools/check_image_ann_alignment.py`：

  * 全量遍历 annfiles：

    * 记录找不到 image 的 annfile；
    * 记录一个 annfile 匹配到多个 image 的冲突（例如同名 .jpg 与 .png 同时存在）。
* 输出报告：

  * missing=0；
  * conflict=0（或给出冲突解决规则，如优先 png）。

**产物**

* `alignment_report.csv`（annfile, resolved_image_path, status）
* `dataset_index.pkl`（如果用做法 A）

> 这样未来你加 RSAR-Interference，只要你的干扰图保持同名（或有规则映射），就不会破坏训练代码。

---

### Milestone 3：SARCLIP 在 torch 1.7.1 下可用（最小改动补丁）

**目标**：在不升级训练栈的前提下，让 SARCLIP 的文本 Transformer 不再因为 `batch_first=True` 崩溃。

#### Step 3.1 SARCLIP 侧的最小兼容补丁思路

你遇到的报错是：
`TypeError: __init__() got an unexpected keyword argument 'batch_first'`

在老 PyTorch 里，`nn.MultiheadAttention` 默认是 **sequence-first**（形状通常是 `(L, N, E)`），而 batch_first 版本是 `(N, L, E)`。([PyTorch Documentation][4])
**补丁核心**：

* 不传 `batch_first=True`；
* forward 时手动 transpose 输入/输出。

#### Step 3.2 推荐补丁实现（兼容新老 PyTorch 的写法）

把下面逻辑放到 SARCLIP 的 Text Transformer 模块（或你 fork/拷贝到 IRAOD 内部的 SARCLIP 文本编码模块）：

```python
# pseudo-code / design
class MHAWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__()
        self.batch_first = False
        try:
            self.mha = nn.MultiheadAttention(embed_dim, num_heads,
                                             dropout=dropout,
                                             batch_first=True, **kwargs)
            self.batch_first = True
        except TypeError:
            # torch<=1.10 fallback
            self.mha = nn.MultiheadAttention(embed_dim, num_heads,
                                             dropout=dropout, **kwargs)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, E) expected by rest of model
        if not self.batch_first:
            x = x.transpose(0, 1)  # (B,L,E)->(L,B,E)
        out, _ = self.mha(x, x, x,
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask,
                          need_weights=False)
        if not self.batch_first:
            out = out.transpose(0, 1)  # (L,B,E)->(B,L,E)
        return out
```

#### Step 3.3 补丁验证（你要的“每一步怎么验”）

**最小单测（强烈建议写）**

* `tests/test_sarclip_mha_compat.py`：

  1. 构造随机输入 `x=torch.randn(B,L,E)`；
  2. forward 输出 shape 必须是 `(B,L,E)`；
  3. 输出不得出现 NaN/Inf；
  4. 多次 forward 结果 deterministic（固定 seed + eval 模式）。

**集成测试**

* 在 IRAOD 环境中：

  * `python tools/sarclip_smoke.py --image path/to/one.png --prompts "an SAR image of ship" ...`
  * 能输出一个分数（即使分数对不对先不管，先保证 pipeline 完整）。

**产物**

* SARCLIP patch（commit / diff）
* `sarclip_smoke.log`（记录耗时与显存）

> 如果这里出现第二波兼容问题（例如 safetensors、某些 torch 新 API），就立刻切到“备选路线：双环境离线打分”，别在主干上死磕。

---

### Milestone 4：把“打分模型”做成 IRAOD 内部可插拔模块（CLIP ↔ SARCLIP）

**目标**：DIOR 继续走 CLIP-CGA；RSAR 走 SARCLIP-CGA；两套逻辑尽量共用一个 CGA 计算框架，只替换 encoder。

#### Step 4.1 抽象出统一接口（关键工程点）

在 IRAOD 里新增一个 scorer/encoder registry（建议目录结构）：

```
iroad/
  scorers/
    __init__.py
    base.py         # BaseScorer: encode_image, encode_text, score(...)
    clip_scorer.py  # 现有 CLIP 封装
    sarclip_scorer.py
    cache.py        # 磁盘/内存 cache
```

统一接口建议输出：

* `image_emb: (B, D)`
* `text_emb: (T, D)`
* `similarity: (B, T)`（cosine similarity）

#### Step 4.2 CGA 的“输入/输出契约”固定下来

无论 CLIP 还是 SARCLIP，你都要让 CGA 看到同一类数据：

* 输入：image（可能是 clean/corrupt）、prompt 文本列表（通常是描述 corruption 或类别）
* 输出：score（标量或向量）

  * 标量：用于 curriculum（例如越“清晰/越不干扰”分越高）
  * 向量：用于多 prompt（例如对多个干扰类型打分）

**检验方法**

* 用同一批图片：

  * CLIP scorer 跑通；
  * SARCLIP scorer 跑通；
  * 两者输出维度一致、数值范围合理（例如 cosine similarity 在 [-1,1] 或经 softmax 后 [0,1]）。

#### Step 4.3 Cache（非常关键，不然训练会慢到不可用）

建议 cache 的 key 至少包含：

* `image_path`（或 image_id）
* `corrupt` 类型/强度（如果有）
* prompt 列表的 hash
* encoder 名称与版本（CLIP-ViT-B/32 vs SARCLIP-ViT-B/32 等）

**检验方法**

* 第一次跑：cache miss 比例接近 100%；
* 第二次跑同样设置：cache hit 比例接近 100%，速度显著提升；
* cache 文件大小与条目数量合理。

---

### Milestone 5：把 RSAR（以及未来 RSAR-干扰集）纳入同一个“corrupt 切换机制”

IRAOD README 已经展示了训练时通过 `--cfg-options corrupt="cloudy"` 这样的方式切 corruption。([GitHub][1])
你要做的是：把 RSAR 的干扰版本也做成同样的开关即可。

#### Step 5.1 干扰数据的目录规范（建议你现在就定）

为了未来最省事，我建议干扰版本的数据结构做到“同名映射”，例如：

```
RSAR_ROOT/
  train/
    annfiles/
    images/                 # clean
    images-interf_jamA/     # 干扰A，同名文件不同内容
    images-interf_jamB/
  val/
    annfiles/
    images/
    images-interf_jamA/
  test/...
```

这样你只要在 dataset init 时：

* 如果 `corrupt is None`: 用 `images/`
* 如果 `corrupt == 'interf_jamA'`: 用 `images-interf_jamA/`

**检验方法**

* 对同一个 annfile：

  * clean 与 interfered 的 resolve image 均存在；
  * 尺寸一致（如果不一致要明确 resize 策略）；
  * 可视化对齐不漂。

#### Step 5.2 干扰强度/类型的配置化

建议把干扰类型做成：

* `corrupt=interf_jamA`（类别）
* `corrupt_severity=1..5`（强度，可选）

---

## 2. 实验设计：完整实验清单（从“跑通”到“论文级对比”）

下面给你一个**可以直接落地成实验追踪表（Excel/CSV）**的清单。每个实验建议至少跑 **3 个随机种子**（seed=0/1/2），并固定：

* backbone、学习率、batch、数据划分；
* evaluator（mAP）；
* prompt 模板与词表（非常关键）。

### 2.1 实验组 A：Sanity / Smoke（必做）

| 编号 | 数据集  | 方法                    | 目的                     | 训练规模      |
| -- | ---- | --------------------- | ---------------------- | --------- |
| A1 | DIOR | Baseline（监督）          | 验证训练/评估链路              | 200 iters |
| A2 | DIOR | UnbiasedTeacher（无CGA） | 验证半监督链路                | 200 iters |
| A3 | DIOR | UT + CGA(CLIP)        | 验证 CLIP-CGA 集成         | 200 iters |
| A4 | RSAR | Baseline（监督）          | 验证 RSAR dataset + eval | 200 iters |
| A5 | RSAR | UT + CGA(CLIP)        | 确认 RSAR 也能跑 CGA        | 200 iters |
| A6 | RSAR | UT + CGA(SARCLIP)     | 确认 SARCLIP 打分可用        | 200 iters |

**验收**：全部无报错；mAP 有数值；可视化输出存在。

---

### 2.2 实验组 B：DIOR 复现与对照（用于证明你没改坏原系统）

| 编号 | 方法       | corrupt    | scorer | 备注                         |
| -- | -------- | ---------- | ------ | -------------------------- |
| B1 | Baseline | None       | -      | supervised 基线              |
| B2 | UT       | None       | -      | 半监督基线                      |
| B3 | UT+CGA   | cloudy     | CLIP   | 与 README 示例一致([GitHub][1]) |
| B4 | UT+CGA   | brightness | CLIP   | corruption 泛化              |
| B5 | UT+CGA   | contrast   | CLIP   | corruption 泛化              |

**输出指标**：mAP 总体、每类 AP、训练曲线、每个 corruption 的 drop。

---

### 2.3 实验组 C：RSAR（核心对比：CLIP vs SARCLIP）

| 编号 | 方法       | scorer  | prompt 模板         | 目的                 |
| -- | -------- | ------- | ----------------- | ------------------ |
| C1 | Baseline | -       | -                 | RSAR supervised 基线 |
| C2 | UT       | -       | -                 | RSAR 半监督基线         |
| C3 | UT+CGA   | CLIP    | 与 DIOR 同模板        | “迁移CLIP”效果         |
| C4 | UT+CGA   | SARCLIP | SAR 模板1（通用）       | 主方案                |
| C5 | UT+CGA   | SARCLIP | SAR 模板2（更贴近干扰/噪声） | prompt ablation    |
| C6 | UT+CGA   | SARCLIP | 模板1               | 去掉 cache（对比速度/稳定性） |

> SARCLIP 是面向 SAR 的 CLIP 框架，README 给了零样本分类、检索、推理脚本与权重获取方式。([GitHub][5]) 你这里只是拿它做“相似度评分器”，不一定要做任何训练。

**输出指标**：

* mAP（整体/分类别）；
* pseudo-label 统计（每 epoch 伪标签数量、平均置信度、筛除比例）；
* CGA score 分布（mean/std，随 epoch 的变化）；
* 训练耗时（iter/s）与显存。

---

### 2.4 实验组 D：RSAR-Interference（你后续加数据后的鲁棒性评测）

| 编号 | train 数据              | test 数据     | scorer  | 目的               |
| -- | --------------------- | ----------- | ------- | ---------------- |
| D1 | clean                 | interf_jamA | CLIP    | 基线鲁棒性            |
| D2 | clean                 | interf_jamA | SARCLIP | 看 SARCLIP 是否更抗域偏 |
| D3 | clean+interf_jamA（混训） | interf_jamA | SARCLIP | 主推：鲁棒提升          |
| D4 | clean+多干扰（混训）         | 各干扰         | SARCLIP | 泛化能力             |

**补充**：如果你干扰有强度等级（1..5），再加一组“强度曲线实验”。

---

## 3. 可视化与辅助材料清单（你问的“全套配套”）

这里给你一个“项目交付级”的材料列表，基本就是论文/报告里常见的那套。

### 3.1 必备可视化（定性）

1. **检测结果叠加图**：

   * 每个数据集（DIOR/RSAR）各抽 50 张；
   * 每个方法（Baseline/UT/UT+CGA_CLIP/UT+CGA_SARCLIP）各输出一套；
   * 干扰集：同一张图 clean vs interference 的检测对比图（并排）。
2. **失败案例集**：

   * Top false positive / false negative；
   * 小目标密集区、强干扰区、边缘截断目标等。

> IRAOD 的测试命令示例里已经用 `--show-dir` 输出可视化结果（RSAR 示例给了 `vis_rsar`）。([GitHub][1])

### 3.2 必备可视化（定量）

1. **训练曲线**：loss、lr、mAP vs epoch（建议 tensorboard）
2. **PR 曲线**：按类别输出 PR（对比 CLIP vs SARCLIP）
3. **CGA score 分布**：

   * histogram/violin：score 的分布；
   * score vs pseudo-label 保留率（散点/曲线）。
4. **鲁棒性曲线**（干扰强度）：mAP vs severity（1..5）
5. **时间/算力**：iter/s、GPU memory、总训练时长

### 3.3 辅助脚本（建议你都做成 tools/）

* `tools/check_image_ann_alignment.py`（扩展名/缺失/冲突报告）
* `tools/vis_random_samples.py`（随机抽样可视化）
* `tools/export_metrics.py`（从 log/json 解析指标到 CSV）
* `tools/plot_all.py`（一键生成曲线图：mAP、score 分布、鲁棒曲线）
* `tools/ablation_table.py`（自动汇总成 LaTeX/markdown 表）
* `tools/sarclip_smoke.py`（SARCLIP 单图打分）
* `tools/cache_benchmark.py`（cache 命中率与速度）

### 3.4 实验追踪与复现材料

* `experiments.csv`：每行一个 run（含 git commit、seed、cfg、数据版本号、权重路径）
* `configs/`：每个实验对应的 config（禁止“手改命令行但不落 config”）
* `README_experiments.md`：说明如何复现任意一个 run
* `MODEL_ZOO.md`：列出你最终保留的 checkpoint 与性能

---

## 4. 最终验收标准与期望结果

### 4.1 阶段性验收（工程视角）

* **验收 1**：DIOR 与 RSAR 都能 “smoke train + eval + 可视化” 完整闭环（Milestone 1）。
* **验收 2**：RSAR 的 `jpg/bmp/png` 全量对齐检查通过（missing=0, conflict=0）（Milestone 2）。
* **验收 3**：SARCLIP 在 torch 1.7.1 环境下能完成打分，不再因为 batch_first 崩溃（Milestone 3）。
* **验收 4**：同一套 CGA 代码，通过 config 切换 scorer=CLIP/SARCLIP 即可运行（Milestone 4）。
* **验收 5**：干扰数据加入后，只新增数据目录与少量配置项即可训练/测试（Milestone 5）。

### 4.2 期望结果（算法效果视角，给一个“合理但不夸张”的目标）

* 在 **RSAR** 上：

  * **UT+CGA(SARCLIP)** 的性能应至少 **不低于** UT+CGA(CLIP)，并且更稳定（方差更小、伪标签筛选更合理）。
* 在 **RSAR-Interference** 上：

  * 用 SARCLIP 做 curriculum/筛选的版本应表现出 **更好的鲁棒性曲线**（随干扰强度增加，mAP 下降更慢）。
* 在 **DIOR** 上：

  * 保持与原 IRAOD 管线一致的表现（证明你的改动是“可插拔不破坏”）。([GitHub][1])

---

## 5. 如果你希望我帮你把“关键补丁点”进一步落到文件级

你现在的问题里，真正会卡进度的是 **SARCLIP 的 batch_first 兼容** 与 **RSAR ann->img 的扩展名 resolve**。
如果你愿意把以下两类信息贴出来（不需要全仓）我就能把方案精确到“改哪几个文件、改哪几行、需要新增哪些 config 字段”：

1. IRAOD 里 CGA/CLIP 打分模块对应的 Python 文件路径与关键函数（你 grep 出来即可）。
2. RSAR dataset 类里 “annfile->img_path” 的那段代码片段。

但即使不贴，现在上面这份任务设计也已经足够你按里程碑推进，把项目稳定跑起来并形成可发表/可交付的实验体系。

[1]: https://github.com/Jordan-Liao/IRAOD/ "https://github.com/Jordan-Liao/IRAOD/"
[2]: https://github.com/pytorch/pytorch/issues/25100 "https://github.com/pytorch/pytorch/issues/25100"
[3]: https://github.com/open-mmlab/mmrotate/blob/main/docs/en/faq.md "https://github.com/open-mmlab/mmrotate/blob/main/docs/en/faq.md"
[4]: https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html "https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html"
[5]: https://github.com/CAESAR-Radi/SARCLIP "https://github.com/CAESAR-Radi/SARCLIP"

---

## 6. Phase 2: Method-Level Frontier Experiments (2026-03-22 ~ 2026-03-26)

在 Phase 1 完成工程闭环后，进入方法创新阶段。以 24ep OrientedRCNN-R50-FPN (mAP=0.701) 为 baseline anchor，通过 open-researcher-v2 自动化实验框架探索方法级改进。

### 6.1 Baseline Anchor

| Config | Schedule | NMS IoU | mAP | Checkpoint |
|--------|----------|---------|-----|-----------|
| OrientedRCNN + R50 + FPN | 24ep, step=[16,22] | 0.30 | **0.701** | work_dirs/frontier_008_24ep/epoch_21.pth |

### 6.2 Frontier Experiments

| ID | Method | mAP | vs Baseline | Status | Experiment ID |
|----|--------|-----|-------------|--------|---------------|
| frontier-001 | Anchor baseline (12ep) | 0.6544 | -0.0466 | reference  E0042 |
| frontier-002 | LR schedule tuning | 0.6609 | -0.0401 | discarded  E0043 |
| frontier-004 | GWD loss | 0.6040 | -0.0970 | discarded (regression)  E0044 |
| frontier-005 | PolyRotate aug | 0.6226 | -0.0784 | discarded  E0045 |
| frontier-006 | Multi-scale training | 0.6366 | -0.0644 | discarded  E0046 |
| frontier-007 | Score threshold tuning | 0.6623 | -0.0387 | discarded  E0047 |
| frontier-008 | **24ep schedule** | **0.7005** | **+0.0000** | **anchor (baseline)**  E0048 |
| frontier-009 | Cosine LR | 0.6659 | -0.0351 | discarded  E0049 |
| frontier-012 | Weight decay 10x | 0.5184 | -0.1826 | discarded (severe)  E0051 |
| frontier-013 | Cosine LR seed1 | 0.6731 | -0.0279 | discarded  E0053 |
| frontier-014 | NMS IoU sweep | 0.6734 (thr=0.30) | — | confirmed NMS=0.30  E0055 |
| frontier-015 | 12ep + NMS sweep | 0.6686 | -0.0324 | discarded  E0057 |
| frontier-020 | FPN → PAFPN | 0.7005 | -0.0005 | discarded (marginal)  E0061 |
| frontier-021 | RoI cls CE → CEFocalLoss | 0.6212 | -0.0798 | discarded (regression)  E0062 |
| frontier-026 | FPN → OCA-FPN | — | — | running  E0063 |

### 6.3 SFOD Iteration Chain (exp_m ~ exp_ay)

Semi-supervised iterative training experiments on RSAR with CGA/LoRA/EMA variations. See docs/semi_supervised_experiments.md for detailed per-class results.

### 6.4 Innovation 1: OrthoNet & OCA-FPN

- **OrthoNet backbone**: Orthogonal channel attention network, registered as `OrthoNet` in mmdet_extension
- **OCA-FPN neck**: FPN + OrthoChannelAttention on each output level
- Config: `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_o_rsar.py`
- frontier-026 (OCA-FPN): **running** on 134 server

### 6.5 Innovation 2: SAR-CLIP LoRA Fine-tuning

用 SARDet100k 训练集裁剪目标 patch + 干扰增强 → LoRA 微调 SAR-CLIP (ViT-L-14) → 提升 CGA 打分精度。

**LoRA 训练**: 4 linear layers, 122,880 trainable params (0.12%), 147,796 patches from RSAR train
**CGA 零样本分类精度** (RSAR 6类):

| Scorer | Accuracy |
|--------|----------|
| ViT-L-14 (no LoRA) | 0.6021 |
| ViT-L-14 + SARDet LoRA | **0.6513** (+4.9%) |

**端到端检测 mAP** (SFOD chain, best iteration exp_ax):

| CGA Scorer | mAP |
|------------|-----|
| No LoRA | 0.6842 |
| With LoRA | **0.6943** (+1.0%) |

**SFOD Iteration Progression** (LoRA-enhanced CGA):

| Iteration | No LoRA mAP | With LoRA mAP | Delta |
|-----------|-------------|---------------|-------|
| exp_x | 0.6694 | 0.6797 | +0.0103 |
| exp_z | 0.6668 | 0.6781 | +0.0113 |
| exp_ab | 0.6706 | 0.6794 | +0.0088 |
| exp_an | 0.6800 | 0.6909 | +0.0109 |
| exp_ar | 0.6828 | 0.6935 | +0.0107 |
| exp_au | 0.6834 | 0.6938 | +0.0104 |
| **exp_ax** | **0.6842** | **0.6943** | **+0.0101** |

### 6.6 Key Findings

1. **24ep schedule** (frontier-008, mAP=0.7005) is the well-tuned anchor
2. Simple neck/loss swaps (PAFPN, CEFocal, GWD, cosine LR) all regress — 24ep anchor is hard to beat
3. **SAR-CLIP LoRA** consistently improves CGA scoring (+4.9% zero-shot accuracy) and end-to-end mAP (+1.0%)
4. **SFOD iteration chain** with LoRA-enhanced CGA reaches mAP=0.6943, stable +1% gain across iterations
5. **OCA-FPN** (OrthoChannelAttention) experiment in progress


---

## §7 Phase 3 完成：RSAR 电子干扰鲁棒性评测

> **状态**: ✅ **全部完成** (2026-04-05)
> **实验编号**: E0097 ~ E0103
> **详细记录**: `docs/experiment.md` Phase 3 部分, `docs/semi_supervised_experiments.md` Phase 3 部分

### 7.1 实验设计

**任务**: 无源目标检测（Source-Free Object Detection）— 在 7 种电子干扰下评估 RSAR 旋转目标检测

**方法**: SFOD + OrthoNet backbone + SARCLIP ViT-L-14 LoRA CGA

**数据协议**:
| 数据集 | 来源 | 是否干扰 | 用途 |
|---|---|---|---|
| train (有标签) | train/images/ | ❌ 否 | 监督学习 |
| val (无标签) | val/images-${corrupt}/ | ✅ 是 | 半监督学习 |
| test (测试) | test/images-${corrupt}/ | ✅ 是 | 评估 |

**训练配置**: 24ep, 5×GPU(RTX 4090), BS=40, SGD lr=0.02 step=[16,22], score_thr=0.5, momentum=0.9996, weight_u=0.5

### 7.2 结果汇总

| 干扰类型 | mAP | 相比 Clean 下降 | 难度 |
|---|---|---|---|
| gaussian_white_noise | **0.569** | -15.1% | ★☆☆ 轻度 |
| point_target | **0.568** | -15.2% | ★☆☆ 轻度 |
| chaff | **0.486** | -27.5% | ★★☆ 中等 |
| noise_suppression | **0.235** | -64.9% | ★★★ 困难 |
| smart_suppression | **0.188** | -71.9% | ★★★ 困难 |
| am_noise_vertical | **0.115** | -82.8% | ★★★★ 极难 |
| am_noise_horizontal | **0.097** | -85.5% | ★★★★ 极难 |
| **平均** | **0.323** | **-51.8%** | — |

### 7.3 技术突破

Phase 3 过程中解决了以下关键工程问题:

1. **伪标签崩溃问题**: 训练 epoch 14-15 mAP 从 0.48 暴跌至 0.005。根因: EMA 教师置信度跌破 score_thr → 伪标签归零 → 学生遗忘干扰域。修复: score_thr 0.7→0.5, momentum 0.998→0.9996, weight_u 1→0.5
2. **NCCL 死锁**: mmcv `_get_max_memory` 的 dist.reduce + `_parse_losses` 的 all_reduce 在单 rank 异常时导致全局死锁。修复: 移除非必要集合通信
3. **NaN 死循环**: loss_bbox_unlabeled NaN 污染 BatchNorm running stats → 永久 NaN。修复: 三层防御（退化 bbox 过滤 + loss 清零 + BN 回滚）
4. **EMA 架构不匹配**: EMA config 用 ResNet50 而学生用 OrthoNet，导致 EMA mAP=0。修复: EMA config 改为 OrthoNet
5. **数据协议修正**: patches.py 误将有标签 train 也替换为干扰图像。修复: 仅对 img_prefix_u 做 corrupt 替换

### 7.4 下一步方向

- [ ] 对强干扰（noise_suppression/smart_suppression/AM噪声）探索更强的域自适应策略
- [ ] 引入 teacher 置信度监控 + 自适应 score_thr 避免伪标签崩溃
- [ ] 探索 frequency-domain augmentation 提升对 AM 噪声的鲁棒性
- [ ] 与 DIOR 数据集交叉验证鲁棒性结论

## §8 Phase 4 完成：RSAR CLIP-guided SFOD 同款对照组

> **状态**: ✅ **全部完成** (2026-04-07)
> **实验编号**: E0104 ~ E0110
> **详细记录**: `docs/experiment.md` Phase 4 部分, `docs/semi_supervised_experiments.md` Phase 4 部分
> **结果文件**: `work_dirs/controls/rsar_clip_guided_sfod/results_controls.csv`, `work_dirs/controls/rsar_clip_guided_sfod/results_controls.md`

### 8.1 统一设置

- **Source model**: `configs/experiments/rsar/frontier_026_ocafpn_24ep_oriented_rcnn_rsar.py` + `work_dirs/frontier_026_ocafpn_24ep/latest.pth`
- **Target adaptation data**: `dataset/RSAR/train/images/`（无标注；不允许使用 `test` 图像做校准/自适应）
- **Evaluation**: `dataset/RSAR/test/images/` + 7 个 `dataset/RSAR/corruptions/*/test/images/`
- **统一脚本**: `scripts/exp_rsar_controls.sh`, `scripts/queue_rsar_long_controls.sh`
- **统一随机种子**: `3407`
- **mean 定义**: `clean_test + 7 corruption_test` 共 8 列的算术平均

### 8.2 结果汇总

| 方法 | mean mAP | 备注 |
|---|---|---|
| clean | **0.6863** | 仅 clean test；作为 source model 参考上限 |
| direct | **0.4804** | 不做任何适配；当前最强 control |
| bn | **0.4802** | 仅校准 BN running stats；与 direct 几乎一致 |
| tent | **0.0038** | BN affine + entropy；基本塌陷 |
| shot | **0.0000** | detection-approx-shot；完全塌陷 |
| selftrain | **0.0085** | UnbiasedTeacher(no CGA)；明显退化 |
| cga | **0.0007** | Self-training + SARCLIP CGA；进一步退化 |

### 8.3 关键结论

1. **最强 control 是 direct test**：`SOURCE_CKPT` 直接评测的 `mean=0.4804`，高于所有参数更新式自适应方法。
2. **BN calibration 基本无收益**：`bn=0.4802` 与 `direct=0.4804` 近乎重合，说明在 clean `train/images` 上仅更新 BN 统计不足以改善 corruptions/test 表现。
3. **Tent/SHOT/Self-training/CGA 全部塌陷**：当前协议下，一旦对模型参数做目标域更新，性能显著低于 source model。
4. **CGA 没有挽救 teacher-student 退化**：`cga=0.0007` 甚至低于 `selftrain=0.0085`，问题不只是重打分，而是整个自训练闭环在该 protocol 下失稳。
5. **Phase 3 与 Phase 4 的关键差别在 target data protocol**：Phase 3 使用 `val/images-${corrupt}` 作为干扰域无标注数据，而本控制组严格限定为 clean `train/images`；后者与 corruptions/test 存在显著分布错位。

### 8.4 下一步方向

- [ ] 若继续做 adaptation baseline，应提供与测试干扰分布一致的无标注 target train（或构造 train-side interference split），否则 direct test 就是更合理的强基线。
- [ ] 为 self-training/CGA 增加 teacher 置信度监控、动态 `tau/score_thr` 与 early-stop，避免在 clean-train → corrupt-test 设定下发生整体塌陷。
- [ ] 将 Phase 4 作为后续论文/报告中的“负对照组”，与 Phase 3 的干扰匹配式 SFOD protocol 明确区分。


# Torch 框架下 Claude Code Skill Suite 实现方案

## 一、总体目标

目标是构建一个面向 PyTorch 深度学习工程流程的 Claude Code Skill Suite，将完整流程拆分为 6 个独立 skill 项目：

1. 数据工程 skill  
2. 模型构建 skill  
3. 训练循环 skill  
4. 验证与调优 skill  
5. 推理部署 skill  
6. 工程化扩展 skill  

推荐设计原则：

- 每个 skill 单一职责
- 统一输入输出契约
- 支持串联调用
- 优先生成工程骨架、配置、代码模板
- 把“决策”交给 Claude，把“执行”交给代码与脚本
- 把自动触发类行为放到 hooks/settings，不放 skill 本体

---

## 二、推荐的顶层架构

建议将整个项目组织成一个 skill collection，而不是 6 个完全无关联的仓库。

### 1. 推荐目录

```text
torch-skill-suite/
├─ skills/
│  ├─ torch-data/
│  ├─ torch-model/
│  ├─ torch-train/
│  ├─ torch-eval-tune/
│  ├─ torch-infer-deploy/
│  └─ torch-engineering/
├─ templates/
│  ├─ classification/
│  ├─ detection/
│  ├─ segmentation/
│  ├─ nlp/
│  └─ timeseries/
├─ schemas/
│  ├─ project_spec.schema.json
│  ├─ data_contract.schema.json
│  ├─ model_contract.schema.json
│  └─ deploy_contract.schema.json
├─ examples/
│  ├─ image_cls_cifar10/
│  ├─ bert_text_cls/
│  └─ unet_segmentation/
└─ docs/
```

### 2. 每个 skill 的定位

每个 skill 不直接“训练模型”，而是负责：

- 收集任务上下文
- 分析当前项目状态
- 生成/修改代码
- 校验工程结构
- 输出标准化产物，供下一个 skill 使用

---

## 三、六个 skill 的职责边界

## 1）数据工程 skill

### 目标
自动完成数据接入、清洗、标注映射、Dataset/DataLoader 构建、数据检查。

### 输入
- 任务类型：classification / detection / segmentation / nlp / tabular / timeseries
- 数据源位置：本地目录、CSV、Parquet、JSON、图像目录等
- 标签字段、特征字段、切分比例
- 预处理要求：resize、normalize、tokenize、augment 等

### 输出
- `datasets/` 目录代码
- `transforms/` 或 `preprocess.py`
- `build_dataloader(...)`
- `data_stats.json`
- `label_mapping.json`
- `data_contract.yaml`

### skill 应做的事
- 识别数据类型
- 生成 Dataset 类
- 生成 train/val/test split 逻辑
- 生成 DataLoader 和 collate_fn
- 生成数据 sanity check 脚本
- 检查 class imbalance、缺失值、shape、坏样本

### 不应做的事
- 不负责模型结构设计
- 不负责训练策略

---

## 2）模型构建 skill

### 目标
根据任务目标和数据契约，自动生成模型结构与模块化代码。

### 输入
- `data_contract.yaml`
- 任务类型
- backbone 偏好：resnet / vit / bert / lstm / unet / custom
- 输出头定义：分类头、检测头、分割头等
- 资源约束：轻量、可部署、高精度优先

### 输出
- `models/` 下模型代码
- `build_model(config)`
- 模型配置文件
- 参数量/FLOPs 粗估
- `model_contract.yaml`

### skill 应做的事
- 根据任务选择合适模型模板
- 自动分离 backbone / neck / head
- 生成 forward
- 支持 pretrained 参数
- 输出 shape 检查和 dummy input 测试代码

### 不应做的事
- 不负责训练循环细节
- 不负责超参搜索

---

## 3）训练循环 skill

### 目标
自动生成可运行、可恢复、可监控的训练系统。

### 输入
- `data_contract.yaml`
- `model_contract.yaml`
- 训练目标：epochs、optimizer、scheduler、loss、mixed precision、DDP 等

### 输出
- `train.py`
- `trainer/` 模块
- checkpoint 保存/恢复逻辑
- logging 配置
- `train_config.yaml`

### skill 应做的事
- 生成训练主循环
- 集成 optimizer / scheduler / loss
- AMP、gradient clipping、accumulation
- checkpoint、resume、best model 保存
- 支持单卡/多卡基本模式
- 输出训练日志结构

### 推荐内置能力
- early stopping
- seed 固定
- reproducibility 配置
- tensorboard / wandb 可选接入

---

## 4）验证与调优 skill

### 目标
自动做评估、误差分析、超参调优建议与实验管理。

### 输入
- 已训练 checkpoint
- 验证集/测试集定义
- 指标目标：accuracy / f1 / auc / iou / bleu 等

### 输出
- `evaluate.py`
- `metrics/`
- `error_analysis.ipynb` 或脚本
- `tuning_plan.yaml`
- `experiment_report.md/json`

### skill 应做的事
- 生成评估脚本
- 按任务自动接指标
- 输出 confusion matrix、PR curve、ROC、per-class 指标
- 给出调优建议：
  - 学习率
  - batch size
  - augment
  - regularization
  - 架构调整
- 如果需要，生成 Optuna/Ray Tune 集成模板

### 核心原则
这个 skill 更偏“分析与决策建议”，不是盲目搜索所有参数。

---

## 5）推理部署 skill

### 目标
把训练好的模型自动转为可服务化的推理系统。

### 输入
- checkpoint / exported model
- 部署目标：CLI / REST API / batch inference / TorchServe / ONNX / TensorRT
- 延迟/吞吐/平台约束

### 输出
- `infer.py`
- `serve.py`
- 导出脚本：TorchScript / ONNX
- API 示例
- benchmark 脚本
- `deploy_contract.yaml`

### skill 应做的事
- 封装前处理/后处理
- 生成 batch 和 online inference 接口
- 导出模型
- 生成 FastAPI/TorchServe 基础服务
- 添加健康检查、版本号、模型加载逻辑
- 提供延迟与吞吐测试脚本

### 部署层建议
先做 3 个层级：

1. 本地推理  
2. API 服务化  
3. 导出优化（ONNX/TorchScript）

不要一开始就上 Kubernetes、Triton、TensorRT 全套。

---

## 6）工程化扩展 skill

### 目标
把前 5 个 skill 产出的项目升级成“可维护深度学习工程”。

### 输入
当前项目代码结构

### 输出
- 配置系统统一化
- 测试骨架
- CI 建议
- package 化结构
- registry/plugin 机制
- 实验管理规范

### skill 应做的事
- 重构为标准项目布局
- 引入 Hydra/Pydantic/OmegaConf 之一
- 增加单元测试/冒烟测试
- 增加 lint/format/type check
- 增加模块注册机制
- 支持多任务/多模型扩展
- 生成文档骨架

### 这个 skill 的意义
前 5 个是“能跑”，第 6 个是“能长期演化”。

---

## 四、6 个 skill 之间的协作协议

这是最关键的部分。如果没有统一协议，6 个 skill 会变成 6 个互不兼容的 prompt。

### 建议统一 4 类契约文件

#### 1. `project_spec.yaml`
全局任务定义

```yaml
task_type: image_classification
objective: classify 10 categories
framework: pytorch
input_modality: image
deployment_target: fastapi
constraints:
  latency_ms: 50
  model_size_mb: 200
```

#### 2. `data_contract.yaml`
数据规范

```yaml
input_shape: [3, 224, 224]
num_classes: 10
label_map:
  0: cat
  1: dog
splits:
  train: data/train
  val: data/val
  test: data/test
```

#### 3. `model_contract.yaml`
模型规范

```yaml
backbone: resnet50
head: linear_cls
num_classes: 10
pretrained: true
loss: cross_entropy
```

#### 4. `deploy_contract.yaml`
部署规范

```yaml
format: onnx
service: fastapi
preprocess: imagenet_norm
postprocess: softmax_topk
batch_infer: true
```

### skill 交接顺序

```text
torch-data
  -> data_contract.yaml

torch-model
  -> model_contract.yaml

torch-train
  -> checkpoints + train_config.yaml

torch-eval-tune
  -> experiment_report + tuning_plan.yaml

torch-infer-deploy
  -> serving app + exported artifacts

torch-engineering
  -> repo normalization + quality gates
```

---

## 五、每个 skill 的交互形式建议

建议每个 skill 都支持两种模式：

### 模式 A：从零生成
用户说：

- “为图像分类项目生成数据工程模块”
- “为当前 PyTorch 项目补齐训练循环”

skill 自动扫描仓库并补代码。

### 模式 B：基于契约推进
用户说：

- “基于 data_contract.yaml 生成模型代码”
- “基于 model_contract.yaml 生成 train.py”

这种模式最稳定，适合多 skill 串联。

---

## 六、推荐的 skill 粒度设计

不要把 skill 做成一个超长 prompt。
建议每个 skill 内部再分成固定子能力模板：

### 例如 `torch-train` 内部分解为：
- 任务识别
- loss/optimizer 推荐
- train loop 生成
- checkpoint 设计
- 日志与实验记录
- 分布式/AMP 可选增强

这样更容易维护 prompt，也方便以后把 skill 升级成更细分版本。

---

## 七、实现阶段建议

## Phase 1：先做 MVP

先只支持一个明确场景，建议：

- 图像分类
- 单机单卡
- 标准目录数据集
- FastAPI 部署
- 基础工程化

对应 6 个 skill 的 MVP：

1. 数据：ImageFolder + transforms + DataLoader  
2. 模型：ResNet/EfficientNet 二选一  
3. 训练：标准训练循环 + checkpoint  
4. 评估：accuracy/f1/confusion matrix  
5. 部署：TorchScript/ONNX + FastAPI  
6. 工程化：配置、测试、lint  

这样最容易闭环。

## Phase 2：扩展任务类型

按顺序扩：

1. 文本分类
2. 分割
3. 检测
4. 时间序列
5. 表格数据

不要一开始多模态全上。

## Phase 3：增强自动化能力

加入：

- 自动读取项目现有代码并补全
- 自动发现缺失模块
- 自动重构成标准结构
- 自动生成实验报告
- 自动选择导出格式与部署方式

---

## 八、实现时的关键技术决策

### 1. 配置系统
建议统一用：

- `YAML + Pydantic` 或
- `Hydra + OmegaConf`

如果你偏工程化、可扩展，选 **Hydra**。  
如果你想先轻量可控，选 **YAML + Pydantic**。

### 2. 模板来源
每个 skill 不要从零生成所有代码，最好有：

- classification 模板
- nlp 模板
- segmentation 模板

Claude 更适合做：
- 模板选择
- 模板裁剪
- 参数填充
- 结构修正

而不是每次全量自由生成。

### 3. 评估与部署抽象
建议统一接口：

- `build_dataset(config)`
- `build_model(config)`
- `build_trainer(config)`
- `build_evaluator(config)`
- `build_predictor(config)`

这样 6 个 skill 更容易共用上下文。

---

## 九、你这个 skill suite 最终应支持的典型工作流

### 工作流 1：从零起项目

```text
/torch-data
/torch-model
/torch-train
/torch-eval-tune
/torch-infer-deploy
/torch-engineering
```

### 工作流 2：补全现有项目

```text
/torch-train
/torch-eval-tune
/torch-engineering
```

### 工作流 3：只做部署

```text
/torch-infer-deploy
```

---

## 十、建议的验收标准

每个 skill 都要有可验证产物。

### torch-data
- 能生成 Dataset/DataLoader
- 能跑通一个 batch
- 能输出数据统计

### torch-model
- 能实例化模型
- dummy input forward 成功

### torch-train
- 能完整训练至少 1 epoch
- 能保存 checkpoint

### torch-eval-tune
- 能对 checkpoint 做评估
- 能输出结构化报告

### torch-infer-deploy
- 能本地推理
- 能启动 API
- 能导出 ONNX/TorchScript 至少一种

### torch-engineering
- 能通过 lint/test
- 工程目录清晰可维护

---

## 十一、建议的落地顺序

最优顺序：

1. 先定义统一契约文件
2. 再做 `torch-data`
3. 再做 `torch-model`
4. 再做 `torch-train`
5. 再做 `torch-eval-tune`
6. 再做 `torch-infer-deploy`
7. 最后做 `torch-engineering`

原因：前 5 个解决业务闭环，第 6 个负责长期演化。

---

## 十二、最终建议

如果你想把这个项目做成“好用的 Claude Code skill 套件”，核心不是 prompt 写多长，而是这三件事：

1. 统一契约
2. 模板化生成
3. 可验证产物

只要这三点做稳，6 个 skill 就能独立、可组合、可扩展。

---

## 参考资料

- Claude Code Skills: https://docs.anthropic.com/en/docs/claude-code/skills
- Claude Code Settings: https://docs.anthropic.com/en/docs/claude-code/settings

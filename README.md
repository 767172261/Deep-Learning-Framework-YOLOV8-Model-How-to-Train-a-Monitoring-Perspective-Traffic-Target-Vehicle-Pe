监控视角交通目标车辆行人检测数据集 YOLO格式｜高清路口实时监控｜行人/车辆/多目标
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/11533ca8344b4597b3778d8ff96c4801.png)

✅ 直接来自路口监控摄像头，非网络爬取、非二次加工
✅ YOLO标准格式、

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8acc34aec644a4d89204f5107987291.png)


### 📊 数据集套餐价格表

| 等级       | 数据量     |     |              | 核心权益 |
|----------------|------------|------------------|-------------------|----------|
| 【1】 | 5,000 张   |    |         | 适合快速验证数据质量与模型适配性。 |
| 【2】 | 50,000 张  |   | | 满足课程设计、毕业设计或小型实验项目需求。 |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a01fc1b6c3ab44bbbcad71d52b231157.png)


---1

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1536a8f976b04f3fa4f819a1225a7312.png)


 **基于 YOLOv8 + 车辆行人检测数据集（2类）** 的完整训练流程与代码，适用于您提供的 **YOLO格式数据集**，包含 **车辆、行人** 两类目标。

---

## 📁 一、项目结构说明

```bash
VehiclePersonDetection/
├── datasets/                     # 数据集目录（YOLO格式）
│   ├── images/
│   │   ├── train/                # 训练图像
│   │   └── val/                  # 验证图像
│   └── labels/
│       ├── train/                # 训练标签（.txt）
│       └── val/                  # 验证标签（.txt）
├── vehicle_person.yaml           # 数据配置文件
├── train.py                      # 模型训练脚本
└── runs/                         # 训练输出目录（自动创建）
```

---

## 🔧 二、1. 安装依赖

```bash
pip install ultralytics opencv-python numpy matplotlib tqdm
```

> 推荐使用 Python 3.8~3.11，PyTorch ≥ 1.8

---

## 🗃️ 三、2. 数据集格式要求（YOLO 标准）

每个图像对应一个 `.txt` 文件，例如：

- `datasets/images/train/001.jpg`
- `datasets/labels/train/001.txt`

`.txt` 内容格式：
```
<class_id> <x_center> <y_center> <width> <height>
```

| 类别 | class_id | 示例 |
|------|----------|------|
| 车辆 | 0        | `0 0.45 0.6 0.2 0.1` |
| 行人 | 1        | `1 0.7 0.5 0.1 0.2` |

> ✅ 所有坐标为归一化值（[0,1]），由 LabelImg 自动转换  
> ✅ 使用 [LabelImg](https://github.com/HumanSignal/labelImg) 进行标注并导出 YOLO 格式

---

## 📂 四、3. 创建 `vehicle_person.yaml` 配置文件

```yaml
# vehicle_person.yaml
train: ./datasets/images/train
val: ./datasets/images/val

nc: 2
names: ['车辆', '行人']
```

> ✅ `nc=2` 表示两个类别  
> ✅ `names` 顺序必须与标注时的 class_id 对应（class_id=0 → 车辆，class_id=1 → 行人）

---

## 🚀 五、4. 训练代码（`train.py`）

```python
# train.py
from ultralytics import YOLO
import os

def main():
    # 确保输出目录存在
    os.makedirs("runs", exist_ok=True)

    # 加载预训练模型（YOLOv8s）
    model = YOLO('yolov8s.pt')  # 支持 yolov8n, yolov8m, yolov8l 等

    # 开始训练
    results = model.train(
        data='vehicle_person.yaml',     # 数据配置文件
        epochs=100,                    # 训练轮数
        imgsz=640,                     # 输入图像尺寸
        batch=16,                      # 批大小（根据显存调整：8/16/32）
        name='vehicle_person_train',   # 实验名称
        optimizer='AdamW',             # 优化器
        lr0=0.001,                     # 初始学习率
        lrf=0.01,                      # 最终学习率 = lr0 * lrf
        patience=15,                   # 早停：若15轮无提升则停止
        save=True,                     # 保存最佳和最后模型
        save_period=10,                # 每10轮保存一次
        device=0,                      # GPU设备ID（0,1,... 或 'cpu'）
        workers=4,                     # 数据加载线程数
        project='runs',                # 输出根目录
        exist_ok=False,                # 若实验名已存在是否覆盖
        verbose=True                   # 打印详细日志
    )

    print("✅ 训练完成！最佳模型路径：", results.save_dir / "weights/best.pt")

if __name__ == '__main__':
    main()
```

---

## ▶️ 六、5. 启动训练

在终端执行：

```bash
python train.py
```

> ⏱️ 训练时间参考（RTX 3060）：
> - batch=16, imgsz=640 → 约 1.0 小时（100 epochs）

---

## 📈 七、6. 训练过程监控

- 日志自动保存在：`runs/detect/vehicle_person_train/`
- 可视化指标：
  - `results.png`：loss、mAP@0.5、precision、recall 曲线
  - `confusion_matrix.png`：混淆矩阵
  - `labels.jpg`：标签分布热力图
- 使用 TensorBoard（可选）：
  ```bash
  tensorboard --logdir runs
  ```

---

## ✅ 八、7. 验证与测试（可选）

训练完成后，可单独验证：

```python
from ultralytics import YOLO

model = YOLO('runs/detect/vehicle_person_train/weights/best.pt')
metrics = model.val(data='vehicle_person.yaml')
print(metrics.box.map)        # mAP@0.5
print(metrics.box.map50)      # mAP@0.5
print(metrics.box.precision)  # Precision
print(metrics.box.recall)     # Recall
```

---

## 📤 九、8. 导出模型（用于部署）

```python
model = YOLO('best.pt')
model.export(format='onnx')      # 导出 ONNX
model.export(format='engine')    # 导出 TensorRT（需 CUDA）
model.export(format='openvino')  # 导出 OpenVINO
```

---

## 💡 十、调优建议（针对车辆行人检测）

| 问题 | 解决方案 |
|------|--------|
| 车辆小目标漏检 | 增加 `imgsz=1280`、使用 `yolov8m` 或 `yolov8l` |
| 行人遮挡误检 | 降低 `conf_threshold=0.3`、增加数据增强 |
| 类别不平衡 | 在 `train.py` 中添加 `class_weights=[1.0, 2.0]` |
| 显存不足 | 减小 `batch`（如 8）、使用 `yolov8n`、设置 `imgsz=416` |

---

## 📎 附：快速检查数据集是否合规

```python
# check_dataset.py
import os

def check_dataset(root='datasets'):
    for split in ['train', 'val']:
        img_dir = os.path.join(root, 'images', split)
        lbl_dir = os.path.join(root, 'labels', split)
        imgs = set(os.listdir(img_dir))
        lbls = set(f.replace('.txt', '.jpg') for f in os.listdir(lbl_dir))
        missing = imgs - lbls
        extra = lbls - imgs
        if missing:
            print(f"⚠️ {split}: 缺少标签的图片: {missing}")
        if extra:
            print(f"⚠️ {split}: 多余的标签: {extra}")
    print("✅ 数据集结构校验完成！")

if __name__ == '__main__':
    check_dataset()
```

---



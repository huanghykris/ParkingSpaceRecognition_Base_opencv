# 停车场车位识别

#### 任务目标

1. 检测整个停车场当中，当前一共有多少辆车，一共有多少个空余的车位
2. 把空余的停车位标识出来，这样用户停车的时候，就可以直接去空余的停车位处， 为停车节省了很多时间

#### 系统架构

本系统由两大核心模块组成：

1. **图像处理模块** - 识别和标注停车位
2. **CNN分类模块** - 判断车位占用状态

#### 模块一：图像处理模块 (`img_process`)

##### 功能概述

本模块是一个基于图像处理的停车场车位检测系统，能够自动识别图像中的停车位，并输出每个车位的坐标信息。系统通过多步骤图像处理技术实现车位检测，最终生成可用于CNN分类的车位数据。

##### 主要功能

- 颜色空间过滤（白/黄）
- 灰度转换
- 边缘检测
- 感兴趣区域(ROI)选择
- 直线检测（霍夫变换）
- 车位矩形识别
- 车位坐标标注与保存

##### 处理流程

1. 颜色过滤 → 2. 灰度转换 → 3. 边缘检测 → 4. ROI选择 → 5. 直线检测 → 6. 车位识别 → 7. 结果保存

##### 输出

- `spot_dict.pickle`: 车位坐标字典
- `cnn_images/`: 各车位截图

#### 模块二：CNN分类模块

##### 功能概述

使用迁移学习(VGG16)训练二分类模型，判断车位是否被占用。

##### 关键配置

```python
# 数据配置
img_width, img_height = 48, 48  # 输入图像尺寸
batch_size = 32                 # 批处理大小
epochs = 15                     # 训练轮数
num_classes = 2                 # 分类数(空闲/占用)

# 模型配置
base_model = VGG16(weights='imagenet', include_top=False)  # 使用预训练VGG16
optimizer = Adam(learning_rate=0.0001, ema_momentum=0.9)  # 优化器配置
loss = "categorical_crossentropy"                         # 损失函数
```

##### 数据增强策略

```python
# 训练数据增强
horizontal_flip = True    # 水平翻转
zoom_range = 0.1          # 随机缩放
shift_range = 0.1         # 平移增强
rotation_range=5          # 旋转增强
```

##### 回调函数

- **ModelCheckpoint**: 保存最佳模型(`car1.model.keras`)
- **EarlyStopping**: 当验证准确率不再提升时提前停止

##### 完整使用流程

1. **准备数据**

   ```txt
   train_data/
   ├── train/
   │   ├── empty/    # 空车位图像
   │   └── occupied/ # 占用车位图像
   └── test/
       ├── empty/
       └── occupied/
   ```

2. **运行图像处理**

   ```python
   spot_dict = img_process(test_images, park)
   ```

3. **训练分类模型**

   ```python
   python train.py  # 包含上述CNN训练代码
   ```

4. **部署使用**

   - 加载训练好的模型(`car1.model.keras`)
   - 使用`spot_dict`中的坐标实时检测车位状态

#### 文件结构

```
project/
├── img_processing.py    # 图像处理模块
├── train.py            # CNN训练代码
├── car1.model.keras    # 训练好的模型
├── spot_dict.pickle    # 车位坐标数据
├── train_data/         # 训练数据集
└── cnn_images/         # 车位截图
```

#### 性能指标

- 输入图像尺寸: 48x48像素
- 基础模型: VGG16(前10层冻结)
- 优化器: Adam with EMA
- 典型训练时间: 约15个epoch

#### 注意事项

1. 训练数据应包含各种光照条件下的车位图像
2. 建议使用GPU加速训练过程
3. 图像处理模块和分类模块输入尺寸需保持一致
4. 定期验证模型在实际场景中的表现

#### 扩展建议

1. 尝试其他基础模型(如ResNet, EfficientNet)
2. 添加数据平衡处理
3. 实现端到端的实时检测系统
4. 添加模型量化功能便于边缘设备部署

#### 版本信息

v1.0 - 初始版本

- 基础图像处理流水线
- VGG16分类模型
- 基础数据增强策略
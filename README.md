# ABQnn - Abaqus Neural Network Integration

本项目实现了在Abaqus用户材料子程序(UMAT)中调用PyTorch Script模型的功能，为有限元分析提供机器学习驱动的材料本构模型。

## 项目结构

```
ABQnn/
├── UMAT_allmodels.for          # Fortran UMAT子程序主文件
├── src/
│   ├── UMAT_auxlib.cpp         # C++辅助库，负责DLL加载
│   └── UMAT_pt_caller.cpp      # C++ PyTorch调用器，执行神经网络推理
└── utils/
    ├── export_models.py        # 从Abaqus ODB导出数据的Python脚本
    └── gen_material_models.py  # 生成多个材料模型作业的Python脚本
```

## 核心组件

### 1. Fortran UMAT子程序 (`UMAT_allmodels.for`)
- 主要的Abaqus用户材料子程序接口
- 通过iso_c_binding绑定调用C++函数
- 根据材料名称自动加载对应的PyTorch Script模型文件
- 处理变形梯度张量、应力和刚度矩阵的传递

### 2. C++组件

#### `UMAT_auxlib.cpp` - 动态库加载器
- 负责加载PyTorch相关的DLL文件
- 实现线程安全的初始化机制
- 提供错误处理和调试输出功能

#### `UMAT_pt_caller.cpp` - PyTorch模型调用器
- 实现PyTorch Script模型的加载和推理
- 使用线程安全的模型缓存机制
- 处理输入/输出张量的格式转换
- 支持以下输入输出：
  - 输入：变形梯度张量 F [3×3]
  - 输出：应变能密度 ψ，Cauchy应力 [6]，刚度矩阵 [6×6]

### 3. Python工具脚本

#### `export_models.py` - 数据导出工具
从Abaqus ODB文件中导出仿真数据用于神经网络训练。

**使用方法:**
```bash
abaqus python export_models.py <jobname_pattern> <output_dir>
```

**功能:**
- 批量处理符合模式的ODB文件
- 导出节点坐标、网格连接、应力、应变和位移等数据
- 输出NumPy格式的数组文件和JSON格式的网格文件

#### `gen_material_models.py` - 批量作业生成器
在Abaqus CAE中批量创建使用不同PyTorch模型的材料和作业。

**使用方法:**
```bash
# 在Abaqus CAE中运行
abaqus python gen_material_models.py
```

**功能:**
- 交互式输入材料和作业参数
- 批量创建材料定义
- 生成作业输入文件或直接提交计算

## 安装和使用

### 环境要求
- Abaqus 2020或更高版本
- PyTorch C++ API (LibTorch)
- Visual Studio编译器（用于编译C++组件）
- Python环境（Abaqus内置或独立）

### 编译步骤
1. 确保PyTorch C++库已正确安装
2. 编译C++组件为动态链接库
3. 将UMAT Fortran文件和相关DLL放置在Abaqus可访问的路径；修改.cpp文件中需要用户修改的若干路径。
4. 将makefile中的Abaqus编译所用的具体lib文件路径修改为真实路径；并修改对应的与编译平台相关的Abaqus控制文件（例：Windows平台的\SMA\site\win86_64.env文件），只需要再link_*部分加入所编译的.lib文件即可。

### 使用流程

1. **准备PyTorch模型**
   - 将训练好的PyTorch模型导出为TorchScript格式(.pt文件)
   - 模型应接受变形梯度张量并输出应力和刚度矩阵

2. **设置材料**
   - 在Abaqus CAE中创建User Material
   - 材料名称应与对应的.pt文件名匹配

3. **运行分析**
   - 在作业设置中指定UMAT子程序路径
   - 提交作业进行计算

4. **数据处理**
   - 使用`export_models.py`从结果ODB文件中提取数据
   - 数据可用于进一步的分析或模型训练

## 模型接口规范

PyTorch模型应遵循以下输入输出规范：

**输入:**
- 变形梯度张量 F: `torch.Tensor [ 3, 3]`

**输出:**
- 应变能密度 ψ: `torch.Tensor [1]`,或者`double`值。
- Cauchy应力: `torch.Tensor [6]`
- 材料切线刚度矩阵（注：参考Abaqus UMAT子程序相关文档；对应Jaumann客观率）: `torch.Tensor [batch_size, 6, 6]`
- 以上应力、刚度矩阵均采用与Abaqus相同的指标顺序约定，即：0对应1方向正应力/应变，3对应1-2方向剪切应力、应变

## 注意事项

- 确保PyTorch模型文件路径正确配置
- C++组件需要正确链接LibTorch库
- 注意线程安全，特别是在并行计算环境中
- 调试模式可通过定义`DEBUG`宏来启用

## 故障排除

- 如果遇到DLL加载错误，检查LibTorch库路径和依赖项
- 确认PyTorch模型文件格式正确且可被加载
- 检查Abaqus用户子程序的编译环境配置

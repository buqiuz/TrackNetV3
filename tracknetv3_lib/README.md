# tracknetv3_lib

TrackNetV3 推理库（自包含，可整体拷贝到其他项目）。从本仓库 `predict.py` 忠实移植，已在升级依赖下（torch 2.5 / numpy 2.4 / opencv 5 / pandas 3 / Pillow 12）与原版输出做过逐帧一致性验证。

## 使用

```python
from tracknetv3_lib import TrackNetV3

model = TrackNetV3('ckpts/TrackNet_best.pt', 'ckpts/InpaintNet_best.pt')  # 只需 TrackNet 时第二个参数传 None
df = model.predict_video('video.mp4')          # -> DataFrame[Frame, Visibility, X, Y]（原始分辨率坐标）
df = model.predict_video('video.mp4', save_dir='pred_result', output_video=True)  # 同时落盘 CSV + 可视化视频
```

命令行示例：

```bash
python -m tracknetv3_lib.example --video_file test.mp4 --output_video
```

## API

`TrackNetV3(tracknet_file, inpaintnet_file=None, device=None)`

- `seq_len` / `bg_mode` 自动从 checkpoint 的 `param_dict` 读取（官方权重为 seq_len=8, bg_mode='concat'），不要写死。
- `device` 缺省自动选择 cuda → mps → cpu。

`predict_video(video_file, eval_mode='weight', large_video=False, batch_size=16, max_sample_num=1800, video_range=None, output_video=False, traj_len=8, save_dir=None, verbose=True)`

- `eval_mode='weight'`（默认）：滑窗时序集成，与原版 `predict.py` 默认行为一致，精度最佳。
- `eval_mode='nonoverlap'`：不重叠采样，速度优先。
- `large_video=True`：流式读视频，长视频不爆内存（较慢）；可配合 `video_range=(start_s, end_s)` / `max_sample_num` 控制背景估计采样。
- 输出坐标已乘 `img_scaler` 还原到原视频分辨率；`Visibility=0` 表示该帧未检出（坐标为 0）。

## 迁移到其他项目

1. 拷贝整个 `tracknetv3_lib/` 目录（纯 Python 包，无仓库内其他依赖）。
2. 准备环境：Python ≥ 3.11，安装 `tracknetv3_lib/requirements.txt`（内含经一致性验证的精确版本组合及 torch CUDA 安装说明；更换版本后需重新用 `verify.py` 验证）。
3. 权重文件（`ckpts/*.pt`）随包分发需遵守仓库 MIT 许可。

## 一致性验证

```bash
# 1. 原版生成 baseline
python predict.py --video_file v.mp4 --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir pred_baseline
# 2. 新库生成结果
python -m tracknetv3_lib.example --video_file v.mp4 --save_dir pred_lib
# 3. 比对（应逐行一致）
python -m tracknetv3_lib.verify --baseline pred_baseline/v_ball.csv --candidate pred_lib/v_ball.csv
```

已验证：GPU 上 `weight` / `nonoverlap` / `large_video` 三种模式输出与原版完全一致；CPU 与 GPU 有 ±1 像素级浮点差异（后端固有，非移植引入）。

## 与原版的实现差异

- 去掉了 `pycocotools` 依赖（原 `test.py` 为导入 3 个工具函数而引入整个评估栈）。
- 不依赖 `data_dir` / `dataset.py` 的磁盘数据集结构与缓存；帧处理逻辑内联在 `_preprocess.py`。
- 去掉 DataLoader 多进程（顺序批处理等价，结果一致）。
- `torch.load` 兼容 torch≥2.6 的 `weights_only` 默认值变更（优先安全加载，失败自动回退）。

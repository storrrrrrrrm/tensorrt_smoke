# tensorrt_smoke

3D检测模型SMOKE:https://github.com/lzccccc/SMOKE 的tensorrt推理代码.

tested on orin(jetpack5.0)
- cuda11.4
- tensorrt8.4.1.5
- opencv4.5

## usage
```
mkdir build
cd build 
cmake ../
make
./smoke_trt #第一次运行该程序时,将smoke_dla34.onnx转换为smoke_dla34.engine
./smoke_trt #使用smoke_dla34.engine做推理
```

## performance
单纯模型推理时间:compute:54ms,fps:18
全流程时间:one frame:469ms,fps:2

## Acknowledgement
[TensorRT-SMOKE](https://github.com/Yibin122/TensorRT-SMOKE)
[mmdetection3d](https://github.com/open-mmlab/mmdetection3d)

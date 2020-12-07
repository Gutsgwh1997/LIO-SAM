## 系统结构

<p align='center'>
    <img src="./config/doc/system.png" alt="drawing" width="800"/>
</p>

LIO_SAM维护了两个因子图，运行速度比实时快4倍!
- "mapOptimization.cpp"模块优化雷达里程计与GPS因子，此因子图在整个运行过程中持续维护。
- "imuPreintegration.cpp"模块优化IMU与雷达里程计因子并且估计IMU的bias。此因子图定期复位，保证IMU频率下的实时位姿估计。

## 激光雷达格式要求

激光雷达在"imageProjection.cpp"模块去畸变，它对点云格式有这两点需求：

"""高精度控制测量卫星可见性规划后端。

模块划分：
    timeutils   时间表示与换算
    orbit       历书（开普勒根数）与 ECEF 轨道计算
    coords      WGS84 坐标变换（大地坐标/ECEF/ENU）
    mask        点位与遮挡轮廓
    dop         精度因子（GDOP/PDOP/HDOP/VDOP/TDOP）
    cache       跨规划共享的 LRU 计算缓存
    visibility  可见性、遮挡判定与单时刻/批量评估
    search      时间轴时段搜索
    planning    单点规划编排
    repository  MongoDB 持久化与版本号
    diff        规划结果差异对比
    scheduler   异步作业调度、取消与重规划取代
    api         Flask 接口
"""

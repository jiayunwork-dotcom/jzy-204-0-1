# GNSS GDOP 批量规划后端

Python 3.12 + Flask + MongoDB 7 + NumPy 实现。轨道传播、坐标变换、遮挡判断、DOP 和时段搜索均在本项目内实现，不依赖卫星导航或天文库。

## 模块划分

- `app/gnss_service/orbit.py`：历书开普勒轨道、Kepler 方程、地球自转、ECEF 卫星位置。
- `app/gnss_service/coordinates.py`：WGS84 经纬高转 ECEF，ECEF 转 ENU，方位角/高度角。
- `app/gnss_service/visibility.py`：方位角-遮挡高度角轮廓，循环线性插值，截止高度角。
- `app/gnss_service/dop.py`：用 ENU 方向单位向量构造 `[E N U 1]` 设计矩阵，输出 G/P/H/V/T DOP。
- `app/gnss_service/search.py`：固定采样网格、整数秒二分细化、最短时段过滤和最小值搜索。
- `app/gnss_service/computation.py`：共享线程安全缓存；同一历书版本、点位版本、时刻不重复传播和 DOP 计算。
- `app/gnss_service/versions.py`：历书版本与点位版本对象构造。
- `app/gnss_service/scheduler.py`：异步线程池作业、状态、进度、协作式取消和让位。
- `app/gnss_service/diff.py`：新旧时段匹配和消失/新增/缩短/延长/平移差异。
- `app/gnss_service/service.py`：领域编排、历书/遮挡触发重算、查询入口。
- `app/gnss_service/repositories.py`：MongoDB 仓储；测试可替换为内存仓储。
- `app/gnss_service/api.py`：Flask HTTP 接口。

## 轨道模型

历书字段：

- `semi_major_axis`（米）或 `sqrt_semi_major_axis`（平方米方根，二选一）
- `eccentricity`
- `inclination`、`raan`、`argument_of_perigee`、`mean_anomaly`：角度
- `raan_rate`：度/日，转换为弧度/秒
- `reference_time`：ISO-8601 时间

模型使用二体平均运动 `n=sqrt(mu/a^3)`，迭代求解平近点角对应的偏近点角，在轨道平面计算位置，经近地点幅角、倾角、升交点旋转得到惯性坐标，再按地球自转角旋转到 ECEF。未引入 J2 长期项、章动、极移和岁差；这是题目给定开普勒历书的自包含模型。

## 遮挡与 DOP

遮挡轮廓按 `[[方位角, 遮挡高度角], ...]` 提供，方位角范围 `[0,360]`，不可重复；`360` 等价于 `0`，两者同时出现会被判重。未给轮廓方向通过首尾循环线性插值连接；无轮廓时使用统一截止高度角。

可见条件：

```text
卫星高度角 >= 该方位遮挡高度角
```

至少 4 颗可见卫星时构建设计矩阵。若矩阵奇异仍判定该时刻不可用。恒等式由协方差阵迹直接保证：

```text
GDOP² = PDOP² + TDOP²
PDOP² = HDOP² + VDOP²
```

## 时间搜索与漏检上界

采用“固定步长网格 + 边界整数秒二分细化”：

1. 从范围起点开始按 `h` 采样，终点强制采样。
2. 找到连续 `available=True` 的采样段。
3. 在段两侧的一真一假整数秒之间二分，得到起止时刻。
4. 在候选时段内逐整数秒读取 GDOP，定位最小值及出现时刻。
5. 剔除连续时长小于 `min_duration_seconds` 的候选。
6. 卫星升起/落下造成的跳变按整数秒状态处理；整数秒扫描不会跳过段内跳变。

默认步长为：

```text
h = min(请求步长或10秒, max(1秒, 最短连续时长 / 2))
```

**漏检性质：** 任意长度严格大于 `h` 的连续可用区间必然包含内部网格点，因此不会漏检。所有合格区间时长至少为 `min_duration_seconds`，而 `h <= min_duration/2`，所以满足最短时长要求的区间不会漏。可能漏掉的是长度不超过 `h` 的孤立尖峰；最短漏检时长上界为 `h`（更精确地说，严格小于等于一个网格间隔的区间可完全落在两网格点之间）。少于 4 星、阈值穿越或遮挡升降导致的短暂亚网格“洞”也可能把相邻区间合并，持续时间上界同样为 `h`。测试 `tests/test_search.py` 同时包含：

- 20 秒、最短合格时长 15 秒的构造尖峰（网格 `h=7.5s`），可被发现；
- 5 秒尖峰、10 秒网格的漏检边界示例。

## 作业、版本和差异

- 提交规划返回 `202` 和作业号，可查询进度和取消。
- 取消作业不会替换计划结果；只有完整计算成功后才写入结果。
- 历书导入生成新版本，并对结束时间仍在未来的计划发起重算。
- 新历书到来时，未完成的旧重算作业标记为 `superseded`；计算线程通过作业状态取消。最终只保留最新结果和一份上一版历史。
- 点位遮挡更新生成点位新版本，只触发引用该点位的未来计划。
- 差异按最大交叠对新旧时段进行一一匹配，输出 `disappeared`、`added`、`changed`、`unchanged`；变化标签包括 `shortened`、`lengthened`、`shifted`。
- 计算缓存键包含历书版本和点位版本；两个计划引用同一点位时共享同一点位、卫星位置和瞬时 DOP，结果与独立计算逐值一致。

## HTTP 接口

- `POST /api/almanacs`：导入历书。
- `GET /api/almanacs`、`GET /api/almanacs/{id}`：版本查询。
- `PUT /api/points/{id}`：创建或更新点位及遮挡轮廓。
- `GET /api/points`、`GET /api/points/{id}`。
- `GET /api/query?almanac_version_id=...&point_id=...&time=...`：单时刻可见卫星和 DOP。
- `POST /api/plans`：提交规划作业。
- `GET /api/plans/{id}`：计划与活动作业号。
- `GET /api/jobs/{id}`：状态和进度。
- `POST /api/jobs/{id}/cancel`：取消。
- `GET /api/plans/{id}/results`：最新结果；`?history=1` 取一份历史。
- `GET /api/plans/{id}/diff`：历史到最新的差异。

规划请求示例：

```json
{
  "point_ids": ["P1", "P2"],
  "start": "2026-10-10T00:00:00Z",
  "end": "2026-10-11T00:00:00Z",
  "gdop_threshold": 6.0,
  "min_duration_seconds": 1800,
  "step_seconds": 600,
  "almanac_version_id": "alm_..."
}
```

## 拒收规则

- 偏心率不在 `[0,1)`。
- 半长轴不为正；平方根形式也必须为正。
- 纬度越界。
- 遮挡方位角不在 `[0,360]` 或重复。
- GDOP 阈值不为正。
- 最短连续时长为负。
- 日期范围结束早于开始。

## 运行

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`，MongoDB 使用 `mongo:7`，数据库名为 `gnss_planning`。

本地测试：

```bash
pip install -r requirements.txt
pytest
```

测试覆盖题设要求：参考几何、平方和关系、加星不变大、抬高遮挡时段不增、短时尖峰、边界秒级精度、历书更新差异、连续两版历书作业取代、取消无部分结果、共享/独立计算一致以及所有拒收规则。

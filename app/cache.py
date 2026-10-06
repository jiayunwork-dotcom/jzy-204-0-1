"""跨规划共享的可见性 LRU 缓存（带在途键合并）。

同一点位可能被多份规划引用；同一时刻 (历书版本, 点位版本, 卫星, 秒)
的高度角/方位角只需算一次。缓存是进程内、线程安全的 LRU。

并发语义：当两个线程同时请求同一个尚未缓存的键时，只有一个线程执行
轨道计算，另一个线程等待并复用结果（in-flight 合并），因此"同一点位
同时被两份规划引用"时同一时刻绝不重复计算。计算结果只依赖根数与点位
版本，所以共享结果与单独计算逐位一致。
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np


class SharedSkyCache:
    """以 (alm_version, point_version, sat_id, epoch_second) 为键。"""

    def __init__(self, maxsize: int = 200_000):
        self.maxsize = maxsize
        self._data: "OrderedDict[tuple, tuple[float,float]]" = OrderedDict()
        self._lock = threading.Lock()
        # 每个在途键一个事件：第一个请求者负责计算并 set 结果
        self._inflight: dict[tuple, dict] = {}
        self.hits = 0
        self.misses = 0
        self._compute_calls = 0
        self._computed_keys: set[tuple] = set()

    def _key(self, alm_version, point_version, sat_id, second):
        return (alm_version, point_version, sat_id, int(second))

    def get_or_compute(self, alm_version, point_version, sat_id, seconds,
                       computer):
        """批量获取；未命中的整数秒调用 ``computer(seconds_array)`` 计算。

        返回 shape=(T,2) 的 (az, el) 数组（弧度）。``computer`` 只接收
        真正需要新计算的去重后秒数组，且每个键最多被计算一次
        （跨线程在途键合并，等待者阻塞到计算完成）。
        """
        seconds = np.asarray(seconds, dtype=np.int64)
        n = seconds.shape[0]
        result = np.empty((n, 2), dtype=float)

        # 本批次内每个唯一秒只保留第一个位置；其余位置在回填时复制
        first_pos: dict[int, int] = {}
        for pos, sec in enumerate(seconds.tolist()):
            first_pos.setdefault(sec, pos)

        # 第一段：锁内分类——命中 / 本线程负责计算 / 等待别的线程
        my_compute_secs: list[int] = []
        waiters: list[tuple[int, threading.Event, int]] = []
        with self._lock:
            for sec, pos in first_pos.items():
                key = self._key(alm_version, point_version, sat_id, sec)
                hit = self._data.get(key)
                if hit is not None:
                    self._data.move_to_end(key)
                    self.hits += 1
                    result[pos] = hit
                    continue
                flight = self._inflight.get(key)
                if flight is None:
                    flight = {"event": threading.Event()}
                    self._inflight[key] = flight
                    my_compute_secs.append(sec)
                else:
                    waiters.append((pos, flight["event"], sec))
                self.misses += 1

        # 第二段：本线程负责的键一次性计算
        if my_compute_secs:
            sec_arr = np.array(sorted(my_compute_secs), dtype=np.int64)
            raw = np.asarray(computer(sec_arr), dtype=float)
            # computer 必须返回 (N,2) 的 (az,el)；常见错误是返回
            # (2,N)（例如 np.asarray((az,el))），这里显式纠正/报错
            if raw.shape == (2, sec_arr.size):
                raw = raw.T
            values = raw.reshape(-1, 2)
            if values.shape[0] != sec_arr.size:
                raise ValueError(
                    f"computer 返回行数 {values.shape[0]} 与请求秒数 "
                    f"{sec_arr.size} 不一致")
            computed: dict[int, tuple[float, float]] = {}
            with self._lock:
                for sec, row in zip(sec_arr.tolist(), values):
                    key = self._key(alm_version, point_version, sat_id, sec)
                    computed[sec] = (float(row[0]), float(row[1]))
                    # 只有登记在途者才写入（防御性：以已有值为准）
                    if key in self._inflight and key not in self._data:
                        self._data[key] = computed[sec]
                        self._data.move_to_end(key)
                        self._computed_keys.add(key)
                while len(self._data) > self.maxsize:
                    self._data.popitem(last=False)
                self._compute_calls += int(sec_arr.size)
            # 先填回本批次结果，再移出在途表，最后唤醒
            for sec, value in computed.items():
                result[first_pos[sec]] = value
            with self._lock:
                events_to_set = []
                for sec in my_compute_secs:
                    key = self._key(alm_version, point_version, sat_id, sec)
                    flight = self._inflight.pop(key, None)
                    if flight is not None:
                        events_to_set.append(flight["event"])
            for ev in events_to_set:
                ev.set()

        # 第三段：等待其它线程负责的键（不含本线程自己计算的）
        for pos, event, sec in waiters:
            if sec in my_compute_secs:
                continue
            event.wait(timeout=30)
            key = self._key(alm_version, point_version, sat_id, sec)
            with self._lock:
                hit = self._data.get(key)
            if hit is None:
                # 理论上不会发生；兜底自行计算保证正确性
                sec_arr = np.array([sec], dtype=np.int64)
                raw = np.asarray(
                    computer(sec_arr.astype(float)), dtype=float)
                if raw.shape == (2, 1):
                    raw = raw.T
                val = raw.reshape(2)
                result[pos] = val
                continue
            result[pos] = hit

        # 复制本批次内重复秒的结果到其余位置
        for pos, sec in enumerate(seconds.tolist()):
            fp = first_pos[sec]
            if fp != pos:
                result[pos] = result[fp]
        return result

    @property
    def compute_calls(self) -> int:
        with self._lock:
            return self._compute_calls

    @property
    def computed_keys(self) -> set[tuple]:
        """历史上真正触发过轨道计算的键（不受 LRU 淘汰影响）。"""
        with self._lock:
            return set(self._computed_keys)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._inflight.clear()
            self._computed_keys.clear()
            self.hits = self.misses = self._compute_calls = 0

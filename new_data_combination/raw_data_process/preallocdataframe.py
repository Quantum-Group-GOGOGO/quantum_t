import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Sequence
from env import *

class PreallocDataFrame:
    """
    一个包装类，底层预分配固定行数内存，高频合并小 DataFrame 时只是就地写入，
    避免每次都重新分配和拷贝大块内存。
    """
    def __init__(self, initial_df: pd.DataFrame, capacity: int = 10000):
        self.buff_size=10000
        # 确保 initial_df 已按索引升序
        initial_df = initial_df.sort_index()
        self.columns = list(initial_df.columns)
        self.capacity = max(capacity, len(initial_df)+10000)
        # 底层数据与索引数组
        self._data = np.empty((self.capacity, len(self.columns)), dtype=float)
        self._data[:] = np.nan
        self._index = np.empty(self.capacity, dtype='datetime64[ns]')
        self._index[:] = np.datetime64('NaT')

        # 指针：已使用长度
        self._ptr = len(initial_df)
        # 写入初始数据
        self._data[:self._ptr, :] = initial_df.values
        self._index[:self._ptr] = initial_df.index.values

    def cut_tail(self, pos: int):
        """
        逻辑上截断尾部：
        调整内部指针 _ptr 到 pos，
        下一次 to_dataframe 会只返回前 pos 行。
        """
        if pos < 0 or pos > self._ptr:
            raise IndexError(f"cut_tail position {pos} out of range [0, {self._ptr}]")
        self._ptr = pos

    def concat_small(self, small_df: pd.DataFrame):
        """就地追加 small_df，确保按索引升序，若空间不足则扩容到刚好能装下 new_df 的行数"""
        small_df = small_df.sort_index()
        n = len(small_df)

        # 扩容检查：如果剩余空间不足，则只扩容额外所需行数
        free = self.capacity - self._ptr
        if free < n:
            extra = n - free
            self._resize(extra)

        # 插入数据
        self._data[self._ptr:self._ptr + n, :] = small_df.values
        self._index[self._ptr:self._ptr + n] = small_df.index.values
        self._ptr += n

    def ensure_capacity(self):
        """
        检查剩余预留空间是否不足半数 (capacity/2)，
        若不足，则将容量扩展至原来的 1.5 倍。
        """
        free = self.capacity - self._ptr
        threshold = 10000
        if free < threshold:
            self._resize(10000)

    def _resize(self, extra: int = 10000):
        """扩容：在原有基础上再加 extra 行"""
        new_cap = self.capacity + extra
        new_data = np.empty((new_cap, len(self.columns)), dtype=float)
        new_data[:] = np.nan
        new_data[:self._ptr] = self._data[:self._ptr]

        new_index = np.empty(new_cap, dtype='datetime64[ns]')
        new_index[:] = np.datetime64('NaT')
        new_index[:self._ptr] = self._index[:self._ptr]

        self.capacity = new_cap
        self._data, self._index = new_data, new_index

    def to_dataframe(self) -> pd.DataFrame:
        """返回当前已填充部分的 Pandas DataFrame 视图"""
        df = pd.DataFrame(self._data[:self._ptr, :],
                          index=self._index[:self._ptr],
                          columns=self.columns)
        return df

    def append_row(self, dt: pd.Timestamp, values: Sequence[float]):
        """
        直接在 _ptr 位置写入单行数据，values 顺序要跟 self.columns 对应。
        """
        # 1) 扩容检查
        if self._ptr + 1 > self.capacity:
            self._resize(extra=self.buff_size)

        # 2) 写索引和数据
        self._index[self._ptr] = np.datetime64(dt)
        self._data[self._ptr, :] = values

        # 3) 指针后移
        self._ptr += 1

    def __getattr__(self, name):
        """Delegate other attributes/methods to the current DataFrame"""
        return getattr(self.to_dataframe(), name)

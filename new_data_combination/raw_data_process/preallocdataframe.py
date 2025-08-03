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

    def append_row_keep_first(self, dt: pd.Timestamp, values: Sequence[float]):
        """
        只有当 dt 不等于最后一行的索引时，才 append；
        如果相同，则保留已有的（first）不做任何操作。
        """
        # 如果已有行且最后一行索引和 dt 相同，就直接 return
        if self._ptr > 0 and self._index[self._ptr - 1] >= np.datetime64(dt):
            print('低权限不被覆写')
            return
        # 否则调用原本的 append_row
        print('新来的QQQ')
        self.append_row(dt, values)

    def append_row_keep_last(self, dt: pd.Timestamp, values: Sequence[float]):
        """
        如果 dt 等于最后一行的索引，就用新 values 覆盖最后一行（keep last）；
        否则正常 append。
        """
        if self._ptr > 0 and self._index[self._ptr - 1] == np.datetime64(dt):
            # 直接覆盖最后一行的数据
            self._data[self._ptr - 1, :] = values
            print('高权限被覆写')
        else:
            # 正常追加
            #print('新来的NQ')
            self.append_row(dt, values)


    def insert_row_keep_first(self, dt: pd.Timestamp, values: Sequence[float]):
        """
        在保持升序的前提下插入一行：
        - 如果 dt 已经存在，什么都不做（保留原有第一条）
        - 否则把新行插入到正确的时间顺序位置
        """
        ts = np.datetime64(dt)

        # 1) 用 numpy 在已用索引里定位插入点
        used = self._index[:self._ptr]
        pos = np.searchsorted(used, ts)

        # 2) 如果 pos 指向的就是相同时间戳，则跳过
        if pos < self._ptr and used[pos] == ts:
            return

        # 3) 扩容检查
        if self._ptr + 1 > self.capacity:
            self._resize(extra=self.buff_size)

        # 4) 从 pos 到 ptr-1 的数据往后移一格
        self._data[pos + 1 : self._ptr + 1] = self._data[pos : self._ptr]
        self._index[pos + 1 : self._ptr + 1] = self._index[pos : self._ptr]

        # 5) 在 pos 处写入新行
        self._index[pos] = ts
        self._data[pos, :] = values

        # 6) 指针后移
        self._ptr += 1

    def insert_row_keep_last(self, dt: pd.Timestamp, values: Sequence[float]):
        """
        在保持升序的前提下插入或替换一行：
        - 如果 dt 已经存在，则用新 values 覆盖那一行（保留最新）
        - 否则把新行插入到正确的时间顺序位置
        """
        ts = np.datetime64(dt)

        # 1) 定位潜在插入/替换点
        used = self._index[:self._ptr]
        pos = np.searchsorted(used, ts)

        # 2) 如果正好相同，覆盖那一行
        if pos < self._ptr and used[pos] == ts:
            self._data[pos, :] = values
            return

        # 3) 扩容检查
        if self._ptr + 1 > self.capacity:
            self._resize(extra=self.buff_size)

        # 4) 数据往后移
        self._data[pos + 1 : self._ptr + 1] = self._data[pos : self._ptr]
        self._index[pos + 1 : self._ptr + 1] = self._index[pos : self._ptr]

        # 5) 插入新行
        self._index[pos] = ts
        self._data[pos, :] = values

        # 6) 指针后移
        self._ptr += 1

    def drop_index_duplicates(self, keep='first') -> 'PreallocDataFrame':
        # 先生成一个纯 Pandas DataFrame
        pdf = self.to_dataframe()
        # 按索引去重
        dedup = pdf.loc[~pdf.index.duplicated(keep=keep)]
        # 用同样的 capacity 构造一个新的 PreallocDataFrame
        new = PreallocDataFrame(dedup, capacity=self.capacity)
        return new
    
    def __getattr__(self, name):
        """Delegate other attributes/methods to the current DataFrame"""
        return getattr(self.to_dataframe(), name)
    
    def __getitem__(self, key):
        # 把所有下标访问都转给 pandas.DataFrame
        return self.to_dataframe().__getitem__(key)
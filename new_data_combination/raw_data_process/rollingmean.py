import pandas as pd

class RollingMean:
    def __init__(self, n: int = 60):
        
        
        self.n = n

        # 缓存状态
        self.col = None
        self.df = None
        self.last_t    = None    # 上次的时间戳
        self.start_pos = None    # 窗口起始行号
        self._sum      = None    # 窗口总和

    def mean_before(self, df: pd.DataFrame, t: pd.Timestamp, col: str) -> float:
        

        if col==self.col and id(df) == id(self.df):
            # 定位结束行号（<= t 的最后一条）
            pos_end = self.df.index.searchsorted(t, side='right') - 1
            if pos_end < 0 or pos_end + 1 < self.n:
                raise ValueError(f"在 {t} 之前不足 {self.n} 条记录")
            pos_start = pos_end - self.n + 1

            # 如果和上次时间一样，直接返回缓存
            if t == self.last_t:
                return self._sum / self.n

            one_min = pd.Timedelta(minutes=1)
            # 时间正好后移 1 分钟，则做增量更新
            if self.last_t is not None and (t - self.last_t) == one_min:
                # 掉出最老一条，加入最新一条
                dropped = self.df.iloc[self.start_pos][col]
                added   = self.df.iloc[pos_end][col]
                self._sum = self._sum - dropped + added
                # 起点往后移 1
                self.start_pos += 1
            else:
                # 否则全量重算
                window = self.df.iloc[pos_start:pos_end + 1][col]
                self._sum = window.sum()
                self.start_pos = pos_start

            # 更新缓存
            self.last_t = t
            return self._sum / self.n
        else:
            # 否则全量重算
            self.col=col
            self.df=df

            # 定位结束行号（<= t 的最后一条）
            pos_end = self.df.index.searchsorted(t, side='right') - 1
            if pos_end < 0 or pos_end + 1 < self.n:
                raise ValueError(f"在 {t} 之前不足 {self.n} 条记录")
            pos_start = pos_end - self.n + 1
            
            window = self.df.iloc[pos_start:pos_end + 1][col]
            self._sum = window.sum()
            self.start_pos = pos_start
            self.last_t = t
            return self._sum / self.n
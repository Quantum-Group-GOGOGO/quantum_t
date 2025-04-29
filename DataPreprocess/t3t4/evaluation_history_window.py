import printh as ph
import pandas as pd
import numpy as np
from tqdm import tqdm

class EvaluationWindow:
    def __init__(self, df, evaluation_length):
        """
        初始化 EvaluationWindow 类

        参数:
        - df: 包含数据的 DataFrame
        - evaluation_length: 评价长度。当为正时，基于未来数据；当为负时，基于历史数据
        """
        self.df = df  # 直接使用传入的 DataFrame
        self.df1 = self.df['close']
        self.evaluation_length = int(evaluation_length)
        self.current_index = 0
        # 添加新列 'evaluation' 并填充为 0.0
        self.df['evaluation'] = 0.0
        self.df['length'] = 0.0
        # 计算第一行
        self.calculate_from_start()

    def calculate_from_start(self):
        self.calculate_current_length()
        if self.current_length > 0:
            self.calculate_weight_from_start()
            self.calculate_diff_from_start()
            self.current_evaluation = np.dot(self.diff, self.weights)
            self.df.at[self.current_index, 'evaluation'] = self.current_evaluation

    def calculate_current_length(self):
        # 针对未来数据：evaluation_length > 0
        if self.evaluation_length > 0:
            # 如果 'post_event' 列可用，保证窗口不超过事件后的数据（若没有可忽略此判断）
            if 'post_event' in self.df.columns and self.df['post_event'].iloc[self.current_index] < self.evaluation_length:
                self.current_length = int(self.df['post_event'].iloc[self.current_index])
            else:
                self.current_length = self.evaluation_length

            # 确保窗口不超过剩余数据的长度
            if self.df.shape[0] - self.current_index - 1 < self.current_length:
                self.current_length = self.df.shape[0] - self.current_index - 1

        # 针对历史数据：evaluation_length < 0
        elif self.evaluation_length < 0:
            if 'pre_event' in self.df.columns and self.df['pre_event'].iloc[self.current_index] < -1*self.evaluation_length:
                self.current_length = -1*int(self.df['pre_event'].iloc[self.current_index])
            else:
                self.current_length = self.evaluation_length
            # 确保窗口不超过剩余数据的长度
            if self.current_index < -1*self.current_length:
                self.current_length = -1*self.current_index
        else:
            self.current_length = 0

    def calculate_weight_from_start(self):
        # 生成线性递减权重向量
        if self.evaluation_length > 0:
            self.weights = np.linspace(2 / self.current_length,
                                    2 / (self.current_length * self.current_length),
                                    int(self.current_length))
        else:
            self.weights = np.linspace(2 / (self.current_length * self.current_length),
                                    -2 / self.current_length,
                                    int(-self.current_length))

    def calculate_diff_from_start(self):
        # 对比当前点与未来数据（evaluation_length > 0）或历史数据（evaluation_length < 0）的差异
        base_value = self.df1.iloc[self.current_index]
        if self.evaluation_length > 0:
            window = self.df1.iloc[self.current_index + 1:self.current_index + self.current_length + 1]
            # (未来值 - 当前值) / 当前值 * 1000
            self.diff = (window - base_value) * 1000 / base_value
        elif self.evaluation_length < 0:
            window = self.df1.iloc[self.current_index + self.current_length:self.current_index]
            # (当前值 - 历史值) / 当前值 * 1000
            self.diff = (base_value - window) * 1000 / base_value
        else:
            self.diff = np.array([])

    def move(self):
        self.calculate_current_length()
        if self.current_length > 0:
            self.calculate_weight_from_start()
            self.calculate_diff_from_start()
            self.current_evaluation = np.dot(self.diff, self.weights)
            self.df.at[self.current_index, 'evaluation'] = self.current_evaluation
        if self.current_length < 0:
            self.calculate_weight_from_start()
            self.calculate_diff_from_start()
            self.current_evaluation = np.dot(self.diff, self.weights)
            self.df.at[self.current_index, 'evaluation'] = self.current_evaluation

    def move_to_end(self):
        for self.current_index in tqdm(range(0, len(self.df))):
            self.move()

    def __del__(self):
        """
        析构函数：对象销毁时调用，可以用于清理资源（此例中无需特殊操作）
        """
        pass

# 使用示例
if __name__ == "__main__":
    #data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
    T3_data_path = data_base + '/type3/Nasdaq_qqq_align_labeled_base.pkl'
    df = pd.read_pickle(T3_data_path)
    #df = df.head(int(len(df) * 0.01))
    
    # 初始化 EvaluationWindow 类
    evaluation_window_480 = EvaluationWindow(df, evaluation_length=-480)
    evaluation_window_480.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_480h'}, inplace=True)
    # 初始化 EvaluationWindow 类
    evaluation_window_300 = EvaluationWindow(df, evaluation_length=-300)
    evaluation_window_300.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_300h'}, inplace=True)
    # 初始化 EvaluationWindow 类
    evaluation_window_120 = EvaluationWindow(df, evaluation_length=-120)
    evaluation_window_120.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_120h'}, inplace=True)
    # 初始化 EvaluationWindow 类
    evaluation_window_60 = EvaluationWindow(df, evaluation_length=-60)
    evaluation_window_60.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_60h'}, inplace=True)
    # 初始化 EvaluationWindow 类
    evaluation_window_30 = EvaluationWindow(df, evaluation_length=-30)
    evaluation_window_30.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_30h'}, inplace=True)

    # 初始化 EvaluationWindow 类
    evaluation_window_480 = EvaluationWindow(df, evaluation_length=480)
    evaluation_window_480.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_480'}, inplace=True)
    # 初始化 EvaluationWindow 类
    evaluation_window_300 = EvaluationWindow(df, evaluation_length=300)
    evaluation_window_300.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_300'}, inplace=True)
    # 初始化 EvaluationWindow 类
    evaluation_window_120 = EvaluationWindow(df, evaluation_length=120)
    evaluation_window_120.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_120'}, inplace=True)
    # 初始化 EvaluationWindow 类
    evaluation_window_60 = EvaluationWindow(df, evaluation_length=60)
    evaluation_window_60.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_60'}, inplace=True)
    # 初始化 EvaluationWindow 类
    evaluation_window_30 = EvaluationWindow(df, evaluation_length=30)
    evaluation_window_30.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_30'}, inplace=True)


    # 打印修改后的 DataFrame
    print(df)
    T4_data_path = data_base + '/type4/Nasdaq_qqq_align_labeled_base_evaluated_history.pkl'
    df.to_pickle(T4_data_path)

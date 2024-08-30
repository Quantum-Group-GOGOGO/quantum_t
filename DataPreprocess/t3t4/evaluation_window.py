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
        - evaluation_length: 评价长度，用于滚动窗口的大小
        """
        self.df = df  # 直接使用传入的 DataFrame
        self.df1 = self.df['close']
        self.evaluation_length = int(evaluation_length)
        self.current_length = self.evaluation_length
        self.current_index = int(0)
        # 添加新列 'evaluation' 并填充为 0.0
        self.df['evaluation'] = 0.0
        # 计算第一行
        self.calculate_from_start()


    def calculate_from_start(self):
        self.calculate_current_length()
        if self.current_length>0:
            self.calculate_weight_from_start()
            self.calculate_diff_from_start()
            self.current_evaluation=np.dot(self.diff,self.weights)
            self.df.at[self.current_index, 'evaluation'] = self.current_evaluation
    
    def calculate_current_length(self):
        if df['post_event'].iloc[self.current_index] < self.evaluation_length:
            self.current_length=int(df['post_event'].iloc[self.current_index])
        else:
            self.current_length=self.evaluation_length
        
        if df.shape[0]-self.current_index-1 < self.evaluation_length:
            self.current_length = df.shape[0]-self.current_index-1

    def calculate_weight_from_start(self):
        self.weights = np.linspace(2/self.current_length, 2/(self.current_length*self.current_length), int(self.current_length))  # 生成线性递减权重向量
    def calculate_diff_from_start(self):
        self.diff=(self.df1.iloc[self.current_index+1:self.current_index+self.current_length+1] - self.df1.iloc[self.current_index])*1000/self.df1.iloc[self.current_index]
    def move(self):
        self.calculate_current_length()
        if self.current_length>0:
                self.calculate_weight_from_start()
                self.calculate_diff_from_start()
                self.current_evaluation=np.dot(self.diff,self.weights)
                self.df.at[self.current_index, 'evaluation'] = self.current_evaluation
    def move_to_end(self):
        for self.current_index in tqdm(range(0,len(self.df)-1)):
            self.move()
        #while self.current_index < len(self.df)-1:
            #self.move()

    def __del__(self):
        """
        析构函数：对象销毁时调用，可以用于清理资源（此例中无需特殊操作）
        """

# 使用示例
if __name__ == "__main__":
    data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    T3_data_path=data_base+'/type3/Nasdaq_qqq_align_labeled_base.pkl'
    df = pd.read_pickle(T3_data_path)

    # 初始化 EvaluationWindow 类
    evaluation_window = EvaluationWindow(df, evaluation_length=480)
    evaluation_window.move_to_end()
    df.rename(columns={'evaluation': 'evaluation_300'}, inplace=True)
    # 打印修改后的 DataFrame
    print(df)
    T4_data_path=data_base+'/type4/Nasdaq_qqq_align_labeled_base_evaluated.pkl'
    df.to_pickle(T4_data_path)
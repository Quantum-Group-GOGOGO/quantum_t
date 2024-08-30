import pandas as pd

class PrintH:
    def __init__(self, dataframe):
        self.df = dataframe
        self.hidden_columns = []

    def set_hidden_columns(self, columns):
        """设置需要隐藏的列"""
        self.hidden_columns = columns

    def add_hidden_column(self, column):
        """添加单个隐藏列"""
        if column not in self.hidden_columns:
            self.hidden_columns.append(column)

    def remove_hidden_column(self, column):
        """移除单个隐藏列"""
        if column in self.hidden_columns:
            self.hidden_columns.remove(column)

    def print(self):
        """打印时隐藏指定列"""
        if self.hidden_columns:
            print(self.df.drop(columns=self.hidden_columns).tail())
        else:
            print(self.df.tail())

    def show_column(self, column):
        """显示指定列"""
        if column in self.df.columns:
            print(self.df[column])

    def get_dataframe(self):
        """获取完整的 DataFrame"""
        return self.df
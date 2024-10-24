import pandas as pd
from sklearn.preprocessing import LabelEncoder

class AFIT():
    def __init__(self , text_name):
        self.text_name = text_name
        self.support_id_list = []
        self.antenna_id_list = []
        self.begin_window_list = []
        self.end_window_list = []
        self.visiable_window = []
        self.task_duration = []
        self.antenna_turnaround = []
        self.df = pd.read_csv(f'./{self.text_name}.dat', header=None, sep='\s+', na_values='')

    def read_support_id_list(self):
        first_column = self.df.iloc[ : , 0]
        first_column_list = first_column.tolist()
        self.support_id_list = first_column_list.copy()
        return self.support_id_list

    def read_antenna_id_list(self):
        second_column = self.df.iloc[ : , 1]
        second_column_list = LabelEncoder().fit_transform(second_column)
        self.antenna_id_list = second_column_list.copy()
        return self.antenna_id_list

    def read_visbable_window(self):
        third_column = self.df.iloc[ : , 2]
        third_column_list = third_column.tolist()
        self.begin_window_list = third_column_list.copy()

        forth_column = self.df.iloc[ : , 3]
        forth_column_list = forth_column.tolist()
        self.end_window_list = forth_column_list.copy()

        if len(third_column_list) == len(forth_column_list):
            for i in range(len(third_column_list)):
                self.visiable_window.append([third_column_list[i] , forth_column_list[i]])
            return self.visiable_window
        else:
            print("The dimension of Visibility window not consistent")

    def read_task_duration(self):
        fifth_column  = self.df.iloc[ : ,4]
        fifth_column_list = fifth_column.tolist()
        self.task_duration = fifth_column_list.copy()
        return self.task_duration

    def read_antenna_turnaround(self):
        sixth_column = self.df.iloc[ : ,5]
        sixth_column_list = sixth_column.tolist()
        return sixth_column_list

    def total_length(self):
        length = len(self.df.iloc[:, 0].tolist())
        return length




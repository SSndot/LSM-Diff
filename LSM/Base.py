# -*- coding = utf-8 -*-
# @Time: 2024/3/16 15:28
# @File: Base.py
from brian2 import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from tqdm import tqdm


class Base:
    @staticmethod
    def count_spike_times(spike_counts, t_start, t_end, tau, t_window, datas):
        for t in tqdm(range(t_start, t_end)):
            for neuron_id, spike_times in datas.items():
                spike_times = np.array(spike_times / ms)  # 将 spike_times 转换为 NumPy 数组并转换单位为秒
                spike_mask = (spike_times >= t) & (spike_times < (t + t_window))
                # count = np.sum(np.exp(-(t - spike_times[spike_mask]) / tau))
                count = np.sum(spike_mask)
                spike_counts[t - t_start][neuron_id] = count
        return spike_counts

    @staticmethod
    def calculate_output_weights(train_output, train_target, alpha=0.5):
        model = Ridge(alpha)
        model.fit(train_output, train_target)
        weights = model.coef_
        return weights

    @staticmethod
    def calculate_rmse(predict, target):
        mse = mean_squared_error(predict, target)
        rmse = np.sqrt(mse)
        return rmse

    @staticmethod
    def draw_predict_result(predict, target):
        time_steps = range(len(target))
        plt.plot(time_steps, target, label='target')
        plt.plot(time_steps, predict, label='predicted')
        plt.legend()
        plt.title("Prediction Result")
        plt.show()


base = Base()
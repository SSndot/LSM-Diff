# -*- coding = utf-8 -*-
# @Time: 2024/3/16 15:28
# @File: base.py
from brian2 import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from tqdm import tqdm
from params import args


class Base:
    @staticmethod
    def count_spike_times(datas, t_start_frame, t_end_frame, tau=args.tau, t_window=args.window_size):
        sample_total = t_end_frame - t_start_frame
        spike_counts = np.zeros((sample_total, args.output_size))
        i = 0
        for t in tqdm(range(t_start_frame, t_end_frame)):
            for neuron_id, spike_times in datas.items():
                spike_times = np.array(spike_times / ms)
                spike_mask = (spike_times >= t * args.frame_duration) & (spike_times < (t+1) * args.frame_duration)
                count = np.sum(np.exp((t * args.frame_duration - spike_times[spike_mask]) / tau))
                spike_counts[i][neuron_id] = count
            i += 1
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

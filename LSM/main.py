# -*- coding = utf-8 -*-
# @Time: 2024/3/17 15:45
# @File: main.py
# -*- coding = utf-8 -*-
# @Time: 2024/3/16 15:27
# @File: Process.py
from brian2 import *
from LSM import nbp, net, monitor
from Input import spikes, data
from Base import base


# 定义仿真参数 ----------------------------------------------
class SimPara:
    def __init__(self):
        self.total_duration = 5000
        self.train_duration = 1000
        self.test_duration = 500
        self.predict_duration = 100

        self.train_t_start = 200
        self.train_t_end = self.train_t_start + self.train_duration

        self.test_t_start = 1400
        self.test_t_end = self.test_t_start + self.test_duration


sim = SimPara()
# ----------------------------------------------------------

# 运行仿真 --------------------------------------------------
start_scope()
spike_times = spikes
net["stimulus"].set_spikes([0] * len(spike_times), spike_times * ms)
net.run(sim.total_duration*ms)
# ----------------------------------------------------------

# 获得数据 --------------------------------------------------
spike_exc_datas = monitor.spike_g_ex.spike_trains()
spike_inh_datas = monitor.spike_g_inh.spike_trains()
# ----------------------------------------------------------

# 超参数 ----------------------------------------------------
tau = 2
t_window = 50.0
# ----------------------------------------------------------

# 获得训练/测试结果 ------------------------------------------
spike_counts_exc = np.zeros((sim.train_duration, nbp.n_ex))
spike_counts_inh = np.zeros((sim.train_duration, nbp.n_inh))

spike_counts_exc = base.count_spike_times(spike_counts_exc, sim.train_t_start, sim.train_t_end, tau, t_window, spike_exc_datas)
spike_counts_inh = base.count_spike_times(spike_counts_inh, sim.train_t_start, sim.train_t_end, tau, t_window, spike_inh_datas)
train_spike_data = np.concatenate((spike_counts_exc, spike_counts_inh), axis=1)

spike_counts_exc = np.zeros((sim.test_duration, nbp.n_ex))
spike_counts_inh = np.zeros((sim.test_duration, nbp.n_inh))

spike_counts_exc = base.count_spike_times(spike_counts_exc, sim.test_t_start, sim.test_t_end, tau, t_window, spike_exc_datas)
spike_counts_inh = base.count_spike_times(spike_counts_inh, sim.test_t_start, sim.test_t_end, tau, t_window, spike_inh_datas)
test_spike_data = np.concatenate((spike_counts_exc, spike_counts_inh), axis=1)
# ----------------------------------------------------------

# 训练权重 --------------------------------------------------
train_target = data[sim.train_t_start + sim.predict_duration:sim.train_t_end + sim.predict_duration]
weights = base.calculate_output_weights(train_spike_data, train_target, 5)
train_predict = np.dot(train_spike_data, weights)
# ----------------------------------------------------------

# 查看脉冲次数与输入之间的关系 --------------------------------
spike_counts_exc_total = np.zeros((2000, nbp.n_ex))
spike_counts_inh_total = np.zeros((2000, nbp.n_inh))
spike_counts_exc_total = base.count_spike_times(spike_counts_exc_total, 0, 2000, tau, t_window, spike_exc_datas)
spike_counts_inh_total = base.count_spike_times(spike_counts_inh_total, 0, 2000, tau, t_window, spike_inh_datas)
total_spike_data = np.concatenate((spike_counts_exc_total, spike_counts_inh_total), axis=1)
count_sum = []
for t_spike in total_spike_data:
    sum = 0
    for spike in t_spike:
        sum += spike
    count_sum.append(sum)
print(count_sum)
base.draw_predict_result(count_sum, data[0:2000])
# ----------------------------------------------------------

# 进行测试 --------------------------------------------------
test_target = data[sim.test_t_start + sim.predict_duration:sim.test_t_end + sim.predict_duration]
test_predict = np.dot(test_spike_data, weights)
rmse = base.calculate_rmse(test_predict, test_target)
print("RMSE = %f" % rmse)
base.draw_predict_result(train_predict, train_target)
base.draw_predict_result(test_predict, test_target)
# ----------------------------------------------------------



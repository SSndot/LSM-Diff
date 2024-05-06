# -*- coding = utf-8 -*-
# @Time: 2024/3/16 17:25
# @File: Train.py
from brian2 import *
from hyperopt import fmin, tpe, hp, Trials
from LSM import nbp, net, monitor
from Base import base
from Input import data, spikes

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


# 定义优化目标函数
def objective(params):
    # 获得训练/测试结果 ------------------------------------------
    train_spike_counts_exc = np.zeros((sim.train_duration, nbp.n_ex))
    train_spike_counts_inh = np.zeros((sim.train_duration, nbp.n_inh))

    train_spike_counts_exc = base.count_spike_times(train_spike_counts_exc, sim.train_t_start, sim.train_t_end,
                                              5, 100, spike_exc_datas)
    train_spike_counts_inh = base.count_spike_times(train_spike_counts_inh, sim.train_t_start, sim.train_t_end,
                                              5, 100, spike_inh_datas)
    train_spike_data = np.concatenate((train_spike_counts_exc, train_spike_counts_inh), axis=1)

    test_spike_counts_exc = np.zeros((sim.test_duration, nbp.n_ex))
    test_spike_counts_inh = np.zeros((sim.test_duration, nbp.n_inh))

    test_spike_counts_exc = base.count_spike_times(test_spike_counts_exc, sim.test_t_start, sim.test_t_end,
                                              5, 100, spike_exc_datas)
    test_spike_counts_inh = base.count_spike_times(test_spike_counts_inh, sim.test_t_start, sim.test_t_end,
                                              5, 100, spike_inh_datas)
    test_spike_data = np.concatenate((test_spike_counts_exc, test_spike_counts_inh), axis=1)
    # ----------------------------------------------------------

    # 训练权重 --------------------------------------------------
    train_target = data[sim.train_t_start + sim.predict_duration:sim.train_t_end + sim.predict_duration]
    weights = base.calculate_output_weights(train_spike_data, train_target, params['alpha'])
    train_predict = np.dot(train_spike_data, weights)
    # ----------------------------------------------------------

    # 进行测试 --------------------------------------------------
    test_target = data[sim.test_t_start + sim.predict_duration:sim.test_t_end + sim.predict_duration]
    test_predict = np.dot(test_spike_data, weights)
    rmse = base.calculate_rmse(test_predict, test_target)

    return {'loss': rmse, 'status': 'ok'}


# 定义参数空间
params = {
    # 'tau': hp.uniform('tau', 0.5, 10),
    # 't_window': hp.quniform('t_window', 10, 200, 10),
    'alpha': hp.uniform('alpha', 0.5, 5)
}

# 创建Trials对象用于记录优化过程
trials = Trials()

# 执行超参数优化
best = fmin(fn=objective,
            space=params,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials)

# 输出最佳参数
print("Best parameters: ", best)
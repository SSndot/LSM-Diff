import numpy as np
from brian2 import *
from D_LSM import generate_network
from PicInput import data_loader
from base import base
from params import args
from tqdm import tqdm


def run_simulation(_spike_times):
    net, monitor = generate_network()
    indices_list = []
    time_list = []
    for i in range(len(_spike_times)):
        for idx, spike_time in enumerate(_spike_times[i]):
            if len(spike_time) == 0:
                continue
            spike_time = list(spike_time / ms)
            indices_list += [idx] * len(spike_time)
            time_list += spike_time
    net["stimulus"].set_spikes(indices_list, time_list * ms)
    net.run(args.sim_duration*ms)

    spike_output_data = monitor.spike_g_output.spike_trains()
    _train_spike_data = base.count_spike_times(spike_output_data, 4, args.train_num)
    _test_spike_data = base.count_spike_times(spike_output_data, args.train_num, args.data_num - 1)

    return _train_spike_data, _test_spike_data


# 运行仿真 --------------------------------------------------
if __name__ == '__main__':
    start_scope()

    train_spike_datas = np.empty((0, args.output_size))
    test_spike_datas = np.empty((0, args.output_size))

    train_targets = []
    test_targets = []

    for i in tqdm(range(args.epochs)):
        image_set = data_loader.get_set(i)
        spike_times = data_loader.get_spike_times(i)

        train_spike_data, test_spike_data = run_simulation(spike_times)

        train_target = image_set[5:args.train_num + 1]
        test_target = image_set[args.train_num + 1:]

        train_spike_datas = np.concatenate((train_spike_datas, train_spike_data), axis=0)
        test_spike_datas = np.concatenate((test_spike_datas, test_spike_data), axis=0)

        train_targets += train_target
        test_targets += test_target

    weights = base.calculate_output_weights(train_spike_datas, train_targets)
    train_predict = np.dot(train_spike_datas, weights)

    test_predict = np.dot(test_spike_datas, weights)
    rmse = base.calculate_rmse(test_predict, test_targets)
    print("RMSE = %f" % rmse)




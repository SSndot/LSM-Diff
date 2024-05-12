import numpy as np
from PIL import Image
from brian2 import *
from params import args
import pickle
from brian2 import prefs
prefs.codegen.target = "numpy"
duration = args.frame_duration * ms


class Data:
    def __init__(self):
        self.target_set = "video_"
        self.encoder = 'poisson'

    def get_set(self, target_batch):
        image_set = []
        target_path = self.target_set + str(target_batch)
        for idx in range(args.data_num):
            pic_name = '/frame_' + str(idx) + '.jpg'
            img_path = args.file_path + target_path + pic_name
            image = Image.open(img_path).convert("L")  # 转换为灰度图像
            resized_image = image.resize((args.compress_resize, args.compress_resize))
            image_array = np.array(resized_image)
            image_set.append(image_array.flatten())
        return image_set

    def get_spike_times(self, target_batch):
        spike_times = []
        target_path = self.target_set + str(target_batch)
        for idx in range(args.data_num):
            pic_name = '/frame_' + str(idx) + '.jpg'
            img_path = args.file_path + target_path + pic_name
            image = Image.open(img_path).convert("L")  # 转换为灰度图像
            resized_image = image.resize((args.compress_resize, args.compress_resize))
            image_array = np.array(resized_image)
            firing_rate = image_array.flatten() * args.firing_amplify * Hz
            poisson_group = PoissonGroup(args.input_size, rates=firing_rate)
            spike_monitor = SpikeMonitor(poisson_group)
            run(duration)
            spike_time = []
            for i in range(args.input_size):
                spikes = spike_monitor.i == i
                spike_neuron_time = spike_monitor.t[spikes] + idx * args.frame_duration * ms
                spike_time.append(spike_neuron_time)
            spike_times.append(spike_time)
        return spike_times


data_loader = Data()

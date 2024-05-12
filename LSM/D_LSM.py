# -*- coding = utf-8 -*-
# @Time: 2024/3/16 13:37
# @File: LSM.py
import numpy as np
from brian2 import *
from dataclasses import dataclass
from params import args
warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"


# 神经元数量参数类 ------------------------------------------
@dataclass
class ExplicitLayer:
    n_x: int = 16
    n_y: int = 16
    n_total: int = n_x * n_y

    def allocate(self, group):
        v = np.zeros((self.n_x, self.n_y), [('x', float), ('y', float)])
        v['x'], v['y'] = np.meshgrid(np.linspace(0, self.n_x - 1, self.n_x),
                                     np.linspace(0, self.n_y - 1, self.n_y),
                                    )
        v = v.reshape(self.n_total)
        np.random.shuffle(v)
        n = 0
        for g in group:
            for i in range(g.N):
                g.x[i], g.y[i] = v[n][0], v[n][1]
                n += 1
        return group


@dataclass
class ReserveLayer:
    n_x: int = 12
    n_y: int = 12
    n_z: int = 60
    ex_inh_ratio: int = 4

    @property
    def n_total(self):
        return self.n_x * self.n_y * self.n_z

    @property
    def n_ex(self):
        return int(self.n_total / (self.ex_inh_ratio + 1) * self.ex_inh_ratio)

    @property
    def n_inh(self):
        return int(self.n_total / (self.ex_inh_ratio + 1))

    @property
    def n_input(self):
        return int(0.1 * self.n_ex)

    @property
    def n_read(self):
        return self.n_ex + self.n_inh

    def allocate(self, group):
        v = np.zeros((self.n_x, self.n_y, self.n_z), [('x', float), ('y', float), ('z', float)])
        v['x'], v['y'], v['z'] = np.meshgrid(np.linspace(0, self.n_x - 1, self.n_x),
                                             np.linspace(0, self.n_y - 1, self.n_y),
                                             np.linspace(0, self.n_z - 1, self.n_z))
        v = v.reshape(self.n_total)
        np.random.shuffle(v)
        n = 0
        for g in group:
            for i in range(g.N):
                g.x[i], g.y[i], g.z[i] = v[n][0], v[n][1], v[n][2]
                n += 1
        return group
# ----------------------------------------------------------


# 监视器类 --------------------------------------------------
class Monitor:
    def __init__(self, st_input, st_ex, st_inh, st_output, sp_stimulus, sp_input, sp_ex, sp_inh, sp_output):
        self.state_g_input = st_input
        self.state_g_ex = st_ex
        self.state_g_inh = st_inh
        self.state_g_output = st_output

        self.spike_stimulus = sp_stimulus
        self.spike_g_input = sp_input
        self.spike_g_ex = sp_ex
        self.spike_g_inh = sp_inh
        self.spike_g_output = sp_output

    @staticmethod
    def draw_spike(spike):
        spike_datas = spike.spike_trains()
        neuron_ids = list(spike_datas.keys())
        neuron_times = list(spike_datas.values())

        plt.figure()
        for idx, times in zip(neuron_ids, neuron_times):
            plt.scatter(times / ms, [idx] * len(times), marker='o')

        plt.xlabel('Time')
        plt.ylabel('Neuron ID')
        plt.title('Spike Scatter Plot')
        plt.show()

    @staticmethod
    def draw_state(state):
        # 绘制电压变化曲线
        plt.plot(state.t / ms, state.v[0] / volt)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (V)')
        plt.title('Neuron Voltage')
        plt.show()
# ----------------------------------------------------------

def generate_network():
    # 定义神经元 ------------------------------------------------
    neuron_in = '''
    I = stimulus(t,i) : 1
    '''

    neuron = '''
    dv/dt = (I-v) / (30*ms) : 1 (unless refractory)
    dg/dt = (-g)/(3*ms) : 1
    dh/dt = (-h)/(6*ms) : 1
    I = (g+h)+13.5: 1
    x : 1
    y : 1
    z : 1
    '''

    neuron_read = '''
    dv/dt = (I-v) / (30*ms) : 1
    dg/dt = (-g)/(3*ms) : 1 
    dh/dt = (-h)/(6*ms) : 1
    I = (g+h): 1
    '''
    # ----------------------------------------------------------

    # 定义突触 --------------------------------------------------
    synapse = '''
    w : 1
    '''

    synapse_STDP = '''
    w : 1
    wmax : 1
    wmin : 1
    Apre : 1
    Apost = -Apre * taupre / taupost * 1.2 : 1
    taupre : second
    taupost : second
    dapre/dt = -apre/taupre : 1 (clock-driven)
    dapost/dt = -apost/taupost : 1 (clock-driven)
    '''

    on_pre_ex = '''
    g+=w
    '''

    on_pre_inh = '''
    h+=w
    '''

    on_pre_ex_STDP = '''
    g+=w
    apre += Apre
    w += apost
    '''

    on_post_ex_STDP = '''
    apost += Apost
    w += apre
    '''

    on_pre_inh_STDP = '''
    h+=w
    apre += Apre
    w -= apost
    '''

    on_post_inh_STDP = '''
    apost += Apost
    w -= apre
    '''

    on_pre_read = '''
    g+=w
    '''
    # ----------------------------------------------------------

    # 定义输入层神经元群 ----------------------------------------------
    il = ExplicitLayer(16, 16)  # 定义神经元模型基本信息

    G_input = NeuronGroup(il.n_total, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                       name='G_input')
    # ----------------------------------------------------------------

    # 定义输出层神经元群 ----------------------------------------------
    ol_1 = ExplicitLayer(16, 16)  # 定义神经元模型基本信息

    G_output_1 = NeuronGroup(ol_1.n_total, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                       name='G_output_1')

    ol_2 = ExplicitLayer(16, 16)  # 定义神经元模型基本信息

    G_output_2 = NeuronGroup(ol_2.n_total, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                       name='G_output_2')

    ol_3 = ExplicitLayer(16, 16)  # 定义神经元模型基本信息

    G_output_3 = NeuronGroup(ol_3.n_total, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                       name='G_output_3')
    # ----------------------------------------------------------------

    # 定义储备池层神经元群 ----------------------------------------------
    rl_1 = ReserveLayer(12, 12, 60)  # 定义神经元模型基本信息

    G_ex_1 = NeuronGroup(rl_1.n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                       name='G_ex_1')

    G_inh_1 = NeuronGroup(rl_1.n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                        name='G_inh_1')

    rl_2 = ReserveLayer(12, 12, 60)  # 定义神经元模型基本信息

    G_ex_2 = NeuronGroup(rl_2.n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                       name='G_ex_2')

    G_inh_2 = NeuronGroup(rl_2.n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                        name='G_inh_2')

    rl_3 = ReserveLayer(12, 12, 60)  # 定义神经元模型基本信息

    G_ex_3 = NeuronGroup(rl_3.n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                       name='G_ex_3')

    G_inh_3 = NeuronGroup(rl_3.n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                        name='G_inh_3')
    # ----------------------------------------------------------

    # 定义神经元群连接 ------------------------------------------
    stimulus = SpikeGeneratorGroup(args.input_size, [], [] * ms, name='stimulus')

    # Layer1
    S_in = Synapses(stimulus, G_input, synapse, on_pre=on_pre_ex, method='euler', name='S_in')
    S_in_E1 = Synapses(G_input, G_ex_1, synapse, on_pre=on_pre_ex, method='euler', name='S_in_E1')
    S_in_I1 = Synapses(G_input, G_inh_1, synapse, on_pre=on_pre_ex, method='euler', name='S_in_I1')
    S_E1_E1 = Synapses(G_ex_1, G_ex_1, synapse_STDP, on_pre=on_pre_ex_STDP, on_post=on_post_ex_STDP, method='euler', name='S_E1_E1')
    S_E1_I1 = Synapses(G_ex_1, G_inh_1, synapse_STDP, on_pre=on_pre_ex_STDP, on_post=on_post_ex_STDP, method='euler', name='S_E1_I1')
    S_I1_E1 = Synapses(G_inh_1, G_ex_1, synapse_STDP, on_pre=on_pre_inh_STDP, on_post=on_post_inh_STDP, method='euler', name='S_I1_E1')
    S_I1_I1 = Synapses(G_inh_1, G_inh_1, synapse_STDP, on_pre=on_pre_inh_STDP, on_post=on_post_inh_STDP, method='euler', name='S_I1_I1')
    S_E1_out1 = Synapses(G_ex_1, G_output_1, 'w = 1 : 1', on_pre=on_pre_ex, method='euler', name='S_E1_out1')
    S_I1_out1 = Synapses(G_inh_1, G_output_1, 'w = 1 : 1', on_pre=on_pre_inh, method='euler', name='S_I1_out1')

    # Layer2
    S_in_E2 = Synapses(G_output_1, G_ex_2, synapse, on_pre=on_pre_ex, method='euler', name='S_in_E2')
    S_in_I2 = Synapses(G_output_1, G_inh_2, synapse, on_pre=on_pre_ex, method='euler', name='S_in_I2')
    S_E2_E2 = Synapses(G_ex_2, G_ex_2, synapse_STDP, on_pre=on_pre_ex_STDP, on_post=on_post_ex_STDP, method='euler', name='S_E2_E2')
    S_E2_I2 = Synapses(G_ex_2, G_inh_2, synapse_STDP, on_pre=on_pre_ex_STDP, on_post=on_post_ex_STDP, method='euler', name='S_E2_I2')
    S_I2_E2 = Synapses(G_inh_2, G_ex_2, synapse_STDP, on_pre=on_pre_inh_STDP, on_post=on_post_inh_STDP, method='euler', name='S_I2_E2')
    S_I2_I2 = Synapses(G_inh_2, G_inh_2, synapse_STDP, on_pre=on_pre_inh_STDP, on_post=on_post_inh_STDP, method='euler', name='S_I2_I2')
    S_E2_out2 = Synapses(G_ex_2, G_output_2, 'w = 1 : 1', on_pre=on_pre_ex, method='euler', name='S_E2_out2')
    S_I2_out2 = Synapses(G_inh_2, G_output_2, 'w = 1 : 1', on_pre=on_pre_inh, method='euler', name='S_I2_out2')

    # Layer3
    S_in_E3 = Synapses(G_output_2, G_ex_3, synapse, on_pre=on_pre_ex, method='euler', name='S_in_E3')
    S_in_I3 = Synapses(G_output_2, G_inh_3, synapse, on_pre=on_pre_ex, method='euler', name='S_in_I3')
    S_E3_E3 = Synapses(G_ex_3, G_ex_3, synapse_STDP, on_pre=on_pre_ex_STDP, on_post=on_post_ex_STDP, method='euler', name='S_E3_E3')
    S_E3_I3 = Synapses(G_ex_3, G_inh_3, synapse_STDP, on_pre=on_pre_ex_STDP, on_post=on_post_ex_STDP, method='euler', name='S_E3_I3')
    S_I3_E3 = Synapses(G_inh_3, G_ex_3, synapse_STDP, on_pre=on_pre_inh_STDP, on_post=on_post_inh_STDP, method='euler', name='S_I3_E3')
    S_I3_I3 = Synapses(G_inh_3, G_inh_3, synapse_STDP, on_pre=on_pre_inh_STDP, on_post=on_post_inh_STDP, method='euler', name='S_I3_I3')
    S_E3_out3 = Synapses(G_ex_3, G_output_3, 'w = 1 : 1', on_pre=on_pre_ex, method='euler', name='S_E3_out3')
    S_I3_out3 = Synapses(G_inh_3, G_output_3, 'w = 1 : 1', on_pre=on_pre_inh, method='euler', name='S_I3_out3')
    # ----------------------------------------------------------

    # 定义神经元群参数 -------------------------------------------
    duration = args.sim_duration
    Dt = args.dt * ms

    G_input.v = '13.5+1.5*rand()'
    G_ex_1.v = '13.5+1.5*rand()'
    G_inh_1.v = '13.5+1.5*rand()'
    G_output_1.v = '13.5+1.5*rand()'

    G_ex_2.v = '13.5+1.5*rand()'
    G_inh_2.v = '13.5+1.5*rand()'
    G_output_2.v = '13.5+1.5*rand()'

    G_ex_3.v = '13.5+1.5*rand()'
    G_inh_3.v = '13.5+1.5*rand()'
    G_output_3.v = '13.5+1.5*rand()'

    G_input.g = '0'
    G_ex_1.g = '0'
    G_inh_1.g = '0'
    G_output_1.g = '0'

    G_ex_2.g = '0'
    G_inh_2.g = '0'
    G_output_2.g = '0'

    G_ex_3.g = '0'
    G_inh_3.g = '0'
    G_output_3.g = '0'

    G_input.h = '0'
    G_ex_1.h = '0'
    G_inh_1.h = '0'
    G_output_1.h = '0'

    G_ex_2.h = '0'
    G_inh_2.h = '0'
    G_output_2.h = '0'

    G_ex_3.h = '0'
    G_inh_3.h = '0'
    G_output_3.h = '0'

    [G_ex_1, G_inh_1] = rl_1.allocate([G_ex_1, G_inh_1])
    [G_ex_2, G_inh_2] = rl_1.allocate([G_ex_2, G_inh_2])
    [G_ex_3, G_inh_3] = rl_1.allocate([G_ex_3, G_inh_3])

    G_input.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_ex_1.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_inh_1.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_output_1.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_ex_2.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_inh_2.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_output_2.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_ex_3.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_inh_3.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    G_output_3.run_regularly('''v = 13.5+1.5*rand()
                        g = 0
                        h = 0
                        ''', dt=duration*Dt)
    # ----------------------------------------------------------

    # 定义神经元群连接参数 ---------------------------------------
    S_in.connect(j='i')
    n_amplify = args.input_amplify
    S_in_E1.connect(j='k for k in range(i*n_amplify, (i+1)*n_amplify)')
    S_in_I1.connect(p=0)
    S_E1_E1.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_E1_I1.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_I1_E1.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_I1_I1.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')

    S_in_E2.connect(j='k for k in range(i*n_amplify*2, (i+1)*n_amplify*2)')
    S_in_I2.connect(p=0)
    S_E2_E2.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_E2_I2.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_I2_E2.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_I2_I2.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')

    S_in_E3.connect(j='k for k in range(i*n_amplify*3, (i+1)*n_amplify*3)')
    S_in_I3.connect(p=0)
    S_E3_E3.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_E3_I3.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_I3_E3.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
    S_I3_I3.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')

    num_ex = int(ol_1.n_total * 0.75)
    num_inh = int(ol_1.n_total * 0.25)
    array_ex_i = np.arange(rl_1.n_ex - num_ex, rl_1.n_ex)
    array_ex_j = np.arange(0, num_ex)
    array_inh_i = np.arange(0, num_inh)
    array_inh_j = np.arange(num_ex, num_ex + num_inh)
    S_E1_out1.connect(i=array_ex_i, j=array_ex_j)
    S_I1_out1.connect(i=array_inh_i, j=array_inh_j)
    S_E2_out2.connect(i=array_ex_i, j=array_ex_j)
    S_I2_out2.connect(i=array_inh_i, j=array_inh_j)
    S_E3_out3.connect(i=array_ex_i, j=array_ex_j)
    S_I3_out3.connect(i=array_inh_i, j=array_inh_j)

    A_EE = 30
    A_EI = 60
    A_IE = -19
    A_II = -19
    A_in = 18
    A_inE = 18
    A_inI = 9

    S_in.w = 'A_in*randn()+A_in'
    S_in_E1.w = 'A_inE*randn()+A_inE'
    S_in_I1.w = 'A_inI*randn()+A_inI'
    S_E1_E1.w = 'A_EE*randn()+A_EE'
    S_I1_E1.w = 'A_IE*randn()+A_IE'
    S_E1_I1.w = 'A_EI*randn()+A_EI'
    S_I1_I1.w = 'A_II*randn()+A_II'

    S_E1_E1.delay = '1.5*ms'
    S_E1_I1.delay = '0.8*ms'
    S_I1_E1.delay = '0.8*ms'
    S_I1_I1.delay = '0.8*ms'

    S_E1_E1.taupre = S_E1_E1.taupost = 20*ms
    S_E1_E1.Apre = 0.1

    S_E1_I1.taupre = S_E1_I1.taupost = 20*ms
    S_E1_I1.Apre = 0.1

    S_I1_E1.taupre = S_I1_E1.taupost = 20*ms
    S_I1_E1.Apre = 0.01

    S_I1_I1.taupre = S_I1_I1.taupost = 20*ms
    S_I1_I1.Apre = 0.01

    S_in_E2.w = 'A_inE*randn()+A_inE'
    S_in_I2.w = 'A_inI*randn()+A_inI'
    S_E2_E2.w = 'A_EE*randn()+A_EE'
    S_I2_E2.w = 'A_IE*randn()+A_IE'
    S_E2_I2.w = 'A_EI*randn()+A_EI'
    S_I2_I2.w = 'A_II*randn()+A_II'

    S_E2_E2.delay = '1.5*ms'
    S_E2_I2.delay = '0.8*ms'
    S_I2_E2.delay = '0.8*ms'
    S_I2_I2.delay = '0.8*ms'

    S_E2_E2.taupre = S_E2_E2.taupost = 20*ms
    S_E2_E2.Apre = 0.1

    S_E2_I2.taupre = S_E2_I2.taupost = 20*ms
    S_E2_I2.Apre = 0.1

    S_I2_E2.taupre = S_I2_E2.taupost = 20*ms
    S_I2_E2.Apre = 0.01

    S_I2_I2.taupre = S_I2_I2.taupost = 20*ms
    S_I2_I2.Apre = 0.01

    S_in_E3.w = 'A_inE*randn()+A_inE'
    S_in_I3.w = 'A_inI*randn()+A_inI'
    S_E3_E3.w = 'A_EE*randn()+A_EE'
    S_I3_E3.w = 'A_IE*randn()+A_IE'
    S_E3_I3.w = 'A_EI*randn()+A_EI'
    S_I3_I3.w = 'A_II*randn()+A_II'

    S_E3_E3.delay = '1.5*ms'
    S_E3_I3.delay = '0.8*ms'
    S_I3_E3.delay = '0.8*ms'
    S_I3_I3.delay = '0.8*ms'

    S_E3_E3.taupre = S_E3_E3.taupost = 20*ms
    S_E3_E3.Apre = 0.1

    S_E3_I3.taupre = S_E3_I3.taupost = 20*ms
    S_E3_I3.Apre = 0.1

    S_I3_E3.taupre = S_I3_E3.taupost = 20*ms
    S_I3_E3.Apre = 0.01

    S_I3_I3.taupre = S_I3_I3.taupost = 20*ms
    S_I3_I3.Apre = 0.01
    # ----------------------------------------------------------

    # 设置监视器 ------------------------------------------------
    state_g_input = StateMonitor(G_input, (['I', 'v']), record=True)
    state_g_ex = StateMonitor(G_ex_1, (['I', 'v']), record=True)
    state_g_inh = StateMonitor(G_inh_1, (['I', 'v']), record=True)
    state_g_output1 = StateMonitor(G_output_1, (['I', 'v']), record=True)
    state_g_output2 = StateMonitor(G_output_2, (['I', 'v']), record=True)
    state_g_output3 = StateMonitor(G_output_3, (['I', 'v']), record=True)


    spike_stimulus = SpikeMonitor(stimulus, name='spike_monitor_stimulus')
    spike_g_input = SpikeMonitor(G_input, name='spike_monitor_input')
    spike_g_ex = SpikeMonitor(G_ex_1, name='spike_monitor_exc')
    spike_g_inh = SpikeMonitor(G_inh_1, name='spike_monitor_inh')
    spike_g_output1 = SpikeMonitor(G_output_1, name='spike_monitor_output1')
    spike_g_output2 = SpikeMonitor(G_output_2, name='spike_monitor_output2')
    spike_g_output3 = SpikeMonitor(G_output_3, name='spike_monitor_output3')
    # ----------------------------------------------------------

    # ------create network-------------
    net = Network(collect())
    monitor = Monitor(state_g_input, state_g_ex, state_g_inh, state_g_output3,
                  spike_stimulus, spike_g_input, spike_g_ex, spike_g_inh, spike_g_output3)

    return net, monitor



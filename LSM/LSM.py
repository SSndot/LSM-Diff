# -*- coding = utf-8 -*-
# @Time: 2024/3/16 13:37
# @File: LSM.py
from brian2 import *
from dataclasses import dataclass
warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"


# 神经元数量参数类 ------------------------------------------
@dataclass
class NeuralBasePara:
    n_x: int = 3
    n_y: int = 3
    n_z: int = 15
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
    def __init__(self, st_ex, st_inh, st_read, sp_ex, sp_inh):
        self.state_g_ex = st_ex
        self.state_g_inh = st_inh
        self.state_g_read = st_read

        self.spike_g_ex = sp_ex
        self.spike_g_inh = sp_inh

    @staticmethod
    def draw_spike(spike):
        spike_datas = spike.spike_trains()
        neuron_ids = list(spike_datas.keys())
        neuron_times = list(spike_datas.values())

        plt.figure()
        for id, times in zip(neuron_ids, neuron_times):
            plt.scatter(times / ms, [id] * len(times), marker='o')

        plt.xlabel('Time')
        plt.ylabel('Neuron ID')
        plt.title('Spike Scatter Plot')
        plt.show()
# ----------------------------------------------------------


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

# 定义神经元群 ----------------------------------------------
nbp = NeuralBasePara(3, 3, 15)  # 定义神经元模型基本信息

G_ex = NeuronGroup(nbp.n_ex, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=3 * ms,
                   name='neurongroup_ex')

G_inh = NeuronGroup(nbp.n_inh, neuron, threshold='v > 15', reset='v = 13.5', method='euler', refractory=2 * ms,
                    name='neurongroup_in')

G_readout = NeuronGroup(nbp.n_read, neuron_read, method='euler', name='neurongroup_read')
# ----------------------------------------------------------

# 定义神经元群连接 ------------------------------------------
stimulus = SpikeGeneratorGroup(1, [], [] * ms, name="stimulus")

S_inE = Synapses(stimulus, G_ex, synapse, on_pre=on_pre_ex, method='euler', name='synapses_inE')

S_inI = Synapses(stimulus, G_inh, synapse, on_pre=on_pre_ex, method='euler', name='synapses_inI')

S_EE = Synapses(G_ex, G_ex, synapse_STDP, on_pre=on_pre_ex_STDP, on_post=on_post_ex_STDP, method='euler', name='synapses_EE')

S_EI = Synapses(G_ex, G_inh, synapse_STDP, on_pre=on_pre_ex_STDP, on_post=on_post_ex_STDP, method='euler', name='synapses_EI')

S_IE = Synapses(G_inh, G_ex, synapse_STDP, on_pre=on_pre_inh_STDP, on_post=on_post_inh_STDP, method='euler', name='synapses_IE')

S_II = Synapses(G_inh, G_inh, synapse_STDP, on_pre=on_pre_inh_STDP, on_post=on_post_inh_STDP, method='euler', name='synapses_II')

S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_ex, method='euler')

S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_inh, method='euler')
# ----------------------------------------------------------

# 定义神经元群参数 -------------------------------------------
duration = 1000
Dt = 1*ms

G_ex.v = '13.5+1.5*rand()'
G_inh.v = '13.5+1.5*rand()'
G_readout.v = '0'
G_ex.g = '0'
G_inh.g = '0'
G_readout.g = '0'
G_ex.h = '0'
G_inh.h = '0'
G_readout.h = '0'

[G_ex, G_inh] = nbp.allocate([G_ex, G_inh])

G_ex.run_regularly('''v = 13.5+1.5*rand()
                    g = 0
                    h = 0
                    ''',dt=duration*Dt)
G_inh.run_regularly('''v = 13.5+1.5*rand()
                    g = 0
                    h = 0
                    ''',dt=duration*Dt)
G_readout.run_regularly('''v = 0
                    g = 0
                    h = 0
                    ''',dt=duration*Dt)
# ----------------------------------------------------------

# 定义神经元群连接参数 ---------------------------------------
S_inE.connect(condition='j<0.1*N_post')
S_inI.connect(p=0)
S_EE.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
S_EI.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
S_IE.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
S_II.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/2**2)')
S_E_readout.connect(j='i')
n_ex = nbp.n_ex
S_I_readout.connect(j='i+n_ex')

A_EE = 30
A_EI = 60
A_IE = -19
A_II = -19
A_inE = 18
A_inI = 9

S_inE.w = 'A_inE*randn()+A_inE'
S_inI.w = 'A_inI*randn()+A_inI'
S_EE.w = 'A_EE*randn()+A_EE'
S_IE.w = 'A_IE*randn()+A_IE'
S_EI.w = 'A_EI*randn()+A_EI'
S_II.w = 'A_II*randn()+A_II'

S_EE.delay = '1.5*ms'
S_EI.delay = '0.8*ms'
S_IE.delay = '0.8*ms'
S_II.delay = '0.8*ms'

S_EE.taupre = S_EE.taupost = 20*ms
S_EE.Apre = 0.1

S_EI.taupre = S_EI.taupost = 20*ms
S_EI.Apre = 0.1

S_IE.taupre = S_IE.taupost = 20*ms
S_IE.Apre = 0.01

S_II.taupre = S_II.taupost = 20*ms
S_II.Apre = 0.01
# ----------------------------------------------------------

# 设置监视器 ------------------------------------------------
state_g_ex = StateMonitor(G_ex, (['I', 'v']), record=True)
state_g_inh = StateMonitor(G_inh, (['I', 'v']), record=True)
state_g_read = StateMonitor(G_readout, (['I', 'v']), record=True)

spike_g_ex = SpikeMonitor(G_ex, name='spike_monitor_exc')
spike_g_inh = SpikeMonitor(G_inh, name='spike_monitor_inh')
# ----------------------------------------------------------

# ------create network-------------
net = Network(collect())
monitor = Monitor(state_g_ex, state_g_inh, state_g_read, spike_g_ex, spike_g_inh)

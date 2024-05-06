# -*- coding = utf-8 -*-
# @Time: 2024/3/16 15:26
# @File: Input.py
from encoder import ModifiedHSA
import numpy as np
import pandas as pd

input_data = pd.read_csv('sunspot.csv')
time = input_data['Month']
data = np.array(input_data['Sunspots'])
enc = ModifiedHSA()
enc.filter_amp = 100
enc.threshold = 52
spikes = enc.encode(data)
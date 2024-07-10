import numpy as np
import pandas as pd
from pyEDM import *

# Load Data
data = np.load('assimilation_data/type_I.npy',allow_pickle=True)[()]
# Choose downsample rate
dsr = 5
# Create pandas dataframe with time differenced states
v = data['Train']['V'][:-1][::dsr]
t = np.arange(1,len(v))
V_diff = v[1:]-v[:-1]
df = {'Time':t,'V_diff':V_diff}
df = pd.DataFrame(df)
# Choose time indicies to find optimal embedding dimension for a given tau

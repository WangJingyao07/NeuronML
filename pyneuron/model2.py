'''
PyNeuron tutorial. 
More about inserting conductance mechanisms
'''
import pickle

import neuron
from neuron import h

import numpy as np
from pylab import *

#h.celsius = 37# reset temperture to see effects

cell = h.Section() # create cell
cell.insert('hh') # insert the Hodgkin-Huxley channel

# change the hh settings
cell.el_hh = -40 # -54.3 #default: -54.3 mV
cell.gkbar_hh = 0.01 # 0.02 #default:0.036 S/cm2

vrec = h.Vector()
trec = h.Vector() 
vrec.record(cell(0.5)._ref_v) # record voltage at center of the Section
trec.record(h._ref_t) # record time

# record the potassium and sodium currents
ik_rec = h.Vector()
ik_rec.record(cell(0.5)._ref_ik)  # pointer to potassium current at center
ina_rec = h.Vector()
ina_rec.record(cell(0.5)._ref_ina) # pointer to sodium current at center

h.finitialize(-60) # initializes voltage
h.dt = 0.025 # timestep in milliseconds
neuron.run(100) # run the simulation for 100 milliseconds

# plot the output
subplot(2,1,1)
plot(np.array(trec),np.array(vrec),label='Vm')
ylabel('Vm (mV)')
xlabel('Time (ms)')
xlim((0,h.tstop))
legend()

subplot(2,1,2)
plot(np.array(trec),np.array(ik_rec),'r',label='$I_{k}$')
plot(np.array(trec),np.array(ina_rec),'g',label='$I_{Na}$') 
xlabel('Time (ms)')
xlim((0,h.tstop))
legend()
tight_layout()
show()

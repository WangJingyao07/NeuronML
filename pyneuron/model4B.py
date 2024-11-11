'''
PyNeuron tutorial. 
Create two basic conductance-based models and connect them via a synapse
'''
import neuron
from neuron import h

import numpy as np
from pylab import *

h.load_file("stdrun.hoc") # for run control

# create two cells
cell1 = h.Section()
cell1.insert('hh')

cell2 = h.Section()
cell2.insert('hh')

trec = h.Vector()
trec.record(h._ref_t) # record time
vrec1 = h.Vector() 
vrec1.record(cell1(0.5)._ref_v) # record voltage from cell1
vrec2 = h.Vector() 
vrec2.record(cell2(0.5)._ref_v) # record voltage from cell 2

# drive cell 1 with a current and connect to cell 2 via a synapse
ic = h.IClamp(cell1(0.5)) # check "help(h.IClamp)"
ic.delay=5 #ms
ic.dur=25 #ms
ic.amp=5 # nA, 5 -> 1 spike, 10 -> regular firing

syn = h.Exp2Syn( cell2(0.5)  ) # put a bi-exponential synapse in cell2 (postsynaptic side)
syn.tau1=0.1 # rise
syn.tau2=1.0 # decay

#nc = h.NetCon(cell1(0.5)._ref_v,syn,0,0.025,0.5)
nc = h.NetCon(cell1(0.5)._ref_v,syn,0,10,0.5) # connect cell1 to target (syn in cell2)
# 0 is voltage threshold for synaptic event generation, 10 is delay between threshold crossing
# and event delivery, 0.5 is weight - can also change weight via nc.weight[0] = x

h.tstop = 45   # simulation duration (ms)
h.v_init = -60 # voltage initialization (mV)
h.dt = 0.025 # timestep (ms)
h.run() # run sim for h.tstop

# plot output
plot(trec.as_numpy(),vrec1.as_numpy(),'k',linewidth=2)
plot(trec.as_numpy(),vrec2.as_numpy(),'r',linewidth=2)
xlim((0,h.tstop))
xlabel('Time (ms)')
ylabel('Vm (mV)')
legend(['cell1 Vm','cell2 Vm'])
tight_layout()
show()


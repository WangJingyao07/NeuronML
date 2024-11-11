'''
PyNeuron tutorial. 
Create basic conductance-based model
Add a synapse
'''
import neuron
from neuron import h

import numpy as np
from pylab import *

h.load_file("stdrun.hoc") # for run control

cell = h.Section() # create the cell
cell.insert('hh')  # put in the Hodgkin-Huxley channel

vrec = h.Vector()
trec = h.Vector() 
vrec.record(cell(0.5)._ref_v) # record voltage from center
trec.record(h._ref_t) # record time

#insert a synapse with a spikegen
syn = h.Exp2Syn( cell(0.5)  ) # bi-exponential synapse (rise and decay time)
syn.tau1=0.1 # rise time constant (milliseconds)
syn.tau2=1.0 # decay time constant (milliseconds)
syn.e = 0  # reversal potential
stim = h.NetStim() # provides stimulus to the synapse
stim.start = 0 # start time of stimulus
stim.interval = 50 # interval between the inputs
stim.number=1000 # number of inputs provided
stim.noise=1 # how noisy the inputs are (between 0 and 1)
# netcon: source, target,threshold (not used here),delay, weight
nc = h.NetCon(stim,syn,0,0.025,0.5) 

# set up recording of spikes
spikes_rec = h.Vector() # this will store the spike times
ncs = h.NetCon(cell(0.5)._ref_v, None,sec=cell)
# cell(0.5)._ref_v voltage-crossing triggers event, note there's no target (None) since just recording spikes
ncs.record(spikes_rec)

h.v_init = -60 # initialization voltage
h.dt = 0.025 # timestep in milliseconds
h.tstop = 1e3 # 1e3 millisecond runtime
h.run() # run simulation for h.tstop

plot(trec.as_numpy(),vrec.as_numpy(),'k',linewidth=2)
plot(spikes_rec.as_numpy(),np.tile(0,len(spikes_rec)),'ro',markersize=10)
xlim((0,h.tstop))
xlabel('Time (ms)')
ylabel('Vm (mV)')
tight_layout()
show()

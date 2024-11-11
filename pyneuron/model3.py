'''
PyNeuron tutorial. 
Create basic conductance-based model
Simulate current injections
'''
import neuron
from neuron import h
h.load_file("stdrun.hoc") # for run control

import numpy as np
from pylab import * # for drawing

cell = h.Section()
cell.insert('hh') # create the cell

vrec = h.Vector()
trec = h.Vector() 
vrec.record(cell(0.5)._ref_v) # record voltage from center
trec.record(h._ref_t) # record time

"""
# create a step current clamp at center of cell
ic = h.IClamp(cell(0.5)) # check "help(h.IClamp)"
ic.delay=200 #ms
ic.dur=100 #ms
ic.amp=2 # nA, 5 -> 1 spike, 10 -> regular firing
"""

# insert a current clamp with continuously fluctuating value
# inject a continues current
h.tstop = 1e3
h.dt = 0.025
AMP = 20 # scales amplitude of current clamp, 4.5 produces sub-threshold response, 20 causes soma spikes normally, 
         # change values to see effect
np.random.seed(21051982)
noise = np.random.rand(h.tstop/h.dt+1)*AMP
icc = h.IClamp(cell(0.5))
icc.delay=0 # start current clamp right away
icc.dur=1e9 # this means the value is played into for simulation duration
nvec = h.Vector( noise )
nvec.play(icc._ref_amp,h.dt)#updates icc's amplitude during the simulation using nvec's values

h.v_init = -60 # this is initialization voltage used by h.run() -- calls finitialize
h.run() # run simulation for duration in h.tstop

# plot output
subplot(1,2,1)
plot(trec.as_numpy(),vrec.as_numpy()) # as_numpy is more efficient than converting/copying Vector to numpy format
xlim((0,h.tstop))
xlabel('Time (ms)')
ylabel('Vm (mV)')
tight_layout()

subplot(1,2,2)
plot(trec.as_numpy(),nvec.as_numpy()) # display part of the noisy current-injection amplitude
xlim((0,h.tstop))
xlabel('Time (ms)')
ylabel('i (nA)')
xlim((0,10))

show()

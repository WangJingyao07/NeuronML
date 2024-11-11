'''
PyNeuron tutorial. 
Create a dendrite & soma model (a model with spatial extent)
'''
import numpy as np
from pylab import *

import neuron
from neuron import h

import sys

h.load_file("stdrun.hoc") # for run control

# h.celsius = 37

useHH = False # True # False # whether to insert an HH channel in the soma - try toggling it
if len(sys.argv) > 1: useHH = bool(int(sys.argv[1]))

# create the soma and set its properties
soma = h.Section()
if useHH:
  soma.insert('hh') # add Hodgkin-Huxley channel so can get a spike
else:
  soma.insert('pas')
  soma.g_pas = 1.0 / 25000.0 # leak conductance
  soma.e_pas=-60  # leak reversal potential

soma.L = 20    # 20 microns long
soma.diam = 20 # 20 microns wide
soma.nseg = 1  # 1 compartment
#soma.Ra = 100  # axial resistivity

# create the dendrite and set its properties
dend = h.Section()
dend.insert('pas')
dend.L = 1000   # 1000 microns long
dend.diam = 4   # 4 microns wide
dend.nseg = 21  # 21 compartments
#dend.Ra = 100   # axial resistivity
dend.g_pas = 1.0 / 25000.0 # leak conductance
dend.e_pas=-60  # leak reversal potential

# connect the dend to the soma
dend.connect(soma,0.5,0) # soma(0.5) -> dend(0)

# provides a stimulus
stim = h.NetStim()
stim.start = 20 # start time in ms
stim.number = 1 # 1 input
stim.noise = 0  # not noisy

h.dt = 0.025 # timestep in milliseconds
h.v_init = -65 # initialization voltage
h.tstop = 80 # simulation duration in milliseconds

# recording
vdend0,vdend1,vsoma = h.Vector(),h.Vector(),h.Vector()
trec = h.Vector() 
vsoma.record(soma(0.5)._ref_v) # record voltage from middle of soma (where it connects to dend)
vdend0.record(dend(0.0)._ref_v) # record voltage from proximal end of dend (closest to soma)
vdend1.record(dend(1.0)._ref_v) # record voltage from distal end of dend (farthest from soma)
trec.record(h._ref_t) # record time variable

csm=cm.ScalarMappable(cmap=cm.jet); csm.set_clim((0,1)); # for coloring lines

minv,maxv = 1e9,-1e9 # min/max voltage for setting plot limits

# move the synapse along the dend, provide a stimulus, and look at the output
for p in np.linspace(0,1,11) :
  syn = h.Exp2Syn(dend(p)) # put synapse on dend at location p
  syn.tau1 = 0.05 # rise time
  syn.tau2 = 5 # decay time
  syn.e = 0 # reversal potential
  nc = h.NetCon(stim,syn,0,0.025,0.05)
  h.run() 
  minv=min(minv,vdend0.min(),vdend1.min(),vsoma.min())
  maxv=max(maxv,vdend0.max(),vdend1.max(),vsoma.max()) # get min,max voltages in both sections
  subplot(1,3,1)
  title('dend(1.0).v') # distal end of dend (furthest from soma)
  xlabel('Time (ms)'); ylabel('Vm (mV)')  
  plot(trec.as_numpy(),vdend1.as_numpy(),color=csm.to_rgba(float(p)/(1.0)),linewidth=2)
  xlim((0,h.tstop)); ylim((minv,maxv))
  subplot(1,3,2)
  title('dend(0.0).v') # proximal end of dend (nearest to soma)
  xlabel('Time (ms)'); 
  plot(trec.as_numpy(),vdend0.as_numpy(),color=csm.to_rgba(float(p)/(1.0)),linewidth=2,label=str(p))
  xlim((0,h.tstop)); ylim((minv,maxv))
  subplot(1,3,3)
  title('soma(0.5).v') # soma voltage at location connected to proximal end of dend
  plot(trec.as_numpy(),vsoma.as_numpy(),color=csm.to_rgba(float(p)/(1.0)),linewidth=2)#,label='pos='+str(p))
  xlim((0,h.tstop)); ylim((minv,maxv))
  xlabel('Time (ms)')
  del syn,nc # garbage collection is a problem: therefore, remove all added mechanisms in your model

subplot(1,3,2)
legend(loc=0) # draw the legend
tight_layout()
show()

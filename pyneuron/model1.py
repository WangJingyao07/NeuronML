'''
PyNeuron tutorial. 
Create basic conductance-based model
Set up recording of parameters to save and plot the data
'''
import pickle # for saving/reading data

import neuron
from neuron import h # hoc interpreter

import numpy as np 
from pylab import * # for drawing

#h.celsius = 37 # set the temperature

cell = h.Section() # create a section (cable)
cell.insert('hh')  # insert a Hodgkin-Huxley channel

vrec = h.Vector() # setup recording Vectors
trec = h.Vector() 
vrec.record(cell(0.5)._ref_v) # record voltage from middle (0.5) of the Section
trec.record(h._ref_t) # record time variable

h.finitialize(-60) # voltage at initialization, in mV
h.dt = 0.025 # 0.025 millisecond timestep
neuron.run(1000) # run simulation for 1000 milliseconds

plot(np.array(trec),np.array(vrec)) # plot the output using matplotlib
xlim((0,h.tstop)) 
xlabel('Time (ms)')
ylabel('v(0.5)')
show()

data = {}
data['vm'] = np.array(vrec)
data['t'] = np.array(trec)
pickle.dump(data,open('data_model1.pkl','w')) # save the output



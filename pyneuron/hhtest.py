# python -i hhtest.py

from pylab import * # for drawing

from neuron import h # import neuron's hoc interpreter
h.load_file("stdrun.hoc") # for run control

h.dt = 0.025 # set some parameters - dt is time-step in milliseconds
h.tstop = 50 # tstop is duration of simulation in milliseconds
#h.celsius = 37# default temperature is 6.3 for squid

soma = h.Section() # make a section
soma.L = 30 # length in microns 
soma.diam = 1 # diameter in microns
soma.nseg = 1 # number of segments (compartments)

# add a hodgkin-huxley channel (includes leak)
soma.insert('hh')

vt = h.Vector()
vt.record(h._ref_t) # record time variable

vvolt  = h.Vector(int(h.tstop/h.dt)+1) # make a Vector 
vvolt.record(soma(0.5)._ref_v)  # to record voltage from the section

vm, vh, vn = [h.Vector() for x in xrange(3)] # record the m,h,n gating variables
vm.record(soma(0.5).hh._ref_m)
vh.record(soma(0.5).hh._ref_h)
vn.record(soma(0.5).hh._ref_n)

ic = h.IClamp(0.5,soma) # make a current clamp, with a depolarizing current inj.
ic.amp = 0.05 # amplitude of current injected (nA)
ic.dur = 10 # duration of current clamp (ms)
ic.delay = 10 # delay until current clamp is turned on (ms)

h.run() # run the simulation

subplot(1,2,1);
plot(vt.as_numpy(),vvolt.as_numpy());
xlabel('Time (ms)');
ylabel('Vm (mV)');
title('Voltage')
xlim((0,h.tstop))

subplot(1,2,2);
plot(vt.as_numpy(),vm.as_numpy(),'r')
plot(vt.as_numpy(),vh.as_numpy(),'g')
plot(vt.as_numpy(),vn.as_numpy(),'b')
xlabel('Time (ms)'); 
title('HH state variables')
legend(['m','h','n'])
xlim((0,h.tstop))

tight_layout()
show()


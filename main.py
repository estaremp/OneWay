#import qutip
from qutip import *
from scipy import *
from scipy.linalg import logm, expm
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#used operations
I = qeye(2)
X = sigmax()
Y = sigmay()
Z = sigmaz()

#parameters
eps_1 = 0.4
eps_2 = 0.9
num_iter = 10

#write +,-,0,1 states
Plus = (basis(2,0) + basis(2,1)).unit()
Minus = (basis(2,0) - basis(2,1)).unit()
Zero = basis(2,0)
One = basis(2,1)

#write state to distillate
Bell_d = (tensor(Zero,Zero) + eps_1*tensor(One,One)).unit()

#write cluster state
CS = (tensor(Plus,Zero,Zero,Plus)+tensor(Plus,Zero,One,Minus)+tensor(Minus,One,Zero,Plus)+tensor(Minus,One,One,Minus)).unit()

State = ket2dm(Bell_d)
Measure = tensor([qeye(2),qeye(2),ket2dm(basis(2,0)),ket2dm(basis(2,0))])+tensor([qeye(2),qeye(2),ket2dm(basis(2,1)),ket2dm(basis(2,1))])
Rotation = tensor([rx(pi/2),rx(-pi/2),rx(pi/2),rx(-pi/2)])
BCNOT = cnot(4,0,2)*cnot(4,1,3)
print(concurrence(State))
for i in range(num_iter):

    State_tot = (tensor(State,ket2dm(Bell_d)))
    State_rot = Rotation*State_tot*Rotation.dag()
    State_cnot = BCNOT*State_rot*BCNOT.dag()
    State_meas = (Measure*State_cnot*Measure.dag())/(Measure.dag()*Measure*State_cnot).tr()

    State_red = State_meas.ptrace([0,1])
    State = State_red

    print(concurrence(State))
import math

import numpy as np
from math import exp, cos, sin, pow
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def test(y, t):
    return np.array([math.exp(2*t)*y[1], -math.exp(-2*t)*y[0]])

def polet(y, t, m, C, rho, S, g, V0, tetha0):
    return np.array([y[2]*cos(y[3]), y[2]*sin(y[3]), (-C*rho*S*y[2]**2)/(2*m) - g*sin(y[3]), -g*cos(y[3])/y[2]])

m=45
C=0.25
rho=1.29
S=0.35
g=9.81
V0=60

tetha0=0.75
tetha1=1.2
tetha2=1.5
tetha3=0.5
tetha4=1.25
tetha5=1.55
tetha6=0.25
tetha7=1.0

y0 = [1.0,1.0]
y0polet=[0,0,V0,tetha0]
y0polet1=[0,0,V0,tetha1]
y0polet2=[0,0,V0,tetha2]
y0polet3=[0,0,V0,tetha3]
y0polet4=[0,0,V0,tetha4]
y0polet5=[0,0,V0,tetha5]
y0polet6=[0,0,V0,tetha6]
y0polet7=[0,0,V0,tetha7]

def rungekutta2(f, y0polet, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0polet)))
    y[0] = y0polet
    for i in range(n - 1):
        h = t[i+1] - t[i]
        y[i+1] = y[i] + h * f(y[i] + f(y[i], t[i], *args) * h / 2., t[i] + h / 2., *args)
    return y


t=np.linspace(0,5.5,100)
sol1=rungekutta2(polet, y0polet3, t, args=(m, C, rho, S, g, V0, tetha3))
plt.plot(sol1[:,0], sol1[:,1], label=r'$\theta_0=30°$')


t=np.linspace(0,7.67,100)
sol1=rungekutta2(polet, y0polet, t, args=(m, C, rho, S, g, V0, tetha0))
plt.plot(sol1[:,0], sol1[:,1], label=r'$\theta_0=40°$')

t=np.linspace(0,9.35,100)
sol1=rungekutta2(polet, y0polet7, t, args=(m, C, rho, S, g, V0, tetha7))
plt.plot(sol1[:,0], sol1[:,1], label=r'$\theta_0=50°$')

t=np.linspace(0,10.35,100)
sol1=rungekutta2(polet, y0polet1, t, args=(m, C, rho, S, g, V0, tetha1))
plt.plot(sol1[:,0], sol1[:,1], label=r'$\theta_0=60°$')

t=np.linspace(0,10.525,100)
sol1=rungekutta2(polet, y0polet4, t, args=(m, C, rho, S, g, V0, tetha4))
plt.plot(sol1[:,0], sol1[:,1], label=r'$\theta_0=70°$')
t=np.linspace(0,11.05,100)
sol1=rungekutta2(polet, y0polet2, t, args=(m, C, rho, S, g, V0, tetha2))
plt.plot(sol1[:,0], sol1[:,1], label=r'$\theta_0=80°$')


print(max(sol1[:,0]))
print(max(sol1[:,1]))

plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
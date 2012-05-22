__author__ = 'ohaas'

import numpy as ny

N=ny.zeros((7,7))
M=ny.zeros((7,7))
v2_t=ny.ones((7,7))
R=2
s=v2_t.shape

a = ny.arange(R,s[0]-R)
(x,y) = ny.meshgrid(a,a)
(x,y) = (x.flatten(),y.flatten())

t1=ny.arange(s[0])
t2=ny.arange(R)
t3,t4 = ny.meshgrid(t1,t2)
t3,t4 = t3.flatten(),t4.flatten()

t5 = ny.concatenate((ny.arange(0,R),ny.arange(s[0]-R, s[0])))
t6,t7 = ny.meshgrid(t5,a)
t6,t7 = t6.flatten(),t7.flatten()


t8=ny.arange(s[0])
t9=ny.arange(s[0]-R, s[0])
t10,t11 = ny.meshgrid(t8,t9)
t10,t11 = t10.flatten(),t11.flatten()


x1 = ny.concatenate((ny.tile(0.0,len(x)),t3,t6,t10))
y1 = ny.concatenate((ny.tile(0.0,len(x)),t4,t7,t11))

for i in ny.arange(len(x1)):
    if i in ny.arange(len(x)):
        c = ny.arange(x[i]-R,x[i]+R+1)
        (m,n)=ny.meshgrid(c,c)
        m=m.flatten()
        n=n.flatten()
        N[x[i],y[i]]=ny.sum(v2_t[m,n])
    else:
        N[x1[i],y1[i]]=(((R*2)+1)**2)*v2_t[x1[i],y1[i]]

M=N/(((R*2)+1)**2)

print N
print (((R*2)+1)**2)
print M
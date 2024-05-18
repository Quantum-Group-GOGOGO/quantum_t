import numpy as np
import matplotlib.pyplot as plt

def uniqueish_color(value=0.0e-4):
    """There're better ways to generate unique colors, but this isn't awful."""
    if value>0.0:
        return plt.cm.gist_ncar(0.7)
    else:
        return plt.cm.gist_ncar(0.1)


result=np.loadtxt(open("evaluation.csv","rb"),delimiter=',',skiprows=0)
st=result[:-1,0:3]
end=result[1:,0:3]
xy = (np.random.random((10, 2)) - 0.5).cumsum(axis=0)

fig, ax = plt.subplots()
for start, stop in zip(st, end):
    x, y , c= zip(start, stop)
    ax.plot(x, y, color=uniqueish_color(c[0]))
plt.show()
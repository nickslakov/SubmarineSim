import vector as vec
import sub as uBoat
import RNN
import numpy as np
import Tkinter as tk
import time
import random as rng

magnitude = 1.0  #Max value of fluid flow (m/s)
deltaT = 0.05    #Timestep (s)
timer = 0
height = 750 #Height and width of window
width = 1000
WATER_DELTA = 0.5
WATER_CLIP = 2
ITERATIONS = 5000


#Vector components are computed using a uniform random variable
waterFlow = vec.Vector(np.random.uniform(-magnitude,magnitude),np.random.uniform(-magnitude,magnitude))
sub = uBoat.Sub() #Initialize the submarine instance

print waterFlow.x, waterFlow.y

f = open('Output.txt', 'w')
f.write(str(waterFlow.x) + ' ' + str(waterFlow.y))
f.close()

#Initialize the simulation window
#window = tk.Tk()
#c = tk.Canvas(window, bg = "grey", height = height, width = width )
#c.pack()

#shape = c.create_polygon(25, 50, -25, 50, -25, -50, 25, -50, fill="orange")
x = sub.position.x
y = sub.position.y

data = list()

def writeFile(vals):
    f = open('Output.txt', 'a')
    for val in vals:
        f.write(str(val.x) + ' ' + str(val.y) + ' ')
    f.write('\n')
    f.close()

i = 0

while i < ITERATIONS:
    #c.coords(shape, x + 25, y + 50, x - 25, y + 50, x - 25, y - 50, x + 25, y - 50)
    #c.update()
    data.append(sub.update(deltaT, waterFlow))
    #writeFile(data[i])
    data[i] += [waterFlow]
    waterFlow.x += 2 * (rng.random() - 0.5) * WATER_DELTA
    waterFlow.y += 2 * (rng.random() - 0.5) * WATER_DELTA

    if waterFlow.x > WATER_CLIP:
        waterFlow.x = WATER_CLIP
    if waterFlow.y > WATER_CLIP:
        waterFlow.y = WATER_CLIP
    #x = sub.position.x
    #y = sub.position.y

    #if i % 50 == 0:
     #   print sub.motorThrust.x, sub.motorThrust.y

    #time.sleep(deltaT)
    i += 1

rnn(data)


#window.mainloop()







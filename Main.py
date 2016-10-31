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
NUM_BATCHES = 20
BATCH_SIZE = 10
SEQUENCE_LENGTH = 50


#Vector components are computed using a uniform random variable
waterFlow = vec.Vector(np.random.uniform(-magnitude,magnitude),np.random.uniform(-magnitude,magnitude))

subs = []

for i in range(NUM_BATCHES*BATCH_SIZE):
    subs.append(uBoat.Sub()) #Initialize the submarine instances

print waterFlow.x, waterFlow.y

f = open('Output.txt', 'w')
f.write(str(waterFlow.x) + ' ' + str(waterFlow.y))
f.close()

#Initialize the simulation window
#window = tk.Tk()
#c = tk.Canvas(window, bg = "grey", height = height, width = width )
#c.pack()

#shape = c.create_polygon(25, 50, -25, 50, -25, -50, 25, -50, fill="orange")

data = np.zeros((NUM_BATCHES,BATCH_SIZE,SEQUENCE_LENGTH,6))

def writeFile(vals):
    f = open('Output.txt', 'a')
    for val in vals:
        f.write(str(val.x) + ' ' + str(val.y) + ' ')
    f.write('\n')
    f.close()

i = 0

for i in range(NUM_BATCHES):
    for j in range(BATCH_SIZE):
       for sub in subs:
           for k in range(SEQUENCE_LENGTH):
                current_positionVec, current_thrustVec = sub.update(deltaT,waterFlow)
                data_timestep = [current_positionVec.x, current_positionVec.y,current_thrustVec.x, current_thrustVec.y, waterFlow.x, waterFlow.y]

                #writeFile(data[i])
                data[i][j][k] = data_timestep

                waterFlow.x += 2 * (rng.random() - 0.5) * WATER_DELTA
                waterFlow.y += 2 * (rng.random() - 0.5) * WATER_DELTA

                if waterFlow.x > WATER_CLIP:
                    waterFlow.x = WATER_CLIP
                if waterFlow.y > WATER_CLIP:
                    waterFlow.y = WATER_CLIP



print "Calling rnn"
RNN.rnn(data)


#window.mainloop()







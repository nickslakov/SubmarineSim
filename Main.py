import vector as vec
import sub as uBoat
import RNN
import numpy as np
import Tkinter as tk
import time
import random as rng
import matplotlib.pyplot as plt

deltaT = 0.01    #Timestep (s)
timer = 0
height = 750 #Height and width of window
width = 1000
NUM_SUBS = 10000
SEQUENCE_LENGTH = 35

subs = []

for i in range(NUM_SUBS):
    subs.append(uBoat.Sub())


#f = open('Output.txt', 'w')
#f.write(str(waterFlow.x) + ' ' + str(waterFlow.y))
#f.close()

#Initialize the simulation window
#window = tk.Tk()
#c = tk.Canvas(window, bg = "grey", height = height, width = width )
#c.pack()

#shape = c.create_polygon(25, 50, -25, 50, -25, -50, 25, -50, fill="orange")

data = np.zeros((NUM_SUBS,SEQUENCE_LENGTH,6))

def writeFile(vals):
    f = open('Output.txt', 'a')
    for val in vals:
        f.write(str(val.x) + ' ' + str(val.y) + ' ')
    f.write('\n')
    f.close()

i = 0
for sub in subs:
    for j in range(SEQUENCE_LENGTH):
        current_positionVec, current_thrustVec, current_waterFlow = sub.update(deltaT)
        data_timestep = [10*current_positionVec.x, 10*current_positionVec.y,current_thrustVec.x, current_thrustVec.y, current_waterFlow.x, current_waterFlow.y]

        #writeFile(data[i])
        data[i,j,:] = data_timestep
    i += 1


fig, ax = plt.subplots()
ax.plot(data[0,:,0], data[0,:,1], '.b-')
ax.plot(data[1,:,0], data[1,:,1], '.r-')
ax.plot(data[2,:,0], data[2,:,1], '.g-')
plt.show()

#print "Calling rnn"
RNN.rnn(data)


#window.mainloop()







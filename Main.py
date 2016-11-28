import sub as uBoat
import RNN
import numpy as np
import matplotlib.pyplot as plt

deltaT = 0.01    #Timestep (s)
NUM_SUBS = 10000
SEQUENCE_LENGTH = 35
SCALE_FACTOR = 10

subs = []

for i in range(NUM_SUBS):
    subs.append(uBoat.Sub())

data = np.zeros((NUM_SUBS,SEQUENCE_LENGTH,6))

#Generate data for all subs.

i = 0
for sub in subs:
    for j in range(SEQUENCE_LENGTH):
        current_positionVec, current_thrustVec, current_waterFlow = sub.update(deltaT)
        data_timestep = [SCALE_FACTOR*current_positionVec.x, SCALE_FACTOR*current_positionVec.y,current_thrustVec.x, current_thrustVec.y, current_waterFlow.x, current_waterFlow.y]

        data[i,j,:] = data_timestep
    i += 1

#Plot some example sequences
fig, ax = plt.subplots()
ax.plot(data[0,:,0], data[0,:,1], '.b-')
ax.plot(data[1,:,0], data[1,:,1], '.r-')
ax.plot(data[2,:,0], data[2,:,1], '.g-')
plt.show()

#Call RNN
RNN.rnn(data)










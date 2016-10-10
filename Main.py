import vector as vec
import sub as uBoat
import numpy as np
import Tkinter as tk
import time

magnitude = 1.0  #Max value of fluid flow (m/s)
deltaT = 0.05    #Timestep (s)
timer = 0
height = 750 #Height and width of window
width = 1000

#Vector components are computed using a uniform random variable
waterFlow = vec.Vector(np.random.uniform(-magnitude,magnitude),np.random.uniform(-magnitude,magnitude))
sub = uBoat.Sub() #Initialize the submarine instance

print waterFlow.x, waterFlow.y

#Initialize the simulation window
window = tk.Tk()
c = tk.Canvas(window, bg = "grey", height = height, width = width )
c.pack()

shape = c.create_polygon(25, 50, -25, 50, -25, -50, 25, -50, fill="orange")
x = sub.position.x
y = sub.position.y

while True:
    c.coords(shape, x + 25, y + 50, x - 25, y + 50, x - 25, y - 50, x + 25, y - 50)
    c.update()
    sub.update(deltaT, waterFlow)
    x = sub.position.x
    y = sub.position.y
    time.sleep(deltaT)

window.mainloop()
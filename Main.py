import vector as vec
import sub as uBoat
import numpy as np
import Tkinter as tk

magnitude = 2.0  #Max value of fluid flow (m/s)
deltaT = 0.01    #Timestep
timer = 0
height = 250 #Height and width of window
width = 300

#Vector components are computed using a uniform random variable
waterFlow = vec.Vector(np.random.uniform(-magnitude,magnitude),np.random.uniform(-magnitude,magnitude))
Sean = uBoat.Sub() #Initialize the submarine

#Initialize the simulation window
window = tk.Tk()
c = tk.Canvas(window, bg = "grey", height = height, width = width )
c.pack()

def updatedisplay(sub, waterFlow, window):
    c.create_polygon(25, 50, -25, 50, -25, -50, 25, -50, fill="orange")
    sub.update(deltaT, waterFlow)
    window.after(10, updatedisplay(sub, waterFlow))

window.after(0, updatedisplay(Sean, waterFlow))
window.mainloop()
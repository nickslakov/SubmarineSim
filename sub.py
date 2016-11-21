import vector as vec
import math
import random as rng
import numpy as np
class Sub:

    def __init__(self):
        self.position = vec.Vector(0.0,0.0)
        self.velocity = vec.Vector(0.0,0.0)
        self.acceleration = vec.Vector(0.0,0.0)
        self.theta = 0.0
        self.omega = 0.0
        self.alpha = 0.0
        self.mass = 9 #kilos
        self.shape = [vec.Vector(0.25,0.5), vec.Vector(-0.25,0.5), vec.Vector(-0.25,-0.5), vec.Vector(0.25,-0.5)]
        self.I = self.calcI()
        self.height = 1
        self.width = 0.4
        self.depth = 0.25
        self.WATER_CLIP = 0.3
        self.WATER_DELTA = 0.05
        self.motorThrust = vec.Vector(0.0,0.0) #Newtons
        self.waterFlow = vec.Vector(np.random.uniform(-self.WATER_CLIP,self.WATER_CLIP),np.random.uniform(-self.WATER_CLIP,self.WATER_CLIP))
        self.MOTOR_DELTA = 1
        self.MOTOR_CLIP = 40



    #This is the method that updates the position, velocity, and acceleration of the submarine.
    #WaterFlow is in units of m/s
    def update(self, deltaTime):

        force = self.calcNetForce(self.waterFlow)

        self.acceleration.x = force.x / self.mass
        self.acceleration.y = force.y / self.mass

        self.velocity.x += self.acceleration.x * deltaTime
        self.velocity.y += self.acceleration.y * deltaTime

        self.position.x += self.velocity.x * deltaTime
        self.position.y += self.velocity.y * deltaTime

        self.motorThrust.x += 2 * (rng.random() - 0.5) * self.MOTOR_DELTA
        self.motorThrust.y += 2 * (rng.random() - 0.5) * self.MOTOR_DELTA

        if self.motorThrust.x > self.MOTOR_CLIP:
            self.motorThrust.x = self.MOTOR_CLIP
        if self.motorThrust.y > self.MOTOR_CLIP:
            self.motorThrust.y = self.MOTOR_CLIP

        self.waterFlow.x += 2 * (rng.random() - 0.5) * self.WATER_DELTA
        self.waterFlow.y += 2 * (rng.random() - 0.5) * self.WATER_DELTA

        if self.waterFlow.x > self.WATER_CLIP:
            self.waterFlow.x = self.WATER_CLIP
        if self.waterFlow.y > self.WATER_CLIP:
            self.waterFlow.y = self.WATER_CLIP

        self.theta += self.omega * deltaTime

        self.omega += self.alpha * deltaTime

        moment = self.calcNetMoment()

        self.alpha = moment / self.I

        return [self.position, self.motorThrust, self.waterFlow]

    def calcNetForce(self, waterFlow):

        #Perform rotate the waterflow vecor into the submarine co-ordinate system
        waterFlow.rotate(self.theta)
        waterForce = vec.Vector(waterFlow.x * math.fabs(waterFlow.x) * self.height * self.depth * 1000, waterFlow.y * math.fabs(waterFlow.y)  * self.width * self.depth* 1000)
        netForce = waterForce.add(self.motorThrust)
        netForce.rotate(-self.theta)
        waterFlow.rotate(-self.theta)

        return netForce

    # 0 as long as the sub is symmetrical about the origin
    def calcNetMoment(self):
        return 0.0


    #Calculates the area moment of inertia of the shape using the general plane polygon formula
    def calcI(self):
        numerator = 0.0
        denominator = 0.0

        i = 0
        while i < len(self.shape):
            j = i+1
            if j == len(self.shape):
                j = 0

            numerator += math.fabs(self.shape[j].cross(self.shape[i])) * (self.shape[i].dot(self.shape[i]) + self.shape[i].dot(self.shape[j]) + self.shape[j].dot(self.shape[j]))
            denominator += math.fabs(self.shape[j].cross(self.shape[i]))
            i += 1

        return self.mass * numerator / (6 * denominator)




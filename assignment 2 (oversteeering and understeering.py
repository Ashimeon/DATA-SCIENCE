# Assignment 2: 
# Design a program to control speed of a car and a bike independently. Also consider the turning phenomenon of each using stering if its a car and handle if its a bike.


class Vehicle:
    def __init__(self,speed):
        self.speed=speed
class car(Vehicle):
    def __init__(self,speed,steering):
        super().__init__(speed)
        self.steering=steering
    def fast(self):
        if self.speed>80:
            print("you are overspeed. Caution: drive slow!")
        else:
            print("Speed is good")
    def turn(self):
        if self.speed>80 and self.steering>720:
            print("you are oversteering")
            self.speed=80
            self.steering=720
            print("speed decreased to 80km/hr and steering changed to 720degree")
        elif self.speed>80 and self.steering<180:
            print("you are understeering")
            self.speed=80
            self.steering=180
            print("speed decreased to 80km/hr and steering changed to 180degree")

        else:
            print("steering is good")
            if self.speed>80:
                self.speed=80
                print("speed deccreased to 80km/hr")
            
class bike(Vehicle):
    def __init__(self,speed,handle):
        super().__init__(speed)
        self.handle=handle
    def fast(self):
        if self.speed>40:
            print("you are overspeed. Caution: drive slow!")
        else:
            print("speed is good")
    def turn(self):
        if self.speed>40 and self.handle>90:
            print("you are oversteering")
            self.speed=40
            self.handle=90
            print("speed decreased to 40km/hr and handle angle changed to 720degree")

        elif self.speed>40 and self.handle<22:
            print("you are understeering")
            self.speed=40
            self.handle=22
            print("speed decreased to 40km/hr and handle angle changed to 22degree")

        else:
            print("handling is good")
            if self.speed>40:
                self.speed=40
                print("speed deccreased to 40km/hr")
            
def carr():
    x=int(input("enter car speed(km/hr): "))
    y=int(input("enter steering angle(in degree): "))
    print("------------------------------------")

    c=Vehicle(x)
    c=car(x,y)
    c.fast()
    c.turn()
def bikee():
    x=int(input("enter bike speed(km/hr): "))
    y=int(input("enter handle angle(in degree): "))
    print("------------------------------------")
    b=bike(x,y)
    b.fast()
    b.turn()

a=input("enter the vehicle type (car:c or bike:b): ")
if a=='c':
    carr()
if a=="b":
    bikee()


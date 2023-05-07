# Design a program to control speed of a car and a bike independently.
# Also consider the turning phenomenon of each using stering if its a car and handle if its a bike.

class Vehicle:
    def __init__(self):
        self.speed = 0

    def ChangeSpeed(self, x):
        self.speed = x


class Car(Vehicle):
    def __init__(self):
        self.steering = "straight"
        super().__init__()

    def steer(self, side):
        self.steering = side


class Bike(Vehicle):
    def __init__(self):
        self.handle = 'straight'
        super().__init__()

    def move(self, side):
        self.handle = side


while (1):
    print("*************************************")
    print("1.Bike Operations\n2.Car Operations\n3.Quit")
    print("*************************************")
    choice = int(input("Choice: "))
    if (choice == 3):
        break
    elif (choice == 1):
        print("Bike Started.")
        b = Bike()
        while (1):
            print("*************************************")

            print(
                "1.Check Speed\n2.Check Direction\n3.Change Speed\n4.Change Direction\n5.Go Back")
            print("*************************************")

            o = int(input("Choice: "))
            if (o == 5):
                print("Bike turned off")
                break
            elif o == 1:
                print("The Bike speed is", b.speed)
            elif o == 2:
                print("Handle is turned", b.handle)
            elif o == 3:
                b.ChangeSpeed(int(input("Enter Speed:")))
                print("Bike speed changed to", b.speed)
            elif o == 4:
                b.move(input("Enter Direction: "))
                print("Handled turned", b.handle)
    elif (choice == 2):
        print("Car Started.")
        b = Car()
        while (1):
            print("*************************************")

            print(
                "1.Check Speed\n2.Check Direction\n3.Change Speed\n4.Change Direction\n5.Go Back")
            print("*************************************")

            o = int(input("Choice: "))
            print("Car turned off")
            if (o == 5):
                break
            elif o == 1:
                print("The Car speed is", b.speed)
            elif o == 2:
                print("Steering is turned", b.steering)
            elif o == 3:
                b.ChangeSpeed(int(input("Enter Speed:")))
                print("Car speed changed to", b.speed)
            elif o == 4:
                b.steer(input("Enter Direction: "))
                print("Steering turned", b.steering)

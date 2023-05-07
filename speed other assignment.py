#vehicle speed will change depending on the speed_limit defined and the speed set during object creation
#0 means vehicle is to turn right and 1 means to turn left

class Vehicle:
     speed_limit = 50
     def __init__(self,name,speed):
          self.name = name
          self.speed = speed

     @classmethod
     def get_speed_limit(cls):
          return cls.speed_limit

     def __get_speed_change_value(self):
          speed_limit = self.get_speed_limit()
          return abs(self.speed - speed_limit)

     def __change_speed(self,action):
          if action == 1:
               while(self.speed >= self.get_speed_limit()):
                    print(self.speed)
                    self.speed -= 1
          else:
               while(self.speed <= self.get_speed_limit()):
                    print(self.speed)
                    self.speed += 1

     def compare_speed(self):
          if(self.speed > self.get_speed_limit()):
               print(f"{self.name} speed is {self.speed} but speed limit is {self.get_speed_limit()}.\n Activating auto-braking mechanism.")
               print(f"Decreasing speed by: {self.__get_speed_change_value()}")
               self.__change_speed(1)
               
          elif (self.speed < self.get_speed_limit()):
               print(f"{self.name} speed is {self.speed} but speed limit is {self.get_speed_limit()}.\n Activating auto-accelerating mechanism.")
               print(f"Increasing speed by: {self.__get_speed_change_value()}")
               self.__change_speed(0)
          else:
               print("You are moving under the speed limit. Drive at the same speed.")

class Car(Vehicle):
     def __init__(self,name, speed, direction=2):
          super().__init__(name,speed)
          self.direction = direction
     
     def speed_control(self):
          print("\n You are a car.")
          self.compare_speed()

     def direction_control(self):
          if(self.direction == 2):
               print("Keep on going straight.")
          elif (self.direction == 1):
               print(f'Your input is {self.direction}. Changing direction.')
               print("Stering activated to turn right")
          else:
               print(f'Your input is {self.direction}. Changing direction.')
               print("Stering activated to turn left")

class Bike(Vehicle):
     def __init__(self,name,speed, direction = 2):
          super().__init__(name,speed)
          self.direction = direction
     
     def speed_control(self):
          print("\n You are a bike.")
          self.compare_speed()

     def direction_control(self):
          if(self.direction == 2):
               print("Keep on going straight.")
          elif (self.direction == 1):
               print(f'Your input is {self.direction}. Changing direction.')
               print("Handle activated to turn right")
          else:
               print(f'Your input is {self.direction}. Changing direction.')
               print("Handle activated to turn left")

#created car object without need for direction change
car1 = Car('formula1', 45)
car1.speed_control()
car1.direction_control()

#created car object with need for direction change to left
car2 = Car('formula2', 52, 0)
car2.speed_control()
car2.direction_control()

#created bike object without need for direction change
bike1 = Bike('R15', 55)
bike1.speed_control()
bike1.direction_control()

#created bike object with need for direction to right
bike2 = Bike('MT15', 50, 1)
bike2.speed_control()
bike2.direction_control()
class Arm:
    fingers = 4


class Human:
    arm = None
    name = None
    age = 2

    def __init__(self, arm):
        self.arm = arm

    def getname(self):
        self.name = "ff"
        print("my name is name" + self.name)

    def getArm(self):
        print(self.arm.fingers)


me = Human(Arm())
me.getArm()

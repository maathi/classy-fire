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


def run():
    a = 1

    if a == 1:
        df = "hello"
        a = 2
    if a == 2:
        print(df)


run()
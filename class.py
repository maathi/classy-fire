class Human:
    name = None
    age = 2

    def getname(self):
        self.name = "ff"
        print("my name is name" + self.name)


me = Human()
me.getname()


def shoot():
    return 1, 2, 3


a, b, c = shoot()
print(a, b, c)

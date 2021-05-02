class LinearA:
    name = 'LinearA'

    @staticmethod
    def f(x):
        return -x + 2


class QuadraticA:
    name = 'QuadraticA'

    @staticmethod
    def f(x):
        return (x-3)*(x-3) + 1


class QuadraticB:
    name = 'QuadraticB'

    @staticmethod
    def f(x):
        return (x-1)*(x-1)*0.3-1

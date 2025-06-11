def use_value(func):
    def wrapper(self):
        func(self.value)

    return wrapper

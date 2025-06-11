class Kernel:

    def __init__(self, name):
        self.name = name

    @staticmethod
    def node_init_index(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def node_forward(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def node_backward(*args, **kwargs):
        raise NotImplementedError

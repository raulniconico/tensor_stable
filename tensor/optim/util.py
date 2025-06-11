import pickle

class ModelConfig:
    def __init__(self,
                 name: str = None,
                 model: str = None,
                 k: tuple = None,
                 shape: tuple = None,
                 activation: str = None,
                 BN: bool = None,
                 LN: bool = None,
                 DO_rate: float = None,
                 init_factor: float = None,
                 kernel: str = None,
                 deg: int = None,
                 poly: str = None,
                 coor: tuple = None,
                 lr: float = None,
                 decay: float = None,
                 sample_size: int = None,
                 batch_size: int = None
                 ):
        self.name = name
        self.model = model
        self.k = k
        self.shape = shape
        self.activation = activation
        self.BN = BN
        self.LN = LN
        self.DO_rate = DO_rate
        self.init_factor = init_factor
        self.kernel = kernel
        self.deg = deg
        self.poly = poly
        self.coor = coor
        self.lr = lr
        self.decay = decay
        self.sample_size = sample_size
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs):
        return {"name": self.name,
                "model": self.model,
                "k": self.k,
                "shape": self.shape,
                "activation": self.activation,
                "BN": self.BN,
                "LN": self.LN,
                "DO_rate": self.DO_rate,
                "init_factor": self.init_factor,
                "kernel": self.kernel,
                "deg": self.deg,
                "poly": self.poly,
                "coor": self.coor,
                "lr": self.lr,
                "decay": self.decay,
                "sample_size": self.sample_size,
                "batch_size": self.batch_size
                }

def save(filename, dict):
    with open(filename, 'wb') as file:
        pickle.dump(dict, file, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as file:
        dict = pickle.load(file)
        return dict
from typing import Union
import numpy as np

from ..HTD import HT_Btree, Node, NodeTask
from ..HTD import polynomial, POLYFIT, POLYVAL
from ..nuwa.util import svd_random

from ..nuwa import ACTIVATIONS

from .kernel import *


class Optim_Btree(HT_Btree):
    """
    Optim Btree is the heritages HT_Btree, it provides optimization learning methods for learning the transfer tensors
    and factors.
    The Kernel of Optim Btree defines how the values go through the Btree. They are independently of the FBR property of
    the Btree.
    """

    # Available kernels for Btree
    KERNEL = {"HTD": HTD_kernel,
              "single_entry": Single_Entry_Kernel,
              "concatenate": Concatenate_Kernel,
              "multi_single_entry": Multi_Single_Entry_Kernel}

    def __init__(self,
                 k: Union[list, tuple, np.ndarray],
                 shape: Union[list, tuple, np.ndarray],
                 name: str,
                 kernel: str,
                 initialisation: str = "xavier",
                 init_factor: float = 1.0,
                 activation: str = None,
                 BN: bool = False,
                 LN: bool = False,
                 device="cpu",
                 model: str = None,
                 dtype=np.float64):
        """

        :param k: List, the rank of each level of node
        :param shape: List, the (explicit or implicit) shape of each node
        :param name: str, name of the Optim Btree for heritage class
        :param kernel: str, name of the kernel
        :param initialisation: str, initialization of the weights
        :param init_factor: float, an initial factor of the weights, multiply element-wisely to the weights
        :param activation: str, activation function
        :param BN: Batch normalization, not support for SGD
        :param LN: Layer normalization, first test Failure
        :param device: str, device for computation
        :param model: str, (To be Deprecated) model name
        :param dtype:
        """

        super(Optim_Btree, self).__init__(shape=shape, device=device, model=model, dtype=dtype)
        self.k = k
        self.name = name
        self.BN = BN
        self.LN = LN
        self.activation = activation
        self.initialisation = initialisation
        self.init_factor = init_factor

        self.kernel = kernel

    def print(self):
        print(
            f"""
        {self.name}:\n
        shape: {self.shape} \n
        device: {self.device} \n
        model: {self.model} \n
        dtype: {self.dtype} \n
        """)

    @property
    def k(self) -> Union[list, tuple]:
        """
        node rank list
        :return : node rank
        """
        return self._k

    @k.setter
    def k(self, k: Union[list, tuple]):
        """
        node rank list
        :param k: node rank list
        :return: None
        """
        self._k = k

    @property
    def dim(self):
        """
        dimension of the fitting PES
        :return: int, the dimension of the fitting PES
        """
        return np.size(self.shape)

    @dim.setter
    def dim(self, dim):
        """
        dimension of the fitting PES
        :param dim:
        :return:
        """
        return

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: str):
        assert kernel in self.KERNEL, "kernel must be one of: " + ", ".join(self.KERNEL)
        self._kernel = self.KERNEL[kernel]

    @property
    def BN(self) -> bool:
        """
        Batch normalisation
        :return:
        """
        return self._BN

    @BN.setter
    def BN(self, BN: bool = True):
        """

        :param BN: bool, default True, to set using batch normalization
        :return:
        """
        self._BN = BN

    @property
    def LN(self) -> bool:
        """
        Layer normalisation

        :return:
        """
        return self._LN

    @LN.setter
    def LN(self, LN: bool = True):
        """
        Layer normalization

        :param LN: bool, default True, to set using batch normalization
        :return:
        """
        self._LN = LN

    @property
    def activation(self):
        """
        activation function

        :return:
        """
        return self._activation

    @activation.setter
    def activation(self, activation: str = None):
        """
        activation function to be used in training
        (Currently only supported in "HTD kernel")

        :param activation: str, name of activation function
        :return: None
        """
        if activation is not None:
            assert activation in ACTIVATIONS.keys(), f"Activation {activation} is not in {ACTIVATIONS.keys()}"
            self._activation = ACTIVATIONS[activation]()
        else:
            self._activation = None

    @property
    def params(self):
        def node_get_params(node):
            return
        return


    def pretrain(self, btree):
        raise NotImplementedError

    def init_factors(self, A: np.ndarray = None, random_p: int = 10, random_c: int = 100) -> None:
        """
        Initialise the factors of the btree.

        The factors are initialized randomly as matrices.

        First, generate a multinomial random matrix; its mean and std are randomly initialized. Then, this matrix is
         SVD and finally only the left factor of SVD is stocked.

        :param A: np.ndarray, None by default. Target tensor to be optimised.
        :param random_p: int, 10 by default. Random factor of randomised SVD
        :param random_c:
        :return: None
        """

        def multinomial(shape: int, k: int) -> np.ndarray:
            """
            Define multinomial function for random matrix

            :param shape: int, the row size of random matrix
            :param k: int, the column size of random matrix
            :return:
            """
            sigma = np.random.uniform(0, 1)
            mu = np.random.uniform(0, 1)
            # sigma, mu = 1, 0
            x, y = np.meshgrid(np.linspace(-1, 1, k), np.linspace(-1, 1, shape))
            d = np.sqrt(x ** 2 + y ** 2) + np.random.uniform(0, 1) * x * y
            gaus = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
            return gaus

        if A is not None:
            """ Use apriori tensor A to generate basis """
            for node in self.get_leaves():
                # generate A_t
                I_t = node.get_cache("mode")
                I_t_p = [i for i in list(range(len(A.shape))) if i not in I_t]
                shape0 = np.prod(np.array(A.shape)[I_t])
                shape1 = np.prod(np.array(A.shape)[I_t_p])

                for i in range(len(I_t)):
                    A = np.moveaxis(A, I_t[-1], 0)

                A_t = A.reshape((shape0, shape1)).astype(self.dtype)
                # SVD
                rank = self.k[-len(node.get_cache("mode"))]
                U_t, S, VH = svd_random(A_t, rank, random_p, full_matrices=False)

                # update U_t, rank and singular value
                node.update_cache({"U_t": U_t.astype(self.dtype), "rank": U_t.shape[1]})

        else:
            """ Use multinomial generated basis """
            for node in self.get_leaves():
                shape = self.shape[int(node.get_cache("mode"))]
                k = self.k[-len(node.get_cache("mode"))]

                """ 
                Following code is commented cause the random generation of orthogonal
                matrix is problematic.
                """
                # if shape >= k:
                #     random_matrix = np.random.randn(shape, k).astype(self.dtype)
                #     U_t = scipy.linalg.orth(random_matrix)
                #     node.update_cache({"U_t": U_t.astype(self.dtype), "rank": U_t.shape[1]})
                # else:
                #     random_matrix = np.random.randn(k, shape).astype(self.dtype)
                #     U_t = scipy.linalg.orth(random_matrix).T
                #     node.update_cache({"U_t": U_t.astype(self.dtype), "rank": U_t.shape[1]})

                if shape >= k:
                    random_matrix = multinomial(shape, k).astype(self.dtype)
                    U_t, S, VH = svd_random(random_matrix, k, random_p, full_matrices=True)
                    node.update_cache({"U_t": U_t.astype(self.dtype), "rank": U_t.shape[1]})
                else:
                    random_matrix = multinomial(k, shape).astype(self.dtype)
                    U_t, S, VH = svd_random(random_matrix, k, random_p, full_matrices=True)
                    node.update_cache({"U_t": U_t.astype(self.dtype)[:shape, :], "rank": U_t.shape[1]})

    def init_weights(self):
        """
        Initial B_t in Btree.

        :return: None
        """
        def node_init_weights(node: Node, k: tuple):
            # Root node
            if node.is_root():
                rank_l = node.left.get_cache("rank")
                rank_r = node.right.get_cache("rank")
                shape_l = node.left.get_cache("shape")[0]
                shape_r = node.right.get_cache("shape")[0]

                if self.initialisation == "xavier":
                    sigma = np.sqrt(2 / (shape_l * shape_r))
                    B_t = np.random.normal(0, sigma, size=(rank_l * rank_r, 1)) * self.init_factor
                elif self.initialisation == "rankwise":
                    B_t = np.random.randn(rank_l * rank_r, 1) * np.sqrt(1 / (rank_l * rank_r)) * self.init_factor
                elif self.initialisation == "none":
                    B_t = np.ones((rank_l * rank_r, 1)) / (rank_l * rank_r) * self.init_factor
                else:
                    raise Exception("Initialisation method is not supported!")

                node.update_cache({"B_t": B_t, "rank": 1}) # the rank in root is always 1 due to the vectorization
                node.update_cache({"B_t_grad": np.zeros_like(B_t)})

            # Inner node
            elif node.is_inner():
                rank = k[-len(node.get_cache("mode"))]
                rank_l = node.left.get_cache("rank")
                rank_r = node.right.get_cache("rank")
                shape_l = node.left.get_cache("shape")[0]
                shape_r = node.right.get_cache("shape")[0]

                if self.initialisation == "xavier":
                    sigma = np.sqrt(2 / (shape_l * shape_r))
                    B_t = np.random.normal(0, sigma, size=(rank_l * rank_r, rank)) * self.init_factor
                elif self.initialisation == "rankwise":
                    B_t = np.random.randn(rank_l * rank_r, rank) * np.sqrt(1 / (rank_l * rank_r)) * self.init_factor
                elif self.initialisation == "none":
                    B_t = np.ones((rank_l * rank_r, rank)) / (rank_l * rank_r * rank) * self.init_factor
                else:
                    raise Exception("Initialisation method is not supported!")

                node.update_cache({"B_t": B_t, "rank": rank})
                node.update_cache({"B_t_grad": np.zeros_like(B_t)})

        self.distribute_task(
            NodeTask(node_init_weights, self.k), sequence="leaves_to_root", start_node_task=True)

    def init_shape(self):
        """
        Initial the shape (in case sparse, the imaginary shape) of U_t

        :return:
        """
        def node_init_shape(node: Node, k: tuple, shape):
            if node.is_root():
                shape_left = node.left.get_cache("shape")
                shape_right = node.right.get_cache("shape")
                node.update_cache({"shape": (shape_left[0] * shape_right[0], 1)})

            elif node.is_inner():
                rank = k[-len(node.get_cache("mode"))]
                shape_left = node.left.get_cache("shape")
                shape_right = node.right.get_cache("shape")
                node.update_cache({"shape": (shape_left[0] * shape_right[0], rank)})

            elif node.is_leaf():
                node.update_cache({"shape": (shape[int(node.get_cache("mode"))], k[-1])})

            # print(node.get_cache("shape"))
        self.distribute_task(
            NodeTask(node_init_shape, self.k, self.shape), sequence="leaves_to_root", start_node_task=True)


    def init_kernel(self):
        """
        Set the forward and backward propagation from kernel

        :return:
        """
        self.__setattr__("node_forward", self.kernel.node_forward)
        self.__setattr__("node_backward", self.kernel.node_backward)
        self.__setattr__("node_init_index", self.kernel.node_init_index)

    def init(self, A: np.ndarray = None):
        assert self.shape is not None, f"The shape must be defined!"
        assert self.k is not None, f"The k must be defined!"

        # kernel provides node forward, back, and init index functions
        self.init_kernel()

        # initial node mode and the shape of node frame
        self.init_mode()
        self.init_shape()

        # init B_t, U_t
        self.init_factors(A)
        self.init_weights()


    def init_index(self, *args, **kwargs):
        """
        Kernel defined function



        :param args:
        :param kwargs:
        :return:
        """
        self.distribute_task(NodeTask(self.node_init_index, *args, **kwargs), sequence="root_to_leaves",
                             start_node_task=True)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError


class HTD_Optim_Btree(Optim_Btree):

    """
    FORBIDDEN TO USE
    """
    _first = True

    def __init__(self,
                 k: tuple,
                 shape: tuple,
                 device: str = "cpu",
                 kernel: str = "HTD",
                 initialisation: str = "xavier",
                 init_factor: float = 1.0,
                 activation: str = None,
                 BN: bool = False,
                 DO_rate: float = 0.,
                 dtype=np.float64):

        super(HTD_Optim_Btree, self).__init__(name="HTD_Optim_Btree",
                                              k=k,
                                              shape=shape,
                                              kernel=kernel,
                                              initialisation=initialisation,
                                              init_factor=init_factor,
                                              activation=activation,
                                              BN=BN,
                                              device=device,
                                              dtype=dtype,
                                              )

        self.DO_rate = DO_rate

    @classmethod
    def pretrain(cls, btree):
        optim_btree = cls(k=btree.k,
                          shape=btree.shape,
                          activation=btree.activation,
                          BN=btree.BN,
                          DO_rate=btree.DO_rate,
                          dtype=btree.dtype)

        optim_btree.algo = btree.algo
        optim_btree.root = btree.root
        return optim_btree

    @property
    def first(self):
        return self._first

    @first.setter
    def first(self, first: bool) -> None:
        self._first = first

    def forward(self, feature):
        """

        :param feature:
        :return:
        """
        self.init_index(feature)

        """
        In the first forward of the btree, the B_t in the root has been initialied in init() but still
        need to be normalised. The normalisation is based on the size of feature. 
        """
        if self.first:
            if self.initialisation == "xavier":
                B_t = self.root.get_cache("B_t")
                shape_l = self.root.left.get_cache("shape")[0]
                shape_r = self.root.right.get_cache("shape")[0]
                self.root.update_cache({"B_t": B_t *
                                               self.init_factor * np.sqrt(2 / (shape_l * shape_r + feature.shape[0]))})

            elif self.initialisation == "rankwise":
                B_t = self.root.get_cache("B_t")
                rank_l = self.root.left.get_cache("rank")
                rank_r = self.root.right.get_cache("rank")
                self.root.update_cache({"B_t": B_t *
                                               self.init_factor * np.sqrt(2 / (rank_l * rank_r + feature.shape[0]))})
            else:
                raise Exception("Initialisation method is not supported!")

            self.first = False

        """ parse to forward """
        A_predict = self.distribute_task(
            NodeTask(self.node_forward,
                     feature=feature,
                     activation=self.activation,
                     BN=self.BN,
                     LN=self.LN,
                     DO_rate=self.DO_rate,
                     dtype=self.dtype), sequence="leaves_to_root", start_node_task=True)
        return A_predict

    def backward(self) -> None:
        self.distribute_task(
            NodeTask(self.node_backward,
                     feature=None,
                     activation=self.activation,
                     BN=self.BN,
                     LN=self.LN,
                     DO_rate=self.DO_rate,
                     dtype=self.dtype,
                     ), sequence="root_to_leaves", start_node_task=True)

    def predict(self, feature):
        A_predict = self.distribute_task(
            NodeTask(self.node_forward, feature, self.activation, self.BN, DO_rate=0), sequence="leaves_to_root",
            start_node_task=True)
        return A_predict


class HT_FBR_Optim_Btree(Optim_Btree):
    _first = True

    def __init__(self,
                 k: Union[tuple, np.ndarray],
                 shape: Union[tuple, np.ndarray],
                 kernel: str = "concatenate",
                 activation: str = None,
                 BN: bool = False,
                 LN: bool = False,
                 DO_rate: float = 0.,
                 poly: str = "none",
                 deg: int = None,
                 coor: tuple = None,
                 device: str = "cpu",
                 dtype=np.float64,
                 *args,
                 **kwargs):
        """
        The Optimisation btree is an implementation of the trainable btree with HSVD or HT-FBR format. Optim_Btree, not
        like its father class, contains every component for optimisation.

        :param k: Union[tuple, np.ndarray], node rank, tuple or np.ndarray.
        :param shape: Union[tuple, np.ndarray], the shape of the imaginary grid of tensor.
        :param device: str, "cpu" by default. Current accuracy setting only allows "cpu" training
        :param activation:
        :param BN:
        :param DO_rate:
        :param poly:
        :param deg:
        :param coor:
        :param dtype:
        """

        super(HT_FBR_Optim_Btree, self).__init__(k=k,
                                                 shape=shape,
                                                 activation=activation,
                                                 BN=BN,
                                                 LN=LN,
                                                 name="HT-FBR_Optim_Btree",
                                                 device=device,
                                                 kernel=kernel,
                                                 dtype=dtype)
        if coor is not None:
            assert len(coor) == np.size(
                shape), "'coor' should be a list whose length is equal to the dimension of 'shape'"

        self.DO_rate = DO_rate
        self.poly = poly
        self.deg = deg
        self.coor = coor

    @property
    def interval(self) -> list:
        assert self.coor is not None
        inter = []
        for i in range(self.dim):
            inter.append((np.min(self.coor[i]), np.max(self.coor[i])))
        return inter

    @interval.setter
    def interval(self, interval):
        return

    @property
    def first(self):
        return self._first

    @first.setter
    def first(self, first: bool) -> None:
        self._first = first

    @classmethod
    def pretrain(cls, btree):
        """
        Load a pretrained BTree model. It heritages the weight, the shape and other polynomial settings if existed.
        Some of the training parameters, on the contrary, are not preserved like Drop rate, training device etc.

        :param btree: Optim_Btree, the pretrained btree.
        :return: Optim_Btree
        """
        optim_btree = cls(k=btree.k, shape=btree.shape)
        optim_btree.root = btree.root
        optim_btree.model = btree.model
        optim_btree.poly = btree.poly
        optim_btree.coor = btree.coor
        optim_btree.activation = btree.activation
        optim_btree.BN = btree.BN

        return optim_btree

    def construct_poly(self):
        assert self.poly is not None, f"The poly must be defined!"
        assert self.coor is not None, f"The coor must be defined!"
        assert self.deg is not None, f"The deg must be defined!"

        if self.poly != "none":
            self.distribute_task(NodeTask(polynomial, POLYFIT[self.poly], self.deg, self.coor, self.dtype),
                                 sequence="leaves_to_root", start_node_task=True)

    """
    Following code is included in the concatenate kernel since only in concatenate kernel need to reconstruct poly 
    in each step
    """
    # def reconstruct_poly(self):
    #     """
    #     Reconstruct the factor of leaves U_t by interpolating U_c with the given coordinates.
    #     The coordinates should have the same dimension as btree.
    #     :param coor:
    #     :return:
    #     """
    #
    #     def node_reconstruct_poly(node, poly_base, dtype):
    #         if node.is_leaf():
    #             coor = node.get_cache("index")
    #             U_c = node.get_cache("U_c")
    #             U_t = reconstruction_polynomial(U_c, poly_base, coor).astype(dtype)
    #             node.update_cache({"U_t": U_t})
    #
    #     assert self.poly is not None, f"The poly must be indicated by setting btree.coor"
    #     task = NodeTask(node_reconstruct_poly, poly_base=POLYVAL[self.poly], dtype=self.dtype)
    #     self.distribute_task(task, sequence="leaves_to_root", start_node_task=True)

    def init(self, A: np.ndarray = None):
        assert self.shape is not None, f"The shape must be defined!"
        assert self.k is not None, f"The k must be defined!"

        super(HT_FBR_Optim_Btree, self).init()
        self.construct_poly()

    def forward(self, feature):


        # 1. initial index in each node, initialisation defined in kernel
        self.init_index(feature)

        """ following line is included in the concatenate kernel"""
        # self.reconstruct_poly()

        # 2. forward defined in kernel
        task = NodeTask(self.node_forward,
                        feature=None,
                        activation=self.activation,
                        BN=self.BN,
                        LN=self.LN,
                        poly_base=POLYVAL[self.poly],
                        DO_rate=self.DO_rate,
                        dtype=self.dtype)
        A_predict = self.distribute_task(task, sequence="leaves_to_root", start_node_task=True)

        return A_predict

    def backward(self) -> None:
        # 1. backpropagated defined in kernel
        task = NodeTask(self.node_backward,
                        feature=None,
                        activation=self.activation,
                        BN=self.BN,
                        LN=self.LN,
                        DO_rate=self.DO_rate)
        self.distribute_task(task, sequence="root_to_leaves", start_node_task=True)

    def grad(self, feature: np.ndarray, h: float = 1e-6) -> np.ndarray:
        assert self.coor is not None, f"the interpolation of coordinates should not be None. To set, use btree.coor = "
        for i in range(self.dim):
            assert self.interval[i][0] < feature[0, i] < self.interval[i][1], \
                "the feature index should be in the interval of 'btree.coor'"

        gradient = np.zeros((self.dim, 1), dtype=self.dtype)
        for i in range(self.dim):
            feature_left = np.array(feature)
            feature_left[0, i] = feature_left[0, i] - h
            feature_right = np.array(feature)
            feature_right[0, i] = feature_right[0, i] + h
            gradient[i, 0] = (self.predict(feature_right) - self.predict(feature_left)) / (2 * h)
        return gradient

    def predict(self, feature):

        self.reconstruct_poly(feature.T)

        A_predict = self.distribute_task(
            NodeTask(self.node_forward, activation=self.activation, BN=self.BN, DO_rate=0, dtype=self.dtype),
            sequence="leaves_to_root",
            start_node_task=True)
        return A_predict

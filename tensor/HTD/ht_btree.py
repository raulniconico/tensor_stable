import pickle
import time
import numpy as np

from ..nuwa import Btree
from ..nuwa import Node
from ..nuwa import NodeTask

# polynomial fit function dictionary
POLYFIT = {"cheb": np.polynomial.chebyshev.chebfit,
           "herm": np.polynomial.hermite.hermfit,
           "lagu": np.polynomial.laguerre.lagfit,
           "none": None}

POLYVAL = {"cheb": np.polynomial.chebyshev.chebval,
           "herm": np.polynomial.hermite.hermval,
           "lagu": np.polynomial.laguerre.lagval}

class HT_Btree(Btree):
    def __init__(self, shape: tuple = None, poly: str = "none", device="cpu", model: str = None, dtype=np.float64):
        """
        HT_Btree contains many functions relate to HTD and HT-FBR. HT_Btree provides support for saving HT btree and
        development for them.

        To decomposite or reconstruct a HT_Btree, one needs to call HTD.decomposition.hsvd().

        :param shape: tuple, None by default, shape of the tensor to be decomposited.
        :param poly: str, None by default, the polynomial for interpolation. Check
        :param device: str, "cpu" by default, current version only supports CPU
        :param model: str, None by default, the algorithm of decomposition.
        :param dtype: np.dtype, np.float64 by default
        """

        assert poly in POLYFIT, f"the poly should be one of the following:{POLYFIT.keys()}"

        super().__init__()
        self._cache = {"shape": None, "poly": None, "tensor_shape": None, "device": "cpu", "model": None}
        self.shape = shape
        self.poly = poly
        self.device = device
        self.model = model
        self.dtype = dtype

    def init_mode(self) -> None:
        """
        Initial the node mode in each node. The node mode is initialized priori the left node.
        For example, if node mode of parent node is [0, 1, 2], then the left child node mode is [0, 1] and
        right node mode is therefore [2].

        :return: None
        """
        dim = len(self.shape)
        node_task = NodeTask(self.node_initial_mode, dim)
        self.distribute_task(node_task, start_node_task=True, sequence="root_to_leaves")

    @classmethod
    def empty(cls, shape: tuple = None):
        """
        Create a new empty btree with given shape.

        :param shape: tuple, the shape of the tensor
        :return:
        """
        btree = cls()
        btree.shape = shape
        dim = len(shape)
        node_task = NodeTask(cls.node_initial_mode, dim)
        btree.distribute_task(node_task, start_node_task=True, sequence="root_to_leaves")
        return btree

    def save(self, file: str = f'cache_{time.time()}') -> None:
        """
        Save binary tree into .btree file

        :param file: str, location of the file to save
        :return: None
        """
        file = file + ".btree"
        with open(file, 'wb+') as outfile:
            pickle.dump(self, outfile)

    @classmethod
    def load(cls, file: str):
        """
        Load btree from file.

        :param file: the position of saved tree.
        :return:
        """
        file = open(file, 'rb')
        btree = pickle.load(file)
        if (type(btree) == cls) or issubclass(type(btree), cls):
            return btree
        else:
            raise ValueError("The loaded file is not a valid btree")

    @staticmethod
    def node_initial_mode(node: Node, dim: int) -> None:
        """
        Initialize mode of node by dim of the tensor

        :param node: Node
        :param dim: int, dimension of the tensor
        :return: None
        """

        """ root"""
        # 0. initial root's param
        if node.is_root():
            mode = np.arange(dim)
            node.update_cache({"mode": mode})

        # 0.initial non-root's param
        else:
            mode = node.get_cache("mode")

        # 0.1 create child node
        length = int(mode.size)
        if length == 1:
            return
        else:
            if length % 2 != 0:
                node.left = Node()
                node.left.update_cache({"mode": mode[: length // 2 + 1]})

                node.right = Node()
                node.right.update_cache({"mode": mode[length // 2 + 1:]})
            else:
                node.left = Node()
                node.left.update_cache({"mode": mode[: length // 2]})

                node.right = Node()
                node.right.update_cache({"mode": mode[length // 2:]})
        return

    def get_singular_value_list(self) -> list:
        """
        Return a singular value list contains all the singular values in each level.
        The Nth element of the list is the ensemble of singular values in each layer of btree.

        :return: singular_value_list: list. List of size l same as the depth of btree. [] at 0th element.
        """

        def node_get_singular_value(node: Node, singular_value_list) -> list:
            if node.is_inner() or node.is_leaf():
                assert node.get_cache("epsilon") is not None, "error in getting singular value. No epsilon founded. " \
                                                              "Need to first initiate a HT_Btree and call hsvd()"
                level = self.get_depth(node)
                singular_value_list[level].append(node.get_cache('epsilon'))
            return singular_value_list

        singular_value_list = [[] for i in range(len(self.shape))]
        task = NodeTask(node_get_singular_value, singular_value_list)
        singular_value_list = self.distribute_task(task, start_node_task=True, sequence="root_to_leaves")
        return singular_value_list

    def get_singular_value_proportion(self) -> list:
        """
        Return a singular value cumulative proportion list.
        The Nth element of the list is the ensemble of singular values' proportion in each layer of btree.

        :return: singular_value_proportion_list: list
        """
        singular_value_list = self.get_singular_value_list()
        singular_value_proportion_list = [[] for i in range(len(self.shape))]
        for level in range(1, len(self.shape)):
            for singular in singular_value_list[level]:
                singular_value_proportion_list[level].append(np.cumsum(singular) / np.sum(singular))
        return singular_value_proportion_list

    def get_total_rank(self) -> int:
        """
        Count the total used node rank from a decomposed binary tree.

        :param btree: btree
        :return total_rank: int, total node rank
        """

        def node_count_rank(node: Node, total_rank) -> None:
            if node.is_root():
                return total_rank
            else:
                rank = node.get_cache("rank")
                node.parent.node_task.args[0] += rank

        total_rank = 0
        total_rank = self.distribute_task(NodeTask(node_count_rank, total_rank),
                                          sequence="leaves_to_root", start_node_task=True)
        return int(total_rank)

    def get_rank_list(self) -> list:
        """
        Return rank in each level
        The Nth element of the list is the ensemble of node ranks in each layer of btree.
        :return: rank_list: list
        """

        def node_get_rank(node: Node, singular_value_list) -> list:
            if node.is_inner() or node.is_leaf():
                level = self.get_depth(node)
                singular_value_list[level].append(node.get_cache("rank"))
            return singular_value_list

        rank_list = [[] for i in range(len(self.shape))]
        task = NodeTask(node_get_rank, rank_list)
        rank_list = self.distribute_task(task, start_node_task=True, sequence="root_to_leaves")
        return rank_list

    def get_total_params(self) -> int:
        """
        Get the total number of parameters in the btree.
        Count all the size of B_t and U_t

        :return: total_params: int, total params in the binary tree
        """

        def node_count_params(node: Node, total_params) -> None:
            ######### root
            if node.is_root():
                total_params += node.get_cache("params")
                return total_params

            ######### inner
            if node.is_inner():
                node.parent.node_task.args[0] += node.get_cache("params")

            ######### leaf
            if node.is_leaf():
                node.parent.node_task.args[0] += node.get_cache("params")

        total_params = 0
        total_params = self.distribute_task(NodeTask(node_count_params, total_params),
                                            sequence="leaves_to_root", start_node_task=True)
        return int(total_params)

    def print_total_rank(self) -> None:
        """
        Count the total used node rank from a decomposed binary tree.

        :param btree: btree
        :return total_rank: int, total node rank
        """

        def node_count_rank(node: Node, total_rank) -> None:
            if node.is_root():
                return total_rank
            else:
                print(node.get_cache("mode"), ":", node.get_cache("rank"))
                rank = node.get_cache("rank")
                node.parent.node_task.args[0] += rank

        print("####### rank list #######")
        total_rank = 0
        total_rank = self.distribute_task(NodeTask(node_count_rank, total_rank),
                                          sequence="leaves_to_root", start_node_task=True)
        print("total rank:", total_rank)

    def print_total_params(self) -> None:
        """
        Get the total number of parameters in the btree.
        Count all the size of B_t and U_t

        :return: total_params: int, total params in the binary tree
        """

        def node_count_params(node: Node, total_params) -> None:
            print(node.get_cache("mode"), ":", node.get_cache("params"))
            ######### root
            if node.is_root():
                total_params += node.get_cache("params")
                return total_params

            ######### inner
            if node.is_inner():
                node.parent.node_task.args[0] += node.get_cache("params")

            ######### leaf
            if node.is_leaf():
                node.parent.node_task.args[0] += node.get_cache("params")

        print("####### param list #######")
        total_params = 0
        total_params = self.distribute_task(NodeTask(node_count_params, total_params),
                                            sequence="leaves_to_root", start_node_task=True)
        print("total params:", total_params)

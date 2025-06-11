from typing import Any

from .task import Task, NodeTask


class Node:
    CACHE_KEY = ["mode", "index", "shape", "A_t", "U_t", "coor", "U_c", "rank", "epsilon", "B_t", "params", "B_index", "B_value"]
    CACHE_OPTIM = ["grad_0", "B_t_grad",  "U_t_BN", "LN", "U_t_s", "kron_prod"]

    def __init__(self, parent=None, left=None, right=None, task: Task = None, node_task: NodeTask = None):

        """
        Node in binary tree has maximum 2 children: left and right, it can have only one parent. A node can be
        distributed a task or node_task.

        :param parent: Node, None by default, parent node, None if it's a root node
        :param left: Node, None by default, left child
        :param right: Node, None by default, right child
        :param task: Task, None by default, task to run
        :param node_task: None by default, NodeTask, node task to run
        :local _cache: dict, None by default, saved cache.
                        It stocks all the necessary cache during HT and reconstruction.
        """

        self.parent = parent
        self.left = left
        self.right = right
        self.task = task
        self.node_task = node_task

        self._cache = {}

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left):
        self._left = left
        if left is not None:
            left.parent = self

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, right):
        self._right = right
        if right is not None:
            right.parent = self

    @property
    def task(self) -> Task or None:
        return self._task

    @task.setter
    def task(self, task):
        self._task = task

    @property
    def node_task(self) -> NodeTask or None:
        return self._node_task

    @node_task.setter
    def node_task(self, node_task):
        self._node_task = node_task

    def update_cache(self, _cache: dict) -> None:
        """
        Update node cache. The cache to stock must in dict.
        :param _cache: dict
        :return: None
        """
        self._cache.update(_cache)

    def get_cache(self, key: str = None) -> Any:
        """
        There are several keywords exist, as
        mode: the mode of the node
        A_t: temporary saved reshaped tensor A with respect to mode. It will be deleted during HT.
        U_t: The left Singular vector of A_t. Stocked
        rank: rank of U_t
        epsilon: Singular value of A_t.
        U_c: Fitted polynomial of U_t used in HT-FBR
        U_t_ori: The copy of U_t in HT-FBR. The U_t will be deleted while U_c is generated
        U_t_rec: Reconstructed U_t from U_c by interpolation
        B_t: Transfer tensor or matrix
        shape: The shape of tensor

        :param key: str, None by default, the key of the cache in _cache dictionary
        :return: Any. The value with respect of the key
        """
        if key is not None:
            assert key in self.CACHE_KEY + self.CACHE_OPTIM, \
                "Key not available. Check node.CACHE_KEY and node.CACHE_OPTIM"
            if key in self._cache.keys():
                return self._cache[key]
            else:
                raise KeyError
        else:
            return self._cache

    def clean_cache(self, key=None):
        """
        Clear cache in the node. If no key given, clear all caches.

        :param key: Any, key of cache dictionary
        :return: None
        """
        if key is not None:
            self._cache.pop(key)
        else:
            self._cache = {}

    def is_inner(self) -> bool:
        """
        Return true if node has child and its parent is not None

        :return: bool
        """
        if (self._left is not None or self._right is not None) \
                and self._parent is not None:
            return True
        else:
            return False

    def is_leaf(self) -> bool:
        """
        Return True if node is a leaf(no child node)
        Else return False

        :return: bool
        """
        if self._left is None and self._right is None:
            return True
        else:
            return False

    def is_root(self) -> bool:
        """
        Return true if node has no parent

        :return: bool
        """
        if self._parent is None:
            return True
        else:
            return False

    def start_task(self) -> Any:
        """
        When node starts a node_task, it always parses itself to task.
        Task receives it and can do some node operations. Use start_task
        if no need.

        :return: Any, returns of node_task
        :return:
        """
        return self.task.start()

    def start_node_task(self) -> Any:
        """
        When node starts a node_task, it always parses itself to task.
        Task receives it and can do some node operations. Use start_task
        if no need.

        :return: Any, returns of node_task
        """
        return self.node_task.start(self)

    def get_result(self) -> Any:
        return self._task().get_result()
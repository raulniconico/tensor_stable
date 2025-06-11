from typing import Any, Union, List
import numpy as np

from .node import Node
from .task import Task, NodeTask


class Btree:
    def __init__(self, root=None):
        """
        Binary tree. A binary tree can join another tree. It will
        create an empty root node if no root is parsed when creating.
        A binary tree has leaves and inner nodes.

        Each node wide function contains a recursion function starts with node.
        Recursion functions will func on every node.

        :param root: Node, root node of binary tree. An empty one is
        created if no root parsed.
        """

        self.root = root

    @property
    def root(self) -> Node:
        return self._root

    @root.setter
    def root(self, root=None) -> None:
        if root is not None:
            self._root = root
        else:
            self._root = Node()

    def add_node(self, node: Node, parent: Node) -> None:
        """
        Add a node to binary tree.

        :param node: Node, the node needs to be added into binary tree
        :param parent: Node, parent node of the node.
        :return: None
        """
        assert self.root is not None, "root is none. use set_root to add root"
        assert (parent.left is None) or (parent.right is None), "parent node has already two children"
        if parent.left is None:
            parent.left = node
        else:
            parent.right = node
        node._parent = parent

    def distribute_task(self, task: Union[Task, NodeTask] = None, start_task: bool = False, start_node_task=False,
                        sequence: str = "leaves_to_root") -> Any:
        """
        Through whole binary tree, meanwhile task can be allocated to each node,
        The task can run in two ways, 1) once the node is distributed a task, it
        If start is ture,the task will be done begin with leaves till the root, or it just
        distributes the task

        :param task: [Task, NodeTask], task or node task allocate to each node
        :param start_task: bool, start the task des que possible
        :param start_node_task: bool, start the node task des que possible
        :param sequence: sting
        :return: Any, returns of recursion. None if not start
        """

        assert sequence in ["leaves_to_root",
                            "root_to_leaves"], "sequence should be in ['leaves_to_root', 'root_to_leaves']"

        def node_distribute_task(node: Node) -> Any:
            # distribute task
            if task is not None:
                if type(task) == Task:
                    node.task = task
                elif type(task) == NodeTask:
                    node.node_task = task

            result = None
            # if root_to_leaves, function will do immediately
            if sequence == "root_to_leaves":
                if start_task:
                    result = node.start_task()

                if start_node_task:
                    result = node.start_node_task()

            # recursion to left child and right child
            if node.left is not None:
                node_distribute_task(node.left)

            if node.right is not None:
                node_distribute_task(node.right)

            # start the task from bottom to top
            if sequence == "leaves_to_root":
                if start_task:
                    result = node.start_task()

                if start_node_task:
                    result = node.start_node_task()

            return result

        result = node_distribute_task(self.root)
        return result

    def start_node_task(self, sequence: str = "leaves_to_root") -> Any:
        """
        Start node's node task.

        :param sequence: str.
        :return: result: Any, result of the nodetask
        """
        result = self.distribute_task(start_node_task=True, sequence=sequence)
        return result

    def start_task(self, sequence: str = "leaves_to_root"):
        """
        Start node's task.

        :param sequence: Str.
        :return: result: Any, result of the nodetask
        """
        result = self.distribute_task(start_task=True, sequence=sequence)
        return result

    def get_root(self) -> Node:
        """
        Get node's root node

        :return root:Node, :
        """
        return self.root

    def get_leaves(self) -> Union[None, List[Node]]:
        """
        Return leaves of binary tree.

        :return: Leaves_list: list, list contains all leaves
        """

        def node_get_leaves(node: Node, leaves_list: Union[list, np.array]):
            """
            :param node: Node
            :param leaves_list: list
            :return: None
            """
            if node.is_root():
                return leaves_list
            elif node.is_leaf():
                node.parent.node_task.args[0].append(node)

        task = NodeTask(node_get_leaves, [])
        leaves_list = self.distribute_task(task, start_node_task=True)
        return leaves_list

    def get_inners(self) -> Union[None, List[Node]]:
        """
        Return all inner nodes (not root, nor leaf) list

        :return: inners_list, list
        """

        def node_get_inners(node: Node, inners_list: Union[list, np.array]):
            if node.is_root():
                return inners_list
            elif node.is_inner():
                node.parent.node_task.args[0].append(node)

        task = NodeTask(node_get_inners, [])
        inners_list = self.distribute_task(task, start_node_task=True)
        return inners_list

    def get_non_leaf(self) -> Union[None, List[Node]]:
        """
        get all non-leaf nodes: root node + inner nodes
        """

        def node_get_non_leaf(node: Node, non_leaf_list: Union[list, np.array]):
            if node.is_root():
                node.node_task.args[0].append(node)
                return non_leaf_list
            elif node.is_inner():
                node.parent.node_task.args[0].append(node)

        task = NodeTask(node_get_non_leaf, [])
        non_leaf_list = self.distribute_task(task, start_node_task=True)
        return non_leaf_list

    def get_all_nodes(self) -> Union[None, Node, List[Node]]:
        """
        Return all the nodes in the binary tree

        :return: all_list, list
        """

        def node_get_all_nodes(node: Node, all_list: [list, np.array]):
            if node.is_root():
                all_list.append(node)
                return all_list
            node.parent.node_task.args[0].append(node)
            return

        task = NodeTask(node_get_all_nodes, [])
        all_list = self.distribute_task(task, start_node_task=True)
        return all_list

    def get_depth(self, node: Node = None) -> int:
        """
        If no node is given, then returns the depth of the binary tree.
        If a node is given, returns the depth of the node

        :param node: Node, None as default
        :return: int, depth of the node or binary tree
        """
        if node is None:
            return int(np.ceil(np.log2(len(self.get_all_nodes()))))
        else:
            depth = 0
            while not node.is_root():
                depth += 1
                node = node.parent
            return depth

    def get_nodes_at_level(self, l: int) -> list:
        """
        Return all the nodes at the level l. Root level is set to level 0

        :param l: level, int, target level
        :return: node_list, list
        """

        def node_get_l(node: Node, node_list: Union[list, np.array]):
            if node.parent is None:
                node.update_cache({"level": 0})
            else:
                node.update_cache({"level": node.parent.get_cache("level") + 1})
            if node.get_cache("level") == l:
                node_list.append(node)
            return node_list

        task = NodeTask(node_get_l, [])
        node_list = self.distribute_task(task, start_node_task=True, sequence="root_to_leaves")
        return node_list

    def clean_cache(self, key: Any = None) -> None:
        """
        Clean cache in every node. If no key is given, clean all cache.

        :param key: Any, None by default.
        """

        def node_clean_cache(node: Node) -> None:
            node.clean_cache(key)

        task = NodeTask(node_clean_cache, key)
        self.distribute_task(task, start_node_task=True, sequence="root_to_leaves")

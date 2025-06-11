import copy
from typing import Any


class Task:
    def __init__(self, func=None, *args, **kwargs):
        """
        Task is parsed into node. When a node begins a task, Task will run func with
        args and kwargs.

        !!! Note: func, args, kwargs and results are shared among the tasks

        :param func: python func, which will run in each node in tree
        :param args: tuple, args parse to the func when func runs
        :param kwargs: dict, keyword args parse to the func when func runs
        :param result: Any, the result of the func after running
        """

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None

    @property
    def args(self) -> Any:
        """
        Get args in the Task.
        Used as data transport between Tasks.

        :return: args: Any
        """
        return self._args

    @args.setter
    def args(self, *args, **kwargs) -> None:
        self._args = list(*args)
        self._kwargs = kwargs

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs: dict):
        self._kwargs = kwargs

    @property
    def result(self) -> Any:
        return self._result

    @result.setter
    def result(self, result: Any) -> None:
        self._result = result

    @classmethod
    def _copy_from_self(cls, func, *args, **kwargs):
        task = cls(func, *args, **kwargs)
        return task

    def copy(self):
        task = self._copy_from_self(copy.deepcopy(self.func), copy.deepcopy(*self.args), copy.deepcopy(**self.kwargs))
        return task

    def start(self, node=None) -> Any:
        """
        When start is called, it will start the func and parse args and
        kwargs into func.

        :param node: None, no need to fill in
        :return: Any, returns of func
        """
        self._result = self.func(*self._args, **self._kwargs)
        return self._result


class NodeTask(Task):
    def __init__(self, func=None, *args, **kwargs):
        """
        NodeTask is a son class of Task class. Function in NodeTask is allowed to
        do graph operations like add nodes, generate son nodes etc. While Task class
        only allows to do simple function without any Node operation.

        In principle functions' return in NodeTask has the following options:
        0. Do nothing and return
        1. Pass only to left child
        2. Pass to left and right children


        :param func: python function, will be run in each node. The first param should be Node type
        :param args: tuple, args parse to the func when func runs
        :param kwargs: dict, keyword args parse to the func when func runs
        :param result: Any, the result of the func after running
        """
        super().__init__(func, *args, **kwargs)

    def start(self, node=None) -> Any:
        """
        Rewrite start() method in Task class. Node class is included as

        :param node: Node
        :return:
        """

        assert node is not None, "node in NodeTask must not be None"
        self._result = self.func(node, *self.args, **self.kwargs)
        return self.result
from typing import Union, List

from .ht_btree import HT_Btree, POLYFIT, POLYVAL
from ..nuwa import Node, NodeTask
from ..nuwa.util import construction_polynomial, reconstruction_polynomial, svd, m_m_m, svd_random, t_m_m

import time
import numpy as np



# node method
def polynomial(node: Node,
               poly_fit,
               deg: int = 3,
               coor: Union[np.ndarray, list] = None,
               dtype: type = np.float64):
    """
    Calculate polynomials of U_t in the target node with given coordinates or no coordinate by using numpy poly_fit.

    :param node: Node
    :param poly_fit: function, np.Polynomial's fit function
    :param deg: int, degree of the polynomial
    :param coor: coordinates for an interpolation.
    :param dtype: type, dtype of data

    :return: None
    """

    ############### leaves and inner
    if node.is_leaf():
        U_t = node.get_cache('U_t')
        # update d: dimension 1 of U_t for polynomial reconstruction

        # polynomial interpolation of U_t
        if coor is not None:
            mode = node.get_cache("mode")
            coor_node = coor[mode[0]].astype(dtype)
        else:
            coor_node = np.linspace(0, U_t.shape[0] - 1, U_t.shape[0]).astype(dtype)
        node.update_cache({"coor": coor_node})

        U_c = construction_polynomial(U_t, poly_fit, deg, coor_node)

        node.update_cache({"U_c": U_c.astype(dtype)})
        node.update_cache({"params": U_c.size})
        # node.clean_cache("U_t")


def generate_A_t(node: Node, A: Union[np.ndarray], dtype: type) -> None:
    """
    Generate A_t for each node, each node's mode must be defined first.
    A_t: the matricization of tensor A.

    :param node: Node
    :param A: np.ndarray
    :param dtype: type, numpy dtype
    :return: None
    """

    assert node.get_cache("mode") is not None, "Could not find 'mode' param in node's cache."

    ############### root
    if node.is_root():
        node.update_cache({"A_t": A.astype(dtype)})

    ############### inner and leaf
    else:
        # Define I_t and I_t_p
        dim = len(A.shape)
        I_t = node.get_cache("mode")
        I_t_p = [i for i in list(range(dim)) if i not in I_t]
        shape0 = np.prod(np.array(A.shape)[I_t])
        shape1 = np.prod(np.array(A.shape)[I_t_p])

        for i in range(len(I_t)):
            A = np.moveaxis(A, I_t[-1], 0)

        A_t = A.reshape((shape0, shape1))
        # # cache A_t in node's cache
        node.update_cache({"A_t": A_t.astype(dtype)})


def SVD_nodewise(node: Node, k: tuple, dtype, random_SVD: bool, random_p: int, full_SVD: bool) -> None:
    """
    Calculate SVD of A_t on each node and save top k columns
    of U_t: the left singular vector of A_t.

    CLear A_t and U_t inside inner

    :param node: Node
    :param k: tuple, node rank of each node
    :param dtype: type, dtype while calculation
    :param random_SVD: bool, if use randomised SVD instead of numpy direct SVD method
    :param random_p: int, randomised SVD parameter. It's the size of the random factor matrix for computing randomised
            SVD
    :param full_SVD: bool, numpy SVD parameter. If SVD a full matrix. See numpy.linalg.svd documentation for details

    :return: None
    """
    ############### leaves and inner
    if node.is_leaf() or node.is_inner():
        assert node.get_cache("mode") is not None, "Could not find 'mode' param in node's cache."
        rank = k[-len(node.get_cache("mode"))]

        start = time.time()
        if random_SVD:
            U_t, S, VH = svd_random(node.get_cache("A_t"), rank, random_p, full_matrices=full_SVD)
        else:
            U_t, S, VH = svd(node.get_cache("A_t"), full_matrices=full_SVD)
            U_t = U_t[:, :rank].astype(dtype)
        finish = time.time()
        print("SVD: ", node.get_cache('mode'), finish - start)

        node.update_cache({"epsilon": S.astype(dtype)})

        ############### leaves
        # calculate U_t on every leaf, save U_t, Clean A_t
        if node.is_leaf():
            # update U_t, rank and singular value
            node.update_cache({"U_t": U_t.astype(dtype)})
            node.update_cache({"rank": U_t.shape[1]})
            node.update_cache({"params": U_t.size})
            # clean A_t
            node.clean_cache("A_t")

        ############### inner
        # calculate B_t and U_t in inner node (not root nor leaves), save B_t and U_t
        if node.is_inner():
            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")

            # update U_t, B_t, epsilon
            node.update_cache({"U_t": U_t.astype(dtype)})

            # calculate B_t
            start = time.time()
            B_t = m_m_m(U_t, U_l, U_r).astype(dtype)
            node.update_cache({"B_t": B_t.astype(dtype)})
            finish = time.time()
            print("m_m_m: ", node.get_cache('mode'), finish - start)

            node.clean_cache("A_t")
            # update rank
            node.update_cache({"rank": U_t.shape[1]})
            node.update_cache({"params": B_t.size})

    ############### root

    # root calculate B_t
    if node.is_root():
        A = node.get_cache("A_t")
        U_l = node.left.get_cache("U_t")
        U_r = node.right.get_cache("U_t")

        start = time.time()
        # # single entry
        # mode = np.linspace(0, A.size - 1, A.size, dtype=int)
        # B_t = np.zeros((1, U_l.shape[1], U_r.shape[1]), dtype=dtype)
        # B_t = (m_m_m_single_entry(mode, A.reshape((-1, 1)), U_l, U_r, 1, B_t)).astype(dtype)

        B_t = m_m_m(A.reshape((-1, 1)), U_l, U_r).astype(dtype)
        node.update_cache({"B_t": B_t.astype(dtype)})

        finish = time.time()
        print("m_m_m root: ", node.get_cache('mode'), finish - start)
        # clean A_t
        node.clean_cache("A_t")
        node.update_cache({"params": node.get_cache("B_t").size})

    # from leaves to root clean U_t in the interior node
    if node.left is not None and node.left.is_inner():
        node.left.clean_cache("U_t")
    if node.right is not None and node.right.is_inner():
        node.right.clean_cache("U_t")


def hsvd(A: np.ndarray,
         k: tuple,
         poly: str = "none",
         deg: int = 3,
         coor: Union[np.ndarray, list] = None,
         dtype: type = np.float64,
         random_SVD: bool = True,
         random_p: int = 10,
         full_SVD: bool = False) -> HT_Btree:
    """
    Hierarchical SVD of tensor. A the input tensor, several versions can
    be applied to HSVD. If no poly is chosen, then HTD vanilla version is
    then been used. If poly is given, then it interpolates each column
    of U_t in the leaf nodes.

    Therefore, the whole process can be expressed as:
    0. check input params
    1. create a binary tree via tensor A, distribute mode cluster t
    2. calculate and distribute A_t to every node
    3. calculate svd of A_t on each node and save mode frame U_t and
        transfer matrix or transfer tensor.
    4. set poly flag in root node for reconstruction
    4.1 (optional) polynomial interpolate U_t in the leaves, or all the node
        depends on the method.

    functions:
        util (from LN.py):
            m_m_m: Calculate B_t given U_t and (U_l, U_r)
            gradient_descent: Gradient descent for finding transfer matrix
            svd: SVD method
            B_M_M_single_entry:
            construction_polynomial:

        node method:
            SVD_nodewise:
            generate_A_t:
            polynomial: Calculate polynomials of U_t in target node

    :param A: np.ndarray, tensor to decomposition
    :param k: tuple, rank on each node
    :param poly: str, set polynomial, "none" by default, choice: ["none", "cheb", "herm"]
    :param deg: int, 3 by default, degree of polynomial
    :param coor: Union[np.ndarray, list], None by default.
    :param dtype: dtype, np.float64 by default. numpy data type (dtype). Data type using during the whole computation
    :param random_SVD: bool, if use randomised SVD instead of numpy direct SVD method
    :param random_p: int, randomised SVD parameter. It's the size of the random factor matrix for computing randomised
            SVD
    :param full_SVD: bool, numpy SVD parameter. If SVD a full matrix. See numpy.linalg.svd documentation for details

    :return: btree: btree
    """

    # node method

    ############### main task
    # 0. check input params
    assert poly in POLYFIT.keys(), "parameter poly must in " + str(POLYFIT.keys())

    # 1. create a binary tree
    btree = HT_Btree.empty(A.shape)

    # 2. distribute A_t to every node
    btree.distribute_task(NodeTask(generate_A_t, A=A, dtype=dtype),
                          sequence="root_to_leaves", start_node_task=True)

    # 3. calculate svd on each node
    btree.distribute_task(
        NodeTask(SVD_nodewise, k=k, dtype=dtype, random_SVD=random_SVD, random_p=random_p, full_SVD=full_SVD),
        sequence="leaves_to_root", start_node_task=True)

    # 4. set flags in a binary tree
    btree.poly = poly
    btree.algo = "HSVD"
    btree.shape = A.shape

    # 4.1 chebyshev polynomial all nodes
    if poly != "none":
        btree.distribute_task(NodeTask(polynomial, POLYFIT[poly], deg, coor, dtype),
                              sequence="leaves_to_root", start_node_task=True)
    return btree


# node method
def node_reconstruction_polynomial(btree, poly: str = None, coor: np.ndarray = None) -> None:
    """
    Reconstruction

    :param btree:
    :param poly:
    :param coor:
    :return:
    """
    def node_reconstruction_polynomial_on_leaf(node: Node,
                                               poly_base,
                                               coor: Union[np.ndarray, list] = None,
                                               dtype=np.float64) -> None:
        """


        Node method, reconstruction U_t from U_c
        :param node: Node
        :param poly_base: list of int, polynomial base
        :param coor: np.ndarray,
        :param dtype: np.dtype
        :return: None
        """
        if node.is_leaf():
            # customized coordinates
            if coor is not None:
                mode = node.get_cache("mode")
                coor_node = coor[mode[0]]
            else:
                coor_node = node.get_cache("coor")

            # reconstruction
            U_c = node.get_cache("U_c")
            U_t = reconstruction_polynomial(U_c, poly_base, coor_node).astype(dtype)
            node.update_cache({"U_t": U_t})

    btree.distribute_task(
        NodeTask(node_reconstruction_polynomial_on_leaf,
                 POLYVAL[poly],
                 coor),
        sequence="leaves_to_root", start_node_task=True)


def reconstruction_from_tree(btree: HT_Btree,
                             coor: Union[np.ndarray, list] = None,
                             dtype: np.dtype = np.float64,
                             feature: Union[np.ndarray, list, List[int]] = None) -> np.ndarray:
    """
    Reconstruct tensor from factors and core tensor. Input a btree
    returns a vector of result

    :param btree: btree, a btree already HSVDed
    :param coor: interpolation coordinates
    :param dtype: type, numpy dtype

    :return: A_h: np.ndarray, vectorised reconstructed tensor
    """

    # node method
    def node_reconstruction(node: Node, feature, dtype):
        """
        HTD, using U_t and B_t directly

        :param node:

        :return: A_h, np.ndarray
        """

        ############### inner
        if node.is_inner():
            # reconstruction U_t from B_t, U_l and U_r
            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")

            B_t = node.get_cache("B_t")

            start = time.time()
            U_t = np.zeros((U_l.shape[0] * U_r.shape[0], B_t.shape[0]), dtype=dtype)
            feature = np.linspace(0,
                                U_l.shape[0] * U_r.shape[0] * B_t.shape[0] - 1,
                                U_l.shape[0] * U_r.shape[0] * B_t.shape[0], dtype=int)
            U_t = t_m_m(feature, B_t, U_l, U_r, U_t)

            finish = time.time()
            print("tmm rec: ", node.get_cache('mode'), np.round(finish - start, 2), "s")
            node.update_cache({"U_t": U_t.astype(dtype)})

        ############### root
        if node.is_root():
            # reconstruction U_t from B_t, U_l and U_r
            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")

            B_t = node.get_cache("B_t")

            if feature is not None:
                start = time.time()
                values = np.zeros((feature.size, 1), dtype=dtype)
                A_h = t_m_m(feature, B_t, U_l, U_r, values).reshape(-1, 1).astype(dtype)
                finish = time.time()

                print("tmm single entry: ", node.get_cache('mode'), np.round(finish - start, 2), "s")
                node.update_cache({"A_h": A_h})

            else:
                start = time.time()
                feature = np.linspace(0, U_l.shape[0] * U_r.shape[0] - 1, U_l.shape[0] * U_r.shape[0], dtype=int)
                values = np.zeros((U_l.shape[0] * U_r.shape[0], 1), dtype=dtype)
                A_h = t_m_m(feature, B_t, U_l, U_r, values).reshape((-1, 1)).astype(dtype)
                finish = time.time()
                print("tmm cut rec: ", node.get_cache('mode'), np.round(finish - start, 2), "s")

                node.update_cache({"A_h": A_h})
            return A_h

    ############### main task
    # 1. reconstruction A_c: chebyshev
    poly = btree.poly
    if poly != "none":
        # Reconstruction with (U_c, B_t)
        node_reconstruction_polynomial(btree, poly, coor)

    dtype = btree.dtype

    # 1. reconstruction of A_h
    A_h = btree.distribute_task(NodeTask(node_reconstruction, feature=feature, dtype=dtype),
                                sequence="leaves_to_root", start_node_task=True)

    return A_h

# def hsvd_self_adaptive(A: np.ndarray,
#                        tol: Union[np.float64, np.float32, float] = 1e-6,
#                        mesh_size: tuple = (20, 20),
#                        sample_point: int = 20,
#                        full_SVD: bool = True,
#                        poly: str = "none",
#                        deg: int = 10,
#                        coor: Union[np.ndarray, list] = None,
#                        dtype: type = np.float64):
#     # node method
#     def generate_A_t(node: Node, A: Union[np.ndarray], dtype: type = dtype) -> None:
#         """
#         Generate A_t for each node, each node's mode must be defined first.
#         A_t: the matricization of tensor A.
#
#         :param node: Node
#         :param A: np.ndarray
#         :param dtype: type, numpy dtype
#         :return: None
#         """
#
#         assert node.get_cache("mode") is not None, "Could not find 'mode' param in node's cache."
#
#         ############### root
#         if node.is_root():
#             node.update_cache({"A_t": A.reshape((-1, 1)).astype(dtype)})
#
#         ############### inner and leaf
#         else:
#             # Define I_t and I_t_p
#             dim = len(A.shape)
#             I_t = node.get_cache("mode")
#             I_t_p = [i for i in list(range(dim)) if i not in I_t]
#             shape0 = np.prod(np.array(A.shape)[I_t])
#             shape1 = np.prod(np.array(A.shape)[I_t_p])
#
#             for i in range(len(I_t)):
#                 A = np.moveaxis(A, I_t[-1], 0)
#
#             A_t = A.reshape((shape0, shape1))
#             # cache A_t in node's cache
#             node.update_cache({"A_t": A_t.astype(dtype)})
#
#     def SVD_self_adaptive(node: Node, tol, mesh_size: tuple = (10, 10), sample_point: int = 100, full_SVD=True,
#                           dtype: type = dtype):
#
#         ################## root
#         if node.is_root():
#             print("##### root node:", node.get_cache("mode"), "#####")
#             # Calculation of A_t, U_l, U_r
#             A = node.get_cache("A_t")
#             left = node.left
#             right = node.right
#
#             start = time.time()
#             U_l, S, VH = svd(left.get_cache("A_t"), full_matrices=full_SVD)
#             finish = time.time()
#             print("SVD", left.get_cache('mode'), np.round(finish - start, 2), "s")
#
#             start = time.time()
#             U_r, S, VH = svd(right.get_cache("A_t"), full_matrices=full_SVD)
#             finish = time.time()
#             print("SVD", right.get_cache('mode'), np.round(finish - start, 2), "s")
#
#             # Calculation of sparse B_t
#             start = time.time()
#             B_index, B_value, U_l_index, U_r_index = (
#                 m_m_m_self_adaptive(A.reshape((U_l.shape[0] * U_r.shape[0], 1)), U_l=U_l, U_r=U_r,
#                                     tol=tol, mesh_size=mesh_size, sample_point=sample_point, dtype=dtype))
#             finish = time.time()
#             print("AMM adaptive", np.round(finish - start, 2), "s")
#
#             left.update_cache({"U_t": U_l[:, U_l_index], "rank": U_l[:, U_l_index].shape[1]})
#             right.update_cache({"U_t": U_r[:, U_r_index], "rank": U_r[:, U_r_index].shape[1]})
#             node.update_cache({"B_index": B_index, "B_value": B_value})
#
#             params = 0
#             for i, B_mesh in enumerate(B_value):
#                 params += B_mesh.size
#             node.update_cache({"params": params})
#
#
#         ################## inner
#         elif node.is_inner():
#             print("####### node:", node.get_cache("mode"), "#######")
#
#             # get U_t, U_l and U_r
#             left = node.left
#             right = node.right
#             U_t = node.get_cache("U_t")
#
#             start = time.time()
#             U_l, S, VH = svd(left.get_cache("A_t"), full_matrices=full_SVD)
#             finish = time.time()
#             print("SVD", left.get_cache('mode'), np.round(finish - start, 2), "s")
#
#             start = time.time()
#             U_r, S, VH = svd(right.get_cache("A_t"), full_matrices=full_SVD)
#             finish = time.time()
#             print("SVD", right.get_cache('mode'), np.round(finish - start, 2), "s")
#
#             # calculate B_t
#             # hybrid, if B_t size bigger than 1e6, then AMM with B sparse, else, m_m_m
#             if U_t.shape[1] * U_l.shape[1] * U_r.shape[1] < 1e16:
#
#                 start = time.time()
#                 node.update_cache({"B_t": m_m_m_cut(U_t, U_l, U_r).astype(dtype)})
#                 finish = time.time()
#                 print("m_m_m: ", node.get_cache('mode'), np.round(finish - start, 2), "s")
#
#                 # Stock U_l and U_r in the child node
#                 left.update_cache({"U_t": U_l, "rank": U_l.shape[1]})
#                 right.update_cache({"U_t": U_r, "rank": U_r.shape[1]})
#                 node.update_cache({"params": node.get_cache("B_t").size})
#
#             else:
#                 start = time.time()
#                 B_index, B_value, U_l_index, U_r_index = m_m_m_self_adaptive(U_t, U_l, U_r, tol=1e-2,
#                                                                              sample_point=30)
#                 node.update_cache({"B_index": B_index, "B_value": B_value})
#                 finish = time.time()
#                 print("A_M_M_B_sparse: ", node.get_cache('mode'), np.round(finish - start, 2), "s")
#
#                 # Stock U_l and U_r in the child node
#                 left.update_cache({"U_t": U_l[:, U_l_index], "rank": U_l[:, U_l_index].shape[1]})
#                 right.update_cache({"U_t": U_r[:, U_r_index], "rank": U_r[:, U_r_index].shape[1]})
#
#                 params = 0
#                 for i, B_mesh in enumerate(B_value):
#                     params += B_mesh.size
#                 node.update_cache({"params": params})
#
#             # clean U_t
#             node.clean_cache("U_t")
#
#         ################## leaf
#         elif node.is_leaf():
#             node.update_cache({"params": node.get_cache("U_t").size})
#
#         # clean A_t in root
#         node.clean_cache("A_t")
#
#     # 1. create a binary tree
#     btree = HT_Btree.from_tensor(A)
#
#     # 2. distribute A_t to every node
#     btree.distribute_task(NodeTask(generate_A_t, A, dtype),
#                           sequence="root_to_leaves", start_node_task=True)
#
#     # 3. calculate svd on each node
#     btree.distribute_task(NodeTask(SVD_self_adaptive,
#                                    tol=tol,
#                                    mesh_size=mesh_size,
#                                    sample_point=sample_point,
#                                    full_SVD=full_SVD,
#                                    dtype=dtype),
#                           sequence="root_to_leaves", start_node_task=True)
#
#     # 4. set poly flag in root node for reconstruction
#     btree.update_cache({"poly": poly, "algo": "HSVD", "shape": A.shape})
#
#     # 4.1  polynomial all nodes
#     if poly != "none":
#         btree.distribute_task(NodeTask(polynomial, POLYFIT[poly], deg, coor, dtype),
#                               sequence="leaves_to_root", start_node_task=True)
#     return btree
#
#
# def reconstruction_self_adaptive(btree: HT_Btree,
#                                  coor: Union[np.ndarray, list] = None,
#                                  dtype: np.dtype = np.float64,
#                                  checkpoint: bool = False) -> np.ndarray:
#     POLYVAL = {"cheb": np.polynomial.chebyshev.chebval,
#                "herm": np.polynomial.hermite.hermval,
#                "lagu": np.polynomial.laguerre.lagval}
#
#     # node method
#     # @save_checkpoint(checkpoint=checkpoint, btree=btree)
#     def node_reconstruction(node: Node):
#         """
#         HTD, using U_t and B_t directly
#
#         :param node:
#
#
#         :return: A_h, np.ndarray
#         """
#
#         ############### inner
#         if node.is_inner():
#             # reconstruction U_t from B_t, U_l and U_r
#             U_l = node.left.get_cache("U_t")
#             U_r = node.right.get_cache("U_t")
#             U_t = np.zeros((U_l.shape[0] * U_r.shape[0], 1), dtype=dtype)
#
#             if node.get_cache("B_t") is not None:
#                 B_t = node.get_cache("B_t")
#
#                 start = time.time()
#                 U_t = t_m_m_cut(B_t, U_l, U_r)
#                 finish = time.time()
#                 print("tmm cut: ", node.get_cache('mode'), np.round(finish - start, 2), "s")
#
#             elif node.get_cache("B_index") is not None and node.get_cache("B_value") is not None:
#                 # B_t = node.get_cache("B_t")
#                 B_index = node.get_cache("B_index")
#                 B_value = node.get_cache("B_value")
#
#                 start = time.time()
#                 U_t = t_m_m_self_adaptive(B_index, B_value, U_l, U_r).astype(dtype)
#                 finish = time.time()
#
#                 print("tmm adapt: ", node.get_cache('mode'), np.round(finish - start, 2), "s")
#
#             node.update_cache({"U_t": U_t})
#
#         ############### root
#         if node.is_root():
#             # reconstruction U_t from B_t, U_l and U_r
#             U_l = node.left.get_cache("U_t")
#             U_r = node.right.get_cache("U_t")
#
#             B_index = node.get_cache("B_index")
#             B_value = node.get_cache("B_value")
#
#             start = time.time()
#             A_h = t_m_m_self_adaptive(B_index, B_value, U_l, U_r).astype(dtype)
#             finish = time.time()
#             print("tmm self adap: ", node.get_cache('mode'), np.round(finish - start, 2), "s")
#
#             return A_h
#
#     ############### main task
#     # 1. reconstruction A_c: chebyshev
#     poly = btree.get_cache("poly")
#     if poly != "none":
#         # # Reconstruction with (U_c, B_t)
#         btree.distribute_task(
#             NodeTask(node_reconstruction_polynomial_on_leaf,
#                      POLYVAL[poly],
#                      coor),
#             sequence="leaves_to_root", start_node_task=True)
#
#     # 1. reconstruction of A_h
#     A_h = btree.distribute_task(NodeTask(node_reconstruction), sequence="leaves_to_root", start_node_task=True)
#     return A_h

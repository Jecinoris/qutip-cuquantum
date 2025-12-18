import qutip.core.data as _data
from qutip import settings
from .state import zeros_like_cuState, CuState
from .operator import CuOperator
from .utils import _compare_hilbert

import cuquantum.densitymat as cudense
from cuquantum.densitymat import Operator

@_data.matmul.register(CuOperator, CuState, CuState)
def matmul_cuoperator_custate_custate(left, right, scale=1., out=None):    

    if left.shape[1] == right.shape[0]:
        dual = False
        merged_hilbert = _compare_hilbert(left.hilbert_dims, right.base.hilbert_space_dims)
    elif left.shape[1] == right.shape[0] * right.shape[1]:
        dual = True
        print(left.hilbert_dims[:len(left.hilbert_dims) // 2], right.base.hilbert_space_dims)
        merged_hilbert = _compare_hilbert(left.hilbert_dims[:len(left.hilbert_dims) // 2], right.base.hilbert_space_dims)
    else:
        raise ValueError("Shape missmatch")

    if not merged_hilbert:
        raise ValueError("Hilbert space missmatch")

    if(scale != 1.):
        left = left * scale
    oper = Operator(merged_hilbert, [left.to_OperatorTerm(dual=dual, hilbert_dims=merged_hilbert)])

    oper.prepare_action(settings.cuDensity["ctx"], right.base)
    if out is None:
        out = zeros_like_cuState(right)

    oper.compute_action(0, [], state_in=right.base, state_out=out.base)

    return out

@_data.matmul.register(CuState, CuOperator, CuState)
def matmul_custate_cuoperator_custate(left, right, scale=1., out=None):
    return matmul_cuoperator_custate_custate(right.transpose(), left.transpose(), scale, out).transpose()
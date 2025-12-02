import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
from qutip.tests.core.data.conftest import (
    random_csr, random_dense, random_diag
)
import pytest
import random
import numpy as np
import cupy as cp
from enum import Enum

cudense = pytest.importorskip("cuquantum.densitymat")

from qutip_cuquantum.operator import CuOperator, ProdTerm, Term
from qutip_cuquantum.utils import Transform
from qutip_cuquantum.state import CuState
from qutip_cuquantum.mixed_dispatch import matmul_cuoperator_custate_custate, matmul_custate_cuoperator_custate
import qutip_cuquantum
cudm_ctx = cudense.WorkStream()
qutip_cuquantum.set_as_default(cudm_ctx)


def _rand_transform(gen):
    """
    Random transform between raw, dag, T, conj, with bias toward common cases.
    """
    return gen.choice(list(Transform), p=[0.4, 0.15, 0.15, 0.3])


def _rand_elementary_oper(size, gen):
    if gen.uniform() < 0.5 and size > 0:
        # 50% Dia format
        mat = random_diag((size, size), gen.uniform()*0.4, False, gen)
    elif gen.uniform() < 0.6:
        # 30% Dense format
        mat = random_dense((abs(size), abs(size)), gen.uniform() > 0.5, gen)
    else:
        # 20% CSR format (not fully supported, converted to dense eventually)
        mat = random_csr((abs(size), abs(size)), gen.uniform()*0.4, False, gen)

    if gen.uniform() < 0.5 and size > 0:
        # Use cuDensity format instead of qutip.
        array_type = np if gen.uniform() < 0.5 else cp
        if isinstance(mat, _data.Dia):
            dia_matrix = mat.as_scipy()
            offsets = list(dia_matrix.offsets)
            data = array_type.zeros(
                (dia_matrix.shape[0], len(offsets)),
                dtype=complex,
            )
            for i, offset in enumerate(offsets):
                end = None if offset == 0 else -abs(offset)
                data[:end, i] = array_type.asarray( dia_matrix.diagonal(offset) )
            mat = cudense.MultidiagonalOperator(data, offsets)

        else:
            mat = cudense.DenseOperator(array_type.array(mat.to_array()))

    return mat


def random_CuOperator(hilbert_dims, N_elementary, seed):
    """
    Generate a random `CuOperator` matrix with the given hilbert_dims.
    """
    generator = np.random.default_rng(seed)
    out = CuOperator(hilbert_dims=hilbert_dims)
    for N in N_elementary:
        term = Term([], generator.normal() + 1j * generator.normal())
        for _ in range(N):
            mode = np.random.randint(len(hilbert_dims))
            size = hilbert_dims[mode]
            oper = _rand_elementary_oper(size, generator)

            term.prod_terms.append(ProdTerm(oper, (mode,), _rand_transform(generator)))
        out.terms.append(term)
    return out


def cases_cuoperator(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""

    def factory(N_elementary, seed):
        return lambda: random_CuOperator(hilbert, N_elementary, seed)

    cases = []

    cases.append(pytest.param(factory([], 0), id="zero"))
    cases.append(pytest.param(factory([0], 0), id="id"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1], seed), id=f"simple_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([3], seed), id=f"3_prods_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1, 1, 1], seed), id=f"3_terms_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1, 2, 3], seed), id=f"complex_{seed}"))

    return cases

class StateType(Enum):
    """Enumeration for quantum state types."""
    KET = 1       # Pure state, column vector (N, 1)
    BRA = 2       # Pure state, row vector (1, N)
    DM = 3        # Density matrix (N, N)
    # DM_VECTOR = 4 # Density matrix as a vector (N*N, 1)



def random_pure_custate(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    N = abs(np.prod(hilbert))
    out = (
        np.random.rand(N, 1) + 1j * np.random.rand(N, 1)
    ).astype(cp.complex128)
    out = _data.Dense(out)
    return CuState(out, hilbert, copy=False)


def random_mixed_custate(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    N = abs(np.prod(hilbert))
    out = (
        np.random.rand(N, N) + 1j * np.random.rand(N, N)
    ).astype(cp.complex128)
    out = _data.Dense(out)
    return CuState(out, hilbert, copy=False)


def random_custate(shape):
    *hilbert, state_type = shape

    if isinstance(state_type, int):
        # Special handling for shapes from generate_scalar_is_ket
        if(state_type == 1):
            state_type = StateType.KET
        else:
            raise ValueError(f"Unsupported state type: {state_type}")

    if state_type == StateType.KET:
        return random_pure_custate(hilbert)
    elif state_type == StateType.DM:
        return random_mixed_custate(hilbert)
    elif state_type == StateType.BRA:
        state = random_pure_custate(hilbert)
        ket_state = random_pure_custate(hilbert)
        return CuState(ket_state.base, shape=(ket_state.shape[1], ket_state.shape[0]), copy=False)
    # elif state_type == StateType.DM_VECTOR:
    #     state = random_mixed_custate(hilbert)
    #     return CuState(state.base, shape=(state.shape[0] * state.shape[1], 1), copy=False)
    else:
        raise ValueError(f"Unsupported state type: {state_type}")


test_tools._ALL_CASES = {
    CuOperator: cases_cuoperator,
    CuState: lambda shape: [lambda: random_custate(shape),],
}

test_tools._RANDOM = {
    CuOperator: lambda hilbert: [lambda: random_CuOperator(hilbert, [2], 0)],
    CuState: lambda shape: [lambda: random_custate(shape),],
}

_compatible_op_state = [
    (pytest.param((2,), id="single"), pytest.param((2, StateType.KET), id="single")),
    (pytest.param((2, 3), id="double"), pytest.param((2, 3, StateType.DM), id="2-dm")),
    (pytest.param((-6,), id="single_weak"), pytest.param((2, 3, StateType.KET), id="2-ket")),
    (pytest.param((2, -4), id="double_weak"), pytest.param((2, 2, 2, StateType.DM), id="3-dm")),
    (pytest.param((2, 2, 2), id="triple"), pytest.param((2, 2, 2, StateType.KET), id="3-ket")),
    # (pytest.param((2, 2, 2, 2, 2, 2), id="triple supeop"), pytest.param((2, 2, 2, StateType.DM_VECTOR), id="3-dm_vector")),
]

_imcompatible_op_state = [
    (pytest.param((2,), id="single"), pytest.param((3, StateType.DM), id="different")),
    (pytest.param((2, 3), id="double"), pytest.param((6, StateType.DM), id="merged")),
    (pytest.param((2, 3), id="double"), pytest.param((3, 2, StateType.DM), id="inverted")),
    (pytest.param((2, -4), id="double_weak"), pytest.param((4, 2, StateType.DM), id="double_weak")),
    (pytest.param((2, 3, -4), id="complex"), pytest.param((6, 2, 2, StateType.DM), id="complex")),
    (pytest.param((2,), id="dm"), pytest.param((2, StateType.BRA), id="bra")),    
]


class TestOpStateMatmul(test_tools.TestMatmul):
    specialisations = [
        pytest.param(matmul_cuoperator_custate_custate, CuOperator, CuState, CuState),
    ]

    shapes = _compatible_op_state
    bad_shapes = _imcompatible_op_state

_compatible_state_op = [
    (pytest.param((2, StateType.BRA), id="single"), pytest.param( (2,), id="single")),
    (pytest.param((2, 3, StateType.BRA), id="2-bra"), pytest.param( (2, 3), id="double")),
    (pytest.param((2, 3, StateType.DM), id="2-dm"), pytest.param((-6,), id="single_weak")),
    (pytest.param((2, 2, 2, StateType.BRA), id="3-bra"), pytest.param((2, -4), id="double_weak")),
    (pytest.param((2, 2, 2, StateType.DM), id="3-dm"), pytest.param((2, 2, 2), id="triple")),
]

_imcompatible_state_op = [
    (pytest.param((2, StateType.KET), id="single"), pytest.param( (2,), id="single")),
]

class TestStateOpMatmul(test_tools.TestMatmul):
    specialisations = [
        pytest.param(matmul_custate_cuoperator_custate, CuState, CuOperator, CuState),
    ]

    shapes = _compatible_state_op
    bad_shapes = _imcompatible_state_op


def test_mixed_dispatch_dual_op_dm():
    op = random_CuOperator((2, 3, 2, 3), [5], 0)
    state = random_custate((2, 3, StateType.DM))
    actual = matmul_cuoperator_custate_custate(op, state).to_array().ravel('F')
    expected = np.matmul(op.to_array(), state.to_array().ravel('F'))
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-7)    

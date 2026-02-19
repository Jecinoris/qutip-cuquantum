# qutip-cuquantum — AI Agent Context

## What This Package Does

`qutip-cuquantum` is a GPU-accelerated backend for [QuTiP](https://qutip.org/) (Quantum Toolbox in Python). It offloads quantum dynamics simulations to NVIDIA GPUs using NVIDIA's [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) library. It targets **large composite quantum systems** where the Hilbert space is a tensor product of multiple subsystems.

The package provides drop-in replacements for QuTiP's data types (`Data`) and ODE integrators so that existing QuTiP code (e.g. `sesolve`, `mesolve`, `mcsolve`) can run on GPU with minimal changes.

## Architecture Overview

```
qutip_cuquantum/
├── __init__.py          # Package init, QuTiP registration, set_as_default()
├── operator.py          # CuOperator — symbolic operator on GPU
├── state.py             # CuState — quantum state on GPU (pure & mixed)
├── qobjevo.pyx          # CuQobjEvo — Cython, time-dependent Hamiltonian wrapper
├── ode.py               # GPU ODE integrators (Vern7, Vern9, Tsit5) + Result class
├── callable.py          # Wraps QuTiP callable coefficients for cuQuantum callbacks
├── mixed_dispatch.py    # Operator @ State dispatch (matmul registration)
├── utils.py             # Hilbert space utilities, Transform enum
├── family.py            # QuTiP family entry point (version reporting)
└── version.py           # Auto-generated version info
```

## Key Concepts

### Data Type System

QuTiP has a pluggable data type system. This package registers two new types:

| Type | Role | Wraps |
|------|------|-------|
| `CuOperator` | Operators (Hamiltonians, collapse ops) | cuQuantum `DenseOperator` / `MultidiagonalOperator` |
| `CuState` | Quantum states (kets, density matrices) | cuQuantum `DensePureState` / `DenseMixedState` |

Both inherit from `qutip.core.data.Data`. A data type group `"cuDensity"` is registered so QuTiP's automatic dispatch can route to these types.

### WorkStream Context

All GPU operations require a `cuquantum.densitymat.WorkStream` context object. It is stored globally in `qutip.settings.cuDensity["ctx"]` and accessed by `CuOperator`/`CuState` at construction time.

### Hilbert Space Dimensions

Operators and states track which modes (subsystems) they act on via `hilbert_dims` tuples. Negative dimensions mark "weak" modes used during tensor product assembly. The `utils._compare_hilbert()` function merges Hilbert spaces when combining operators.

### Time-Dependent Systems

`CuQobjEvo` (Cython) wraps QuTiP's `QobjEvo` for GPU. Time-dependent coefficients become `CPUCallback` objects that cuQuantum calls during ODE stepping. List-format `[H0, [H1, coeff1], ...]` is supported.

## How the Backend Is Activated

```python
from cuquantum.densitymat import WorkStream
import qutip_cuquantum

# Option 1: global switch
qutip_cuquantum.set_as_default(WorkStream())

# Option 2: context manager
with qutip_cuquantum.CuQuantumBackend(WorkStream()):
    result = qutip.sesolve(H, psi0, tlist)
```

`set_as_default()` does the following:
- Sets `settings.core["default_dtype"] = "cuDensity"`
- Registers GPU integrators (`CuVern7`, `CuVern9`, `CuTsit5`) as solver defaults
- Replaces solver result classes with GPU-aware `Result`
- Disables `auto_real_casting` (required for GPU path)

## Source File Details

### `operator.py` — CuOperator

- `CuOperator(Data)`: Symbolic representation of a quantum operator.
- Internal structure: list of `Term`, each containing `ProdTerm` entries (tensor product factors).
- `ProdTerm` holds a cuQuantum `DenseOperator` or `MultidiagonalOperator`, its target Hilbert mode, and a `Transform` (direct/conj/transpose/adjoint).
- Key methods: `copy()`, `to_array()`, `conj()`, `transpose()`, `adjoint()`, `to_OperatorTerm()`.
- Arithmetic: `+`, `-`, `*` (scalar), `@` (matmul/compose), `&` (tensor/kron).
- Converts to cuQuantum's `OperatorTerm` for GPU execution.

### `state.py` — CuState

- `CuState(Data)`: Wraps cuQuantum state objects.
- Accepts: `DensePureState`, `DenseMixedState`, `cupy.ndarray`, `CuPyDense`, or any QuTiP `Data`.
- Pure states: shape `(N, 1)` — stored as `DensePureState`.
- Mixed states: shape `(N, N)` — stored as `DenseMixedState`.
- Supports MPI for multi-GPU distributed states.
- Key methods: `copy()`, `to_array()`, `to_cupy()`, `conj()`, `transpose()`, `adjoint()`.

### `qobjevo.pyx` — CuQobjEvo (Cython)

- `CuQobjEvo(QobjEvo)`: Wraps a time-dependent Hamiltonian for GPU.
- Converts list-based QobjEvo into cuQuantum `Operator` with `CPUCallback` coefficients.
- Key methods: `matmul_data(t, state)`, `expect_data(t, state)`.
- Uses `prepare_action()` / `compute_action()` from cuQuantum.

### `ode.py` — Integrators

- `CuIntegratorVern7`, `CuIntegratorVern9`, `CuIntegratorTsit5`: Wrap QuTiP's Runge-Kutta integrators.
- Convert state to `CuState` on entry, wrap system as `CuQobjEvo`.
- `Result(qt_Result)`: Custom result class that computes expectation values on GPU via `CuQobjEvo.expect`.
- `CuMCIntegrator`: Monte Carlo integrator wrapper for `mcsolve`.

### `callable.py` — Coefficient Wrapping

- `wrap_coeff(coeff)`: Wraps scalar coefficient functions for cuQuantum callbacks.
- `wrap_funcelement(element, ...)`: Wraps function-based operator elements that return a `Qobj`.

### `mixed_dispatch.py` — Operator-State Dispatch

- Registers `matmul` dispatch: `CuOperator @ CuState → CuState`.
- Uses cuQuantum's `Operator.prepare_action()` and `compute_action()`.
- Handles dual (superoperator) mode.

### `utils.py` — Utilities

- `Transform` enum: `DIRECT`, `CONJ`, `TRANSPOSE`, `ADJOINT`.
- Transform composition tables: `conj_transform`, `trans_transform`, `adjoint_transform`.
- `_compare_hilbert(left, right)`: Merge two Hilbert space dimension tuples, used when adding or composing operators.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `qutip` | `>=5.2.1` | Core quantum framework |
| `cupy` | CUDA-version specific | GPU arrays, GPU memory |
| `cuquantum-python` | CUDA-version specific | cuQuantum Python bindings |
| `numpy` | any | Array operations |
| `scipy` | any | Via qutip |
| `mpi4py` | optional | Multi-GPU / distributed |
| `qutip-cupy` | optional | CuPyDense conversions |
| `Cython` | build-time | Compiles `qobjevo.pyx` |

CUDA optional dependency groups in `pyproject.toml`:
- `cuda11`: `cupy-cuda11x`, `cuquantum-python-cu11`
- `cuda12`: `cupy-cuda12x`, `cuquantum-python-cu12`

## Testing

Tests are in `tests/` and use `pytest`. Key test files:

| File | Covers |
|------|--------|
| `test_operator.py` | CuOperator creation, arithmetic, transforms, kron, OperatorTerm conversion |
| `test_state.py` | CuState creation, arithmetic, norms, traces, pure/mixed |
| `test_solver.py` | sesolve, mesolve, mcsolve with constant and time-dependent Hamiltonians |
| `test_mixed.py` | CuOperator @ CuState matmul dispatch |
| `test_family.py` | QuTiP family entry point version reporting |
| `test_mpi.py` | Multi-GPU distributed state assembly |

Run tests: `pytest tests/` (requires GPU and all dependencies installed).

## Build System

- `pyproject.toml`: Declares build dependencies (setuptools, Cython, qutip, numpy).
- `setup.py`: Compiles `qobjevo.pyx` Cython extension using QuTiP's include dirs. Reads version from `VERSION` file, appends git hash for dev builds.
- Entry point: `[project.entry-points."qutip.family"] qutip_cuquantum = "qutip_cuquantum.family"` — registers with QuTiP's plugin system.

## Common Patterns in the Codebase

### Getting the WorkStream context
```python
ctx = settings.cuDensity["ctx"]
```

### Converting QuTiP Data to CuOperator
```python
from qutip.core import data as _data
dense = _data.to(_data.Dense, qobj.data)
array = dense.to_array()
# ... build DenseOperator or MultidiagonalOperator from array
```

### Building an OperatorTerm for cuQuantum
```python
op_term = operator.to_OperatorTerm(hilbert_dims, dual=False)
# Returns cuquantum.densitymat.OperatorTerm
```

### Executing operator action on a state
```python
from cuquantum.densitymat import Operator
cu_op = Operator(ctx, operands, operator_term)
cu_op.prepare_action(state._base)
cu_op.compute_action(t, state._base, result._base)
```

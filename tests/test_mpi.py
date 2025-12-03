import numpy as np
import cupy as cp
import pytest
import qutip
from qutip_cuquantum.state import CuState

MPI = pytest.importorskip("mpi4py.MPI")
cudm = pytest.importorskip("cuquantum.densitymat")

num_devices = cp.cuda.runtime.getDeviceCount()
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

if size < 2:
    pytest.skip("Skipping MPI tests: requires at least 2 MPI processes", allow_module_level=True)

dev = cp.cuda.Device(rank % num_devices)
dev.use()

cudm_ctx = cudm.WorkStream(device_id=dev.id)
cudm_ctx.set_communicator(comm=MPI.COMM_WORLD.Dup(), provider="MPI")

qutip.settings.cuDensity["ctx"] = cudm_ctx

SEED = 12345
cp.random.seed(SEED)
np.random.seed(SEED)

def test_mpi_pure_custate_to_cupy():
    hilbert = (20, 10, 3, 4)
    N = abs(np.prod(hilbert))
    arr = (cp.random.rand(N, 1) + 1j * cp.random.rand(N, 1)).astype(cp.complex128)    
    state = CuState(arr, hilbert, copy=False)
    cp.testing.assert_allclose(state.to_cupy(), arr, atol=1e-10, rtol=1e-7)    

def test_mpi_mixed_custate_to_cupy():
    hilbert = (20, 10, 3, 4)
    N = abs(np.prod(hilbert))
    arr = (cp.random.rand(N, N) + 1j * cp.random.rand(N, N)).astype(cp.complex128)    
    state = CuState(arr, hilbert, copy=False)
    cp.testing.assert_allclose(state.to_cupy(), arr, atol=1e-10, rtol=1e-7)    

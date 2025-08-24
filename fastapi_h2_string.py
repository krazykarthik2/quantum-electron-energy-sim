import asyncio
from fastapi import FastAPI, HTTPException, Query
from concurrent.futures import ThreadPoolExecutor
import logging
import gc
import threading
import psutil
import os
import numpy as np
from scipy.optimize import minimize

# Qiskit / PySCF imports
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms import NumPyMinimumEigensolver

from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector

# Optional memory_profiler decorator (no-op if not installed)
try:
    from memory_profiler import profile
except Exception:
    def profile(f):
        return f

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=2)   # keep small for Colab

# Simple in-memory cache
results_cache = {}
cache_lock = threading.Lock()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # MB

def build_problem_from_geometry(geometry: str):
    """
    Create driver and run it to obtain the problem (driver.run()).
    Returns (problem, nuclear_repulsion).
    """
    driver = PySCFDriver(atom=geometry, unit=DistanceUnit.ANGSTROM, basis="sto3g")
    problem = driver.run()
    nuc_rep = problem.nuclear_repulsion_energy
    # free driver ASAP (we don't need it after .run())
    del driver
    gc.collect()
    return problem, nuc_rep

@profile
def compute_both_energies(geometry: str, vqe_reps: int = 1, vqe_maxiter: int = 2000):
    """
    Compute classical exact energy and a statevector-based VQE energy (TwoLocal).
    Returns dict: {"classical": float, "vqe": float, "vqe_info": {...}}
    """
    logging.info(f"compute_both_energies: geometry={geometry}")
    mem0 = get_memory_usage()
    logging.info(f"mem before: {mem0:.2f} MB")

    problem = None
    classical_energy = None
    vqe_energy = None
    vqe_info = {}

    try:
        # Build problem
        problem, nuc_rep = build_problem_from_geometry(geometry)
        mem1 = get_memory_usage()
        logging.info(f"problem built, mem: {mem1:.2f} MB (Δ {mem1-mem0:.2f} MB)")

        # Map to qubits (Jordan-Wigner)
        mapper = JordanWignerMapper()
        second_q_ops = problem.second_q_ops()
        main_op = second_q_ops[0]                  # electronic Hamiltonian (2nd-quant)
        qubit_op = mapper.map(main_op)             # mapped qubit operator

        # Dense matrix for expectation evaluation
        H_mat = np.array(qubit_op.to_matrix(), dtype=complex)
        n_qubits = int(np.log2(H_mat.shape[0]))
        logging.info(f"H matrix size: {H_mat.shape} (n_qubits={n_qubits})")

        # === Classical exact diagonalization (electronic + nuclear repulsion) ===
        evals, _ = np.linalg.eigh(H_mat)
        classical_energy = float(np.min(evals) + nuc_rep)
        logging.info(f"classical (exact) energy: {classical_energy:.6f} Ha (nuc_rep={nuc_rep:.6f})")

        # === VQE (statevector-based) using TwoLocal ansatz ===
        # Keep ansatz small: reps = vqe_reps
        ansatz = TwoLocal(n_qubits, "ry", "cz", reps=2)
        n_params = ansatz.num_parameters
        logging.info(f"TwoLocal ansatz: qubits={ansatz.num_qubits}, params={n_params}, reps={vqe_reps}")

        # Energy function: includes nuclear repulsion
        def energy_from_params(params: np.ndarray) -> float:
            # ensure params is 1D np array
            params = np.asarray(params, dtype=float).ravel()
            # assign_parameters is the modern API
            circ = ansatz.assign_parameters(params)
            sv = Statevector.from_instruction(circ)
            vec = sv.data
            # expectation value <v|H|v> + nuclear repulsion
            return float(np.real(np.vdot(vec, H_mat @ vec)) + nuc_rep)

        # initial guess: small random values (helps escape flat regions)
        rng = np.random.default_rng(42)
        init = 0.05 * rng.standard_normal(n_params) if n_params > 0 else np.array([])

        # if there are no parameters (rare), skip optimization
        if n_params == 0:
            vqe_energy = energy_from_params(init)
            vqe_info['optimizer'] = 'none'
            vqe_info['message'] = 'ansatz has zero parameters; direct eval'
        else:
            # Run optimizer (COBYLA is robust for small parameter counts)
            try:
                res = minimize(
                    energy_from_params,
                    x0=init,
                    method="COBYLA",
                    options={"maxiter": int(vqe_maxiter), "tol": 1e-6}
                )
                vqe_energy = float(res.fun)
                vqe_info['optimizer'] = 'COBYLA'
                vqe_info['success'] = bool(res.success)
                vqe_info['message'] = str(res.message)
                vqe_info['nfev'] = int(res.nfev) if hasattr(res, 'nfev') else None
            except Exception as e:
                logging.exception("VQE optimization failed")
                vqe_energy = f"Error: VQE optimization failed: {e}"
                vqe_info['optimizer'] = 'COBYLA'
                vqe_info['success'] = False
                vqe_info['message'] = str(e)

        # cleanup local objects
        del mapper, second_q_ops, main_op, qubit_op, H_mat, ansatz
        gc.collect()

    except Exception as exc:
        logging.exception("Error in compute_both_energies")
        return {"error": str(exc)}

    finally:
        mem_end = get_memory_usage()
        logging.info(f"mem after compute: {mem_end:.2f} MB (Δ {mem_end - mem0:.2f} MB)")

    return {
        "classical": classical_energy,
        "vqe": vqe_energy,
        "vqe_info": vqe_info
    }

@app.get("/ground_state_energy")
async def ground_state_energy(
    geometry: str = Query(..., description='Molecule geometry string, e.g. "H 0 0 0; H 0 0 0.735"'),
    vqe_reps: int = Query(1, ge=1, le=3, description="TwoLocal ansatz repetitions (small integer)"),
    vqe_maxiter: int = Query(2000, ge=10, le=10000, description="Max iterations for VQE optimizer"),
    recompute: bool = Query(False, description="Force recomputation even if cached")
):

    """
    Returns both classical exact energy and VQE (statevector) energy.
    """
    mem_before = get_memory_usage()
    logging.info(f"Request geometry={geometry} | mem_before={mem_before:.2f} MB")

    # Cache check (use geometry + vqe params as key)
    cache_key = (geometry, int(vqe_reps), int(vqe_maxiter))
    with cache_lock:
        if cache_key in results_cache and not recompute:
            logging.info(f"Cache hit for {cache_key}")
            return {"energy": results_cache[cache_key], "cached": True}
        elif cache_key in results_cache and recompute:
            logging.info(f"Recomputing despite cache hit for {cache_key}")


    # offload heavy op to thread pool
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, compute_both_energies, geometry, vqe_reps, vqe_maxiter)

    # if error bubble up
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # store in cache
    with cache_lock:
        results_cache[cache_key] = result

    mem_after = get_memory_usage()
    logging.info(f"Request finished geometry={geometry} | mem_after={mem_after:.2f} MB (Δ {mem_after - mem_before:.2f} MB)")

    return {"energy": result, "cached": False}

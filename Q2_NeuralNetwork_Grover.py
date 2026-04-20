#!/usr/bin/env python3
"""
Q2 Neural Network — (čisto kvantno, bez klasičnog treniranja):
  Kvantno „učenje težina“ = GROVER / amplitude amplification nad parametrizovanim stanjem.
  Oracle označava TOP-M brojeva po istorijskoj frekvenciji iz CELOG CSV-a.
  Broj iteracija: k = round((π/4) · √(N/M)) — optimalni Grover broj.
  Statevector → Born verovatnoće → NEXT rastuća sedmorka ∈ {1..39}.
  Seed = 39; isti start uvek daje isti rezultat.

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.quantum_info import Statevector

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_QUBITS = 6          # 2^6 = 64 stanja; indeksi 0..38 -> brojevi 1..39  (default)
N_NUMBERS = 7
N_MAX = 39
K_MARKED = 7          # koliko top-frekventnih brojeva oracle obeležava   (default)

# Deterministička grid-optimizacija (nq, M, broj iteracija) po meri cos(bias, freq_csv).
# Broj iteracija: ili optimalna Grover formula k* = round((π/4)·√(N/M)), ili k* ± 1.
GRID_NQ = (6, 7, 8)
GRID_K = (3, 5, 7, 10, 15)
GRID_ITER_DELTA = (-1, 0, 1)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    """Histogram pojavljivanja brojeva 1..39 u celom H."""
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def top_k_indices(freq: np.ndarray, k: int) -> List[int]:
    """Top-k indeksi (0..38) po frekvenciji — deterministički, stabilni sort."""
    order = np.argsort(-freq, kind="stable")
    return [int(i) for i in order[:k]]


# =========================
# Grover: oracle + diffusion
# =========================
def build_oracle(nq: int, marked_indices: List[int]) -> QuantumCircuit:
    """Fazni oracle: -1 na obeleženim stanjima u komp. bazi, +1 inače."""
    diag = np.ones(2 ** nq, dtype=complex)
    for m in marked_indices:
        if 0 <= m < 2 ** nq:
            diag[m] = -1.0 + 0j
    return Diagonal(diag.tolist())


def build_diffusion(nq: int) -> QuantumCircuit:
    """Difuzor 2|s⟩⟨s| - I, gde je |s⟩ = H^⊗n |0⟩ (standardni Grover oblik)."""
    qc = QuantumCircuit(nq, name="Diff")
    qc.h(range(nq))
    qc.x(range(nq))
    qc.h(nq - 1)
    if nq >= 2:
        qc.mcx(list(range(nq - 1)), nq - 1)
    else:
        qc.z(0)
    qc.h(nq - 1)
    qc.x(range(nq))
    qc.h(range(nq))
    return qc


def optimal_iterations(n: int, m: int) -> int:
    """k = round((π/4) · √(N/M)), minimum 1 iteracija."""
    if m <= 0 or n <= 0:
        return 0
    return max(1, int(round((np.pi / 4.0) * np.sqrt(n / m))))


# =========================
# Readout: NEXT sedmorka
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    """Za indeks < n_max koristi direktno bucket = idx; za idx >= n_max dodaje po modulu."""
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


def grover_probs(nq: int, marked: List[int], k_iter: int) -> np.ndarray:
    oracle = build_oracle(nq, marked)
    diff = build_diffusion(nq)
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    for _ in range(max(0, k_iter)):
        qc.compose(oracle, range(nq), inplace=True)
        qc.compose(diff, range(nq), inplace=True)
    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2
    s = float(p.sum())
    return p / s if s > 0 else p


# =========================
# Determ. grid-optimizacija (nq, M, iter) po meri cos(bias, freq_csv)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    f_csv_n = f_csv / float(f_csv.sum() or 1.0)
    best = None
    for nq in GRID_NQ:
        N_space = 2 ** nq
        for M in GRID_K:
            marked = top_k_indices(f_csv, M)
            k_star = optimal_iterations(N_space, M)
            for d in GRID_ITER_DELTA:
                k_iter = max(1, k_star + d)
                try:
                    probs = grover_probs(nq, marked, k_iter)
                    b = bias_39(probs)
                    score = cosine(b, f_csv_n)
                except Exception:
                    continue
                key = (score, -nq, -M, -abs(d))
                if best is None or key > best[0]:
                    best = (
                        key,
                        dict(nq=nq, M=M, k_iter=k_iter, delta=d, score=score, marked=marked),
                    )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q2 NN (B): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| N:", 2 ** best["nq"],
        "| M (marked):", best["M"],
        "| iter:", best["k_iter"],
        "(Δ vs k*: {})".format(best["delta"]),
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )
    print("top-M brojevi (1..39):", sorted(int(m + 1) for m in best["marked"]))

    probs = grover_probs(best["nq"], best["marked"], best["k_iter"])
    pred = pick_next_combination(probs)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q2 NN: CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 6 | N: 64 | M (marked): 15 | iter: 2 (Δ vs k*: 0) | cos(bias, freq_csv): 0.97047
top-M brojevi (1..39): [7, 8, 9, 10, 11, 22, 23, 26, 29, 32, 33, 34, 35, 37, 39]
predikcija NEXT: (7, 8, x, y, z, 22, 23)
"""



"""
Q2_NeuralNetwork_Grover.py — Grover / Amplitude Amplification

Učita CEO CSV, izračuna histogram brojeva 1..39.
Oracle obeleži top-M najčešćih brojeva (mapirano u nq-bitne komp. bazne indekse).
Krene iz uniformne superpozicije H^⊗nq, primeni Grover iteracije (oracle + difuzor) k* puta.
Statevector → Born → bias_39 → NEXT rastuća sedmorka.
Deterministička grid-optimizacija (nq, M, Δiter oko k*) po meri cos(bias, freq_csv).

Kanonski Grover: fazni oracle (Diagonal(-1/+1)) + standardni difuzor 2|s⟩⟨s|-I.
Optimalan broj iteracija: k* = round((π/4)·√(N/M)).
Egzaktni Statevector, bez šuma i uzorkovanja.

Prednosti:
Čisto kvantno, bez klasičnog treninga.
Jako jednostavno i literaturno čisto — standardni Grover.
Deterministički izbor broja iteracija (zatvorena formula), plus grid ±1 iteracije za fino biranje.
Brzo, malo parametara.

Nedostaci:
„Marked“ skup je samo top-M po globalnoj frekvenciji — efektivno amplifikuje ono što je i onako najčešće; ne koristi strukturu CSV-a (sekvenca, korelacije parova).
Mera optimizacije cos(bias, freq_csv) u suštini vodi ka reprodukciji same frekvencije — tautološki signal. Što je M veće (bliže 39), to je poklapanje bolje bez ičeg „naučenog“.
mod-39 mapiranje 2^nq → 39 kanti briše razlike unutar kante.
Nema temporalne zavisnosti (redosled izvlačenja se ignoriše).
Eksponencijalno po nq kroz Statevector.
"""

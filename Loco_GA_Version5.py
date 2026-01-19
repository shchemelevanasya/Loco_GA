# -*- coding: utf-8 -*-
"""
Loco_GA — Genetic Algorithm for locomotive assignment

Changes in this version:
1) Dynamic automatic adjustment of criterion weights per generation based on population component magnitudes.
   - We compute raw components (idle, empty_time, mass) per individual, derive scales and set weights inversely
     proportional to mean magnitudes so each criterion contributes comparably.
   - The derived weights are recorded in generation stats.

2) Generation-curve now stores and displays per-generation statistics:
   - best, mean, std, min fitness, and the dynamic weights used that generation.
   - Streamlit plots best/mean/min for every generation.

3) Tables formatting improved:
   - Assignment and locomotive summary are shown as pandas DataFrames in Streamlit (st.dataframe/st.table),
     producing more readable, tabular views.

Multiprocessing and distance->time model are preserved.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import random
import copy
import time
import math
import logging
import concurrent.futures
import multiprocessing
import os
import statistics

import pandas as pd
import matplotlib.pyplot as plt

# Streamlit optional
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# Logging
logger = logging.getLogger("Loco_GA")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler("loco_ga.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# ---------------------
# Data classes
# ---------------------
@dataclass
class Locomotive:
    id: int
    loco_type: str
    power: float
    remaining_resource: float
    home_depot: str


@dataclass
class Train:
    id: int
    weight: float
    route: Tuple[str, str]
    departure_time: float
    duration: float


# ---------------------
# Geometry and distance->time
# ---------------------
def compute_distance_3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.hypot(math.hypot(dx, dy), dz)


def distance_to_time(distance: float,
                     speed_kmh: float = 60.0,
                     slope_elevation_diff: float = 0.0,
                     slope_penalty_coefficient: float = 0.05) -> float:
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be > 0")
    if distance <= 0:
        return 0.0
    slope = abs(slope_elevation_diff) / max(distance, 1e-6)
    penalty = 1.0 + slope_penalty_coefficient * slope
    hours = (distance / speed_kmh) * penalty
    return hours


# ---------------------
# Synthetic data (3D coords)
# ---------------------
def generate_synthetic_data(
        num_locomotives: int = 10,
        num_trains: int = 20,
        depot_names: Tuple[str, ...] = ("A", "B", "C"),
        station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
        seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
    if station_coords is None:
        station_coords = {}
        n = len(depot_names)
        radius = max(5, n) * 10.0
        for i, name in enumerate(depot_names):
            angle = 2 * math.pi * i / max(1, n)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            elev = random.uniform(-50, 200)
            station_coords[name] = (x, y, elev)

    locomotives: Dict[int, Locomotive] = {}
    for i in range(num_locomotives):
        locomotives[i] = Locomotive(
            id=i,
            loco_type="2ЭС6",
            power=random.uniform(4000, 7000),
            remaining_resource=random.uniform(20, 50),
            home_depot=random.choice(list(depot_names))
        )

    trains: Dict[int, Train] = {}
    for j in range(num_trains):
        dep = random.choice(list(depot_names))
        arr = random.choice([d for d in depot_names if d != dep])
        trains[j] = Train(
            id=j,
            weight=random.uniform(3000, 6000),
            route=(dep, arr),
            departure_time=random.uniform(0, 24),
            duration=random.uniform(2, 6)
        )

    logger.info("Synthetic data generated: %d locos, %d trains, %d stations",
                len(locomotives), len(trains), len(station_coords))
    return locomotives, trains, station_coords


# ---------------------
# Chromosome
# ---------------------
class Chromosome:
    def __init__(self, assignment: Dict[int, List[int]]):
        self.assignment = assignment
        self._fitness: Optional[float] = None

    @property
    def fitness(self) -> float:
        if self._fitness is None:
            raise RuntimeError("Fitness ещё не вычислен – вызовите fitness_function()")
        return self._fitness

    @fitness.setter
    def fitness(self, v: float):
        self._fitness = float(v)


# ---------------------
# Fast feasibility
# ---------------------
def build_lookup_tables(trains: Dict[int, Train]):
    train_weight = {tid: t.weight for tid, t in trains.items()}
    train_duration = {tid: t.duration for tid, t in trains.items()}
    train_dep = {tid: t.route[0] for tid, t in trains.items()}
    train_arr = {tid: t.route[1] for tid, t in trains.items()}
    return {"weight": train_weight, "duration": train_duration, "dep": train_dep, "arr": train_arr}


def is_feasible_fast(chromosome: Chromosome,
                     locomotives: Dict[int, Locomotive],
                     trains: Dict[int, Train]) -> bool:
    lookup = build_lookup_tables(trains)
    weight = lookup["weight"]
    duration = lookup["duration"]

    for loco_id, train_ids in chromosome.assignment.items():
        loco = locomotives.get(loco_id)
        if loco is None:
            return False
        for tid in train_ids:
            if weight[tid] > loco.power:
                return False
        total_dur = 0.0
        for tid in train_ids:
            total_dur += duration[tid]
            if total_dur > loco.remaining_resource:
                return False
    return True


# ---------------------
# Empty-run time (distance->time)
# ---------------------
def calculate_empty_run_time(chromosome: Chromosome,
                             locomotives: Dict[int, Locomotive],
                             trains: Dict[int, Train],
                             station_coords: Dict[str, Tuple[float, float, float]],
                             reposition_speed_kmh: float = 60.0,
                             slope_penalty_coefficient: float = 0.05) -> float:
    total_time = 0.0
    lookup_dep = {tid: trains[tid].route[0] for tid in trains}
    lookup_arr = {tid: trains[tid].route[1] for tid in trains}

    for loco_id, train_ids in chromosome.assignment.items():
        if not train_ids:
            continue
        loco = locomotives.get(loco_id)
        if loco is None:
            continue
        first_dep = lookup_dep[train_ids[0]]
        depot_name = loco.home_depot
        if depot_name in station_coords and first_dep in station_coords:
            depot_coord = station_coords[depot_name]
            first_coord = station_coords[first_dep]
            dist = compute_distance_3d(depot_coord, first_coord)
            elev_diff = first_coord[2] - depot_coord[2]
            total_time += distance_to_time(dist, reposition_speed_kmh, elev_diff, slope_penalty_coefficient)
        for a, b in zip(train_ids, train_ids[1:]):
            prev_arr = lookup_arr[a]
            next_dep = lookup_dep[b]
            if prev_arr in station_coords and next_dep in station_coords:
                coord_a = station_coords[prev_arr]
                coord_b = station_coords[next_dep]
                dist = compute_distance_3d(coord_a, coord_b)
                elev_diff = coord_b[2] - coord_a[2]
                total_time += distance_to_time(dist, reposition_speed_kmh, elev_diff, slope_penalty_coefficient)
    return total_time


# ---------------------
# Components and fitness
# ---------------------
def calculate_train_mass(chromosome: Chromosome, trains: Dict[int, Train]) -> float:
    return sum(trains[t_id].weight for train_ids in chromosome.assignment.values() for t_id in train_ids)


def compute_components(chromosome: Chromosome,
                       locomotives: Dict[int, Locomotive],
                       trains: Dict[int, Train],
                       station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                       reposition_speed_kmh: float = 60.0,
                       slope_penalty_coefficient: float = 0.05) -> Tuple[float, float, float]:
    """Return (idle_count, empty_time_hours, mass_total) for a chromosome."""
    idle = 1 if all(len(lst) == 0 for lst in [chromosome.assignment.get(next(iter(chromosome.assignment)), [])]) else 0
    # more correct idle: locomotive-level
    idle = sum(1 for lst in chromosome.assignment.values() if not lst)
    mass = calculate_train_mass(chromosome, trains)
    if station_coords is not None:
        empty_time = calculate_empty_run_time(chromosome, locomotives, trains, station_coords,
                                              reposition_speed_kmh, slope_penalty_coefficient)
    else:
        empty_time = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values())
    return idle, empty_time, mass


def derive_dynamic_weights(population: List[Chromosome],
                           locomotives: Dict[int, Locomotive],
                           trains: Dict[int, Train],
                           station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None) -> Tuple[float, float, float]:
    """
    Derive weights so that components contribute comparably.
    Approach:
      - compute mean absolute value for each component across population
      - set raw_weight = 1 / (mean_component + eps)
      - normalize raw_weights to sum to 1
    """
    eps = 1e-6
    comp_list = []
    for chrom in population:
        idle, empty_time, mass = compute_components(chrom, locomotives, trains, station_coords)
        comp_list.append((idle, empty_time, mass))

    if not comp_list:
        return (0.33, 0.33, 0.34)

    means = [statistics.mean([c[i] for c in comp_list]) for i in range(3)]
    # Prevent zero means; if a mean is zero, give it small positive so weight not infinite
    raw = [1.0 / (m + eps) for m in means]
    s = sum(raw)
    weights = tuple(r / s for r in raw)
    return weights


def fitness_function(chromosome: Chromosome,
                     locomotives: Dict[int, Locomotive],
                     trains: Dict[int, Train],
                     station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                     weights=(0.4, 0.3, 0.3),
                     reposition_speed_kmh: float = 60.0,
                     slope_penalty_coefficient: float = 0.05) -> float:
    idle = sum(1 for lst in chromosome.assignment.values() if not lst)
    mass = calculate_train_mass(chromosome, trains)
    if station_coords is not None:
        empty_time = calculate_empty_run_time(chromosome, locomotives, trains, station_coords,
                                              reposition_speed_kmh, slope_penalty_coefficient)
    else:
        empty_time = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values())

    mass_scale = max(1.0, sum(t.weight for t in trains.values()))
    empty_scale = max(1.0, empty_time)
    idle_scale = max(1.0, len(locomotives))

    mass_n = mass / mass_scale
    empty_n = empty_time / empty_scale
    idle_n = idle / idle_scale

    fitness = -weights[0] * idle_n - weights[1] * empty_n + weights[2] * mass_n
    chromosome.fitness = fitness
    return fitness


# ---------------------
# Population initialization
# ---------------------
def generate_initial_population(population_size: int,
                                locomotives: Dict[int, Locomotive],
                                trains: Dict[int, Train],
                                max_attempts: int = 1000) -> List[Chromosome]:
    population: List[Chromosome] = []
    loco_ids = list(locomotives.keys())
    train_ids_master = list(trains.keys())
    attempts = 0
    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        assignment = {lid: [] for lid in loco_ids}
        train_ids = train_ids_master[:]
        random.shuffle(train_ids)
        for t in train_ids:
            assignment[random.choice(loco_ids)].append(t)
        chrom = Chromosome(assignment)
        if is_feasible_fast(chrom, locomotives, trains):
            population.append(chrom)
    if not population:
        assignment = {lid: [] for lid in loco_ids}
        for i, t in enumerate(train_ids_master):
            assignment[loco_ids[i % len(loco_ids)]].append(t)
        population.append(Chromosome(assignment))
        logger.warning("Initial feasible population not found stochastically, fallback used")
    return population


# ---------------------
# Genetic operators
# ---------------------
def tournament_selection(population: List[Chromosome], k: int = 3) -> Chromosome:
    if not population:
        raise ValueError("Популяция пуста – нельзя провести отбор")
    k = min(k, len(population))
    candidates = random.sample(population, k)
    return max(candidates, key=lambda c: c.fitness)


def crossover_assignment(assign1: Dict[int, List[int]], assign2: Dict[int, List[int]]) -> Dict[int, List[int]]:
    child_assignment: Dict[int, List[int]] = {}
    for loco_id in assign1.keys():
        if random.random() < 0.5:
            child_assignment[loco_id] = copy.deepcopy(assign1[loco_id])
        else:
            child_assignment[loco_id] = copy.deepcopy(assign2.get(loco_id, []))
    return child_assignment


def mutation_assignment(assignment: Dict[int, List[int]], mutation_rate: float = 0.1):
    loco_ids = list(assignment.keys())
    for loco_id, trains_list in list(assignment.items()):
        if trains_list and random.random() < mutation_rate:
            t = random.choice(trains_list)
            trains_list.remove(t)
            dest = random.choice(loco_ids)
            assignment[dest].append(t)


# ---------------------
# Multiprocessing workers (unchanged)
# ---------------------
def _fitness_worker_serial(args):
    assignment, locomotives, trains, station_coords, weights, reposition_speed_kmh, slope_penalty_coefficient = args
    chrom = Chromosome(assignment)
    fitness_function(chrom, locomotives, trains, station_coords, weights,
                     reposition_speed_kmh, slope_penalty_coefficient)
    return chrom.assignment, chrom.fitness


def _child_worker_serial(args):
    assign_p1, assign_p2, locomotives, trains, mutation_rate = args
    child_assign = crossover_assignment(assign_p1, assign_p2)
    mutation_assignment(child_assign, mutation_rate)
    child = Chromosome(child_assign)
    if is_feasible_fast(child, locomotives, trains):
        return child_assign
    return None


# ---------------------
# GeneticAlgorithm with dynamic weights and stats logging
# ---------------------
class GeneticAlgorithm:
    def __init__(self, locomotives: Dict[int, Locomotive], trains: Dict[int, Train],
                 population_size: int = 50, generations: int = 100,
                 tournament_k: int = 3, mutation_rate: float = 0.1,
                 weights=(0.4, 0.3, 0.3),
                 station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                 progress_callback: Optional[callable] = None,
                 use_multiprocessing: bool = True,
                 multiprocessing_threshold: int = 100):
        self.locomotives = locomotives
        self.trains = trains
        self.population_size = max(1, int(population_size))
        self.generations = max(1, int(generations))
        self.tournament_k = max(1, min(int(tournament_k), self.population_size))
        self.mutation_rate = float(mutation_rate)
        self.initial_weights = tuple(float(w) for w in weights)
        self.station_coords = station_coords
        self.progress_callback = progress_callback
        self.use_multiprocessing = bool(use_multiprocessing)
        self.multiprocessing_threshold = int(multiprocessing_threshold)
        self.cpu_count = max(1, multiprocessing.cpu_count())

    def _evaluate_population_with_dynamic_weights(self, population: List[Chromosome]) -> Dict[str, Any]:
        """
        1) Compute raw components for each individual (idle, empty_time, mass)
        2) Derive dynamic weights
        3) Compute fitness per individual using derived weights (possibly in parallel)
        4) Return generation stats: best, mean, std, min, weights
        """
        # compute components serially (empty_time is relatively heavy)
        comps = []
        for chrom in population:
            idle, empty_time, mass = compute_components(chrom, self.locomotives, self.trains, self.station_coords)
            comps.append((idle, empty_time, mass))

        # derive weights
        weights = derive_dynamic_weights(population, self.locomotives, self.trains, self.station_coords)

        # evaluate fitness (parallel if beneficial)
        if self.use_multiprocessing and len(population) >= self.multiprocessing_threshold and self.cpu_count > 1:
            args = [
                (chrom.assignment, self.locomotives, self.trains, self.station_coords, weights, 60.0, 0.05)
                for chrom in population
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(args))) as exc:
                for assignment, fitness in exc.map(_fitness_worker_serial, args):
                    for chrom in population:
                        if chrom.assignment == assignment:
                            chrom.fitness = fitness
                            break
        else:
            for chrom in population:
                fitness_function(chrom, self.locomotives, self.trains, self.station_coords, weights)

        fitnesses = [chrom.fitness for chrom in population]
        best = max(fitnesses) if fitnesses else float("-inf")
        mean = statistics.mean(fitnesses) if fitnesses else float("nan")
        std = statistics.pstdev(fitnesses) if fitnesses else float("nan")
        mn = min(fitnesses) if fitnesses else float("inf")

        stats = {"best": best, "mean": mean, "std": std, "min": mn, "weights": weights}
        return stats

    def _generate_children(self, population: List[Chromosome], target_count: int) -> List[Chromosome]:
        children: List[Chromosome] = []
        if self.use_multiprocessing and len(population) >= self.multiprocessing_threshold and self.cpu_count > 1:
            parent_pairs = []
            for _ in range(target_count * 2):
                p1 = random.choice(population)
                p2 = random.choice(population)
                parent_pairs.append((p1.assignment, p2.assignment, self.locomotives, self.trains, self.mutation_rate))
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(parent_pairs))) as exc:
                for result in exc.map(_child_worker_serial, parent_pairs):
                    if result is not None:
                        children.append(Chromosome(result))
                    if len(children) >= target_count:
                        break
        else:
            attempts = 0
            while len(children) < target_count and attempts < target_count * 20:
                attempts += 1
                p1 = tournament_selection(population, self.tournament_k)
                p2 = tournament_selection(population, self.tournament_k)
                child_assign = crossover_assignment(p1.assignment, p2.assignment)
                mutation_assignment(child_assign, self.mutation_rate)
                child = Chromosome(child_assign)
                if is_feasible_fast(child, self.locomotives, self.trains):
                    children.append(child)
        return children

    def run(self) -> Chromosome:
        population = generate_initial_population(self.population_size, self.locomotives, self.trains)
        generation_stats = []
        for gen in range(self.generations):
            stats = self._evaluate_population_with_dynamic_weights(population)
            # record stats
            generation_stats.append({"gen": gen, **stats})
            if self.progress_callback:
                try:
                    # progress_callback now expected to accept (gen, stats_dict)
                    self.progress_callback(gen, stats)
                except TypeError:
                    # backward compatibility: pass best only
                    try:
                        self.progress_callback(gen, stats["best"])
                    except Exception:
                        pass
                except Exception:
                    pass

            children = self._generate_children(population, self.population_size)
            new_population = children
            if len(new_population) < self.population_size:
                elites = sorted(population, key=lambda c: c.fitness, reverse=True)
                i = 0
                while len(new_population) < self.population_size:
                    e = elites[i % len(elites)]
                    new_population.append(Chromosome(copy.deepcopy(e.assignment)))
                    i += 1
            population = new_population

        # final evaluation and final stats
        final_stats = self._evaluate_population_with_dynamic_weights(population)
        generation_stats.append({"gen": self.generations, **final_stats})
        # store stats on object for later inspection
        self.generation_stats = generation_stats
        best = max(population, key=lambda c: c.fitness)
        logger.info("GA finished: best fitness = %.6f", best.fitness)
        return best


# ---------------------
# Reporting & DataFrames for nicer tables
# ---------------------
class GAReporter:
    def __init__(self):
        self.generation_log: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def log_generation(self, gen: int, stats: Dict[str, Any]):
        # stats is dict with keys 'best','mean','std','min','weights'
        entry = {"gen": gen, "best": stats.get("best"), "mean": stats.get("mean"),
                 "std": stats.get("std"), "min": stats.get("min"), "weights": stats.get("weights")}
        self.generation_log.append(entry)

    def elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def print_summary(self, solution: Chromosome):
        print("\n=== 1. Время расчёта и итоговая пригодность ===")
        print(f"Время расчёта, с: {self.elapsed():.2f}")
        print(f"Итоговая целевая функция: {solution.fitness:.6f}")


def build_assignment_dataframe(solution: Chromosome,
                               locomotives: Dict[int, Locomotive],
                               trains: Dict[int, Train]) -> pd.DataFrame:
    rows = []
    for loco_id, train_ids in solution.assignment.items():
        loco = locomotives.get(loco_id)
        depot = loco.home_depot if loco else ""
        remaining = loco.remaining_resource if loco else None
        if not train_ids:
            rows.append({
                "Locomotive": loco_id,
                "Depot": depot,
                "Remaining_resource_h": remaining,
                "Train_id": None,
                "From": None,
                "To": None,
                "Departure": None,
                "Arrival": None,
                "Train_weight": None,
                "Train_duration": None
            })
        else:
            for t_id in train_ids:
                t = trains[t_id]
                rows.append({
                    "Locomotive": loco_id,
                    "Depot": depot,
                    "Remaining_resource_h": remaining,
                    "Train_id": t.id,
                    "From": t.route[0],
                    "To": t.route[1],
                    "Departure": round(t.departure_time, 2),
                    "Arrival": round(t.departure_time + t.duration, 2),
                    "Train_weight": round(t.weight, 1),
                    "Train_duration": round(t.duration, 2)
                })
    df = pd.DataFrame(rows, columns=[
        "Locomotive", "Depot", "Remaining_resource_h", "Train_id", "From", "To",
        "Departure", "Arrival", "Train_weight", "Train_duration"
    ])
    return df


def build_locomotive_summary_dataframe(locomotives: Dict[int, Locomotive],
                                       trains: Dict[int, Train],
                                       solution: Chromosome) -> pd.DataFrame:
    rows = []
    for loco_id, loco in locomotives.items():
        train_ids = solution.assignment.get(loco_id, [])
        if train_ids:
            last_train = trains[train_ids[-1]]
            current_location = last_train.route[1]
            remaining_resource = loco.remaining_resource - sum(trains[t_id].duration for t_id in train_ids)
        else:
            current_location = loco.home_depot
            remaining_resource = loco.remaining_resource
        rows.append({
            "Locomotive": loco_id,
            "Depot": loco.home_depot,
            "Current_location": current_location,
            "Remaining_resource_h": round(remaining_resource, 2),
            "Assigned_trains_count": len(train_ids)
        })
    df = pd.DataFrame(rows, columns=["Locomotive", "Depot", "Current_location", "Remaining_resource_h", "Assigned_trains_count"])
    return df


def plot_generation_curve(reporter: GAReporter):
    if not reporter.generation_log:
        fig, ax = plt.subplots()
        ax.set_title("No generation data")
        return fig
    df = pd.DataFrame(reporter.generation_log)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["gen"], df["best"], label="best", marker="o")
    ax.plot(df["gen"], df["mean"], label="mean", marker="o")
    ax.plot(df["gen"], df["min"], label="min", marker="o")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Generation curve (best / mean / min)")
    ax.legend()
    ax.grid(True)
    return fig


def plot_weights_evolution(reporter: GAReporter):
    if not reporter.generation_log:
        fig, ax = plt.subplots()
        ax.set_title("No weights data")
        return fig
    gens = [e["gen"] for e in reporter.generation_log]
    ws = [e["weights"] for e in reporter.generation_log]
    w_idle = [w[0] for w in ws]
    w_empty = [w[1] for w in ws]
    w_mass = [w[2] for w in ws]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(gens, w_idle, label="w_idle", marker="o")
    ax.plot(gens, w_empty, label="w_empty", marker="o")
    ax.plot(gens, w_mass, label="w_mass", marker="o")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weight")
    ax.set_title("Weights evolution")
    ax.legend()
    ax.grid(True)
    return fig


def plot_assignment_matplotlib(solution: Chromosome, trains: Dict[int, Train]):
    fig, ax = plt.subplots(figsize=(10, max(4, len(solution.assignment) * 0.3)))
    y = 0
    ylabels = []
    for loco_id, train_ids in solution.assignment.items():
        for t_id in train_ids:
            t = trains[t_id]
            ax.barh(y, t.duration, left=t.departure_time, height=0.4)
        ylabels.append(str(loco_id))
        y += 1
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Locomotives (id)")
    ax.set_title("Assignment timeline")
    plt.tight_layout()
    return fig


# ---------------------
# Streamlit UI with improved displays
# ---------------------
def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not available in this environment. Install streamlit to use UI.")
        return

    st.set_page_config(page_title="Loco_GA", layout="wide")
    st.title("Loco_GA — Genetic Algorithm for Locomotive Assignment")
    st.sidebar.header("Settings")

    num_locomotives = st.sidebar.slider("Number of locomotives", 1, 200, 10)
    num_trains = st.sidebar.slider("Number of trains", 1, 500, 20)
    depot_names_str = st.sidebar.text_input("Stations/Depots (comma-separated)", "A,B,C")
    depot_names = tuple(s.strip() for s in depot_names_str.split(",") if s.strip())
    seed = st.sidebar.number_input("Random seed (0 means random)", min_value=0, value=0, step=1)
    if seed == 0:
        seed = None

    population_size = st.sidebar.number_input("Population size", min_value=2, max_value=2000, value=60)
    generations = st.sidebar.number_input("Generations", min_value=1, max_value=2000, value=120)
    tournament_k = st.sidebar.number_input("Tournament size", min_value=1, max_value=population_size, value=5)
    mutation_rate = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.15, 0.01)

    # initial weights are shown but will be overridden by dynamic derivation each generation
    weights_idle = st.sidebar.slider("Initial weight: idle (will be adapted)", 0.0, 1.0, 0.4, 0.05)
    weights_empty = st.sidebar.slider("Initial weight: empty (will be adapted)", 0.0, 1.0, 0.3, 0.05)
    weights_mass = st.sidebar.slider("Initial weight: mass (will be adapted)", 0.0, 1.0, 0.3, 0.05)
    w_sum = max(1e-6, weights_idle + weights_empty + weights_mass)
    init_weights = (weights_idle / w_sum, weights_empty / w_sum, weights_mass / w_sum)

    use_mp = st.sidebar.checkbox("Use multiprocessing for GA internals (when large)", value=True)
    mp_threshold = st.sidebar.number_input("MP threshold (population size)", min_value=2, max_value=500, value=100)

    if st.button("Generate data"):
        with st.spinner("Generating synthetic data..."):
            locomotives, trains, station_coords = generate_synthetic_data(
                num_locomotives=num_locomotives,
                num_trains=num_trains,
                depot_names=depot_names,
                seed=seed
            )
            st.session_state["locomotives"] = locomotives
            st.session_state["trains"] = trains
            st.session_state["station_coords"] = station_coords
        st.success("Data generated")

    if "locomotives" not in st.session_state:
        locomotives, trains, station_coords = generate_synthetic_data(
            num_locomotives=num_locomotives,
            num_trains=num_trains,
            depot_names=depot_names,
            seed=seed
        )
        st.session_state["locomotives"] = locomotives
        st.session_state["trains"] = trains
        st.session_state["station_coords"] = station_coords

    locomotives = st.session_state["locomotives"]
    trains = st.session_state["trains"]
    station_coords = st.session_state["station_coords"]

    st.sidebar.markdown(f"Locomotives: {len(locomotives)}")
    st.sidebar.markdown(f"Trains: {len(trains)}")
    st.sidebar.markdown(f"Stations: {len(station_coords)}")

    # UI placeholders
    col1, col2 = st.columns((2, 1))
    with col2:
        st.markdown("### Current generation stats")
        gen_stats_placeholder = st.empty()
        st.markdown("### Weights evolution")
        weights_plot_placeholder = st.empty()

    with col1:
        st.markdown("### Assignment")
        assignment_placeholder = st.empty()
        st.markdown("### Locomotive summary")
        loco_summary_placeholder = st.empty()

    reporter = GAReporter()
    reporter.start()

    run_button = st.button("Run Genetic Algorithm (synchronous)")

    if run_button:
        def progress_callback(gen: int, stats: Dict[str, Any]):
            # update current gen stats
            ws = stats.get("weights", (0.0, 0.0, 0.0))
            gen_stats_placeholder.metric(label=f"Gen {gen}", value=f"best {stats['best']:.6f}", delta=f"mean {stats['mean']:.6f}")
            # append to reporter
            reporter.log_generation(gen, stats)

        try:
            ga = GeneticAlgorithm(
                locomotives, trains,
                population_size=int(population_size),
                generations=int(generations),
                tournament_k=int(tournament_k),
                mutation_rate=float(mutation_rate),
                weights=init_weights,
                station_coords=station_coords,
                progress_callback=progress_callback,
                use_multiprocessing=bool(use_mp),
                multiprocessing_threshold=int(mp_threshold)
            )
            with st.spinner("Running GA... this may take a while for large populations"):
                best = ga.run()
            st.success("GA finished")
            st.session_state["solution"] = best
            st.session_state["reporter"] = reporter
        except Exception as e:
            logger.exception("Error during GA run: %s", e)
            st.error(f"GA error: {e}")

    if "solution" in st.session_state:
        solution: Chromosome = st.session_state["solution"]
        reporter: GAReporter = st.session_state.get("reporter", reporter)

        # Assignment table as DataFrame
        df_assign = build_assignment_dataframe(solution, locomotives, trains)
        assignment_placeholder.dataframe(df_assign.style.format({
            "Remaining_resource_h": "{:.1f}",
            "Departure": "{:.2f}",
            "Arrival": "{:.2f}",
            "Train_weight": "{:.1f}",
            "Train_duration": "{:.2f}"
        }), height=400)

        # Locomotive summary
        df_loco = build_locomotive_summary_dataframe(locomotives, trains, solution)
        loco_summary_placeholder.table(df_loco)

        # Timeline plot
        fig_timeline = plot_assignment_matplotlib(solution, trains)
        st.pyplot(fig_timeline)

        # Generation curve and weights evolution
        fig_gc = plot_generation_curve(reporter)
        st.pyplot(fig_gc)
        fig_ws = plot_weights_evolution(reporter)
        weights_plot_placeholder.pyplot(fig_ws)


# ---------------------
# CLI fallback
# ---------------------
def run_cli_demo():
    locomotives, trains, station_coords = generate_synthetic_data()
    ga = GeneticAlgorithm(locomotives, trains,
                          population_size=60, generations=40,
                          tournament_k=5, mutation_rate=0.15,
                          station_coords=station_coords,
                          use_multiprocessing=True,
                          multiprocessing_threshold=20)
    reporter = GAReporter()
    reporter.start()
    best = ga.run()
    reporter.print_summary(best)
    print(build_assignment_dataframe(best, locomotives, trains))
    fig = plot_assignment_matplotlib(best, trains)
    plt.show()


# ---------------------
# Entrypoint
# ---------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE and os.environ.get("RUN_STREAMLIT", "1") == "1":
        run_streamlit_app()
    else:
        logger.info("Streamlit not available or disabled — running CLI demo")
        run_cli_demo()
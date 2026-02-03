# -*- coding: utf-8 -*-
"""
Loco_GA — Genetic Algorithm for locomotive assignment
Complete corrected and optimized version:
 - Мутации: swap_locos, replace_loco, range_shuffle
 - Кроссоверы: one_point, two_point, uniform, priority
 - Адаптивная вероятность мутации
 - Адаптивный выбор операторов кроссовера (внутри выбранного набора)
 - Возможность выбора набора операторов и задания базовых вероятностей для каждого оператора
 - Fitness: raw objective stored (penalty). Поддержка minimize (по умолчанию) и maximize.
 - Отчёты: best_signed и интерпретация
 - Улучшённые графики и таблицы, экспорт PNG / CSV
 - Streamlit UI: загрузка данных, выбор операторов/вероятностей, Minimize/Maximize, скачивание графиков/таблиц
 - Защита от отсутствия seaborn: надёжный fallback для matplotlib style
 - OPTIMIZED: Reduced memory footprint, faster computation, better caching
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
import random
import copy
import time
import math
import logging
import concurrent.futures
import multiprocessing
import os
import statistics
import io
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Try seaborn for nicer style if available
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    USE_SEABORN = True
except Exception:
    USE_SEABORN = False
    _preferred_styles = ["seaborn-v0_8", "seaborn", "ggplot", "tableau-colorblind10", "classic", "default"]
    for _style in _preferred_styles:
        try:
            plt.style.use(_style)
            break
        except Exception:
            continue

# Global Matplotlib rc settings for academic figures
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.constrained_layout.use": True
})

# Streamlit optional
try:
    import streamlit as st
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
    try:
        fh = logging.FileHandler("loco_ga.log", mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass


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
    reposition_speed_kmh: float


@dataclass
class Train:
    id: int
    weight: float
    route: Tuple[str, str]
    departure_time: float
    duration: float
    category: int = 0


@dataclass
class TrainLookup:
    """Optimized lookup tables for trains"""
    weight: Dict[int, float] = field(default_factory=dict)
    duration: Dict[int, float] = field(default_factory=dict)
    dep: Dict[int, str] = field(default_factory=dict)
    arr: Dict[int, str] = field(default_factory=dict)
    category: Dict[int, int] = field(default_factory=dict)
    dep_time: Dict[int, float] = field(default_factory=dict)


# ---------------------
# Utilities (exporting)
# ---------------------
def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data


def download_link_bytes(data: bytes, filename: str, mime: str) -> str:
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------
# Geometry and time conversion
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
# Synthetic data
# ---------------------
def generate_synthetic_data(
        num_locomotives: int = 10,
        num_trains: int = 20,
        depot_names: Tuple[str, ...] = ("A", "B", "C"),
        station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
        loco_types: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)

    if loco_types is None:
        loco_types = {
            "2ЭС6": {"power_range": (4500, 6500), "speed_kmh": 60.0, "resource_range": (20, 50)},
            "ЧС7": {"power_range": (3000, 5000), "speed_kmh": 50.0, "resource_range": (15, 40)},
            "ТЭП70": {"power_range": (5000, 7500), "speed_kmh": 70.0, "resource_range": (25, 60)}
        }

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
    type_names = list(loco_types.keys())
    for i in range(num_locomotives):
        tname = random.choice(type_names)
        props = loco_types[tname]
        power = random.uniform(props["power_range"][0], props["power_range"][1])
        resource = random.uniform(props["resource_range"][0], props["resource_range"][1])
        speed = float(props.get("speed_kmh", 60.0))
        locomotives[i] = Locomotive(
            id=i,
            loco_type=tname,
            power=power,
            remaining_resource=resource,
            home_depot=random.choice(list(depot_names)),
            reposition_speed_kmh=speed
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
            duration=random.uniform(2, 6),
            category=random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
        )

    logger.info("Generated %d locomotives, %d trains, %d stations", len(locomotives), len(trains), len(station_coords))
    return locomotives, trains, station_coords


# ---------------------
# Chromosome and feasibility
# ---------------------
class Chromosome:
    def __init__(self, assignment: Dict[int, List[int]]):
        self.assignment = assignment
        self._fitness: Optional[float] = None

    @property
    def fitness(self) -> float:
        if self._fitness is None:
            raise RuntimeError("Fitness not computed")
        return self._fitness

    @fitness.setter
    def fitness(self, v: float):
        self._fitness = float(v)


def build_train_lookup(trains: Dict[int, Train]) -> TrainLookup:
    """Optimized: Build lookup tables once, reuse them"""
    lookup = TrainLookup()
    for tid, t in trains.items():
        lookup.weight[tid] = t.weight
        lookup.duration[tid] = t.duration
        lookup.dep[tid] = t.route[0]
        lookup.arr[tid] = t.route[1]
        lookup.category[tid] = t.category
        lookup.dep_time[tid] = t.departure_time
    return lookup


def is_feasible_fast(chromosome: Chromosome,
                     locomotives: Dict[int, Locomotive],
                     train_lookup: TrainLookup) -> bool:
    """Optimized: Use pre-built lookup tables"""
    weight = train_lookup.weight
    duration = train_lookup.duration

    for loco_id, train_ids in chromosome.assignment.items():
        loco = locomotives.get(loco_id)
        if loco is None:
            return False
        for tid in train_ids:
            if weight.get(tid, float('inf')) > loco.power:
                return False
        total_dur = sum(duration.get(tid, 0.0) for tid in train_ids)
        if total_dur > loco.remaining_resource:
            return False
    return True


# ---------------------
# Components & fitness
# ---------------------
def compute_time_components_for_chromosome(chrom: Chromosome,
                                           locomotives: Dict[int, Locomotive],
                                           train_lookup: TrainLookup,
                                           station_coords: Dict[str, Tuple[float, float, float]],
                                           slope_penalty_coefficient: float = 0.05) -> Dict[str, float]:
    idle_time_sum = 0.0
    empty_time_sum = 0.0
    train_wait_time_sum = 0.0
    loco_wait_time_sum = 0.0
    used_locos = 0

    for loco_id, train_ids in chrom.assignment.items():
        if not train_ids:
            continue
        used_locos += 1
        sorted_trains = sorted(train_ids, key=lambda t_id: train_lookup.dep_time[t_id])
        loco = locomotives[loco_id]
        loco_available_time = 0.0
        current_loc = loco.home_depot
        
        for t_id in sorted_trains:
            dep_station = train_lookup.dep[t_id]
            if current_loc in station_coords and dep_station in station_coords:
                coord_a = station_coords[current_loc]
                coord_b = station_coords[dep_station]
                dist = compute_distance_3d(coord_a, coord_b)
                elev_diff = coord_b[2] - coord_a[2]
                reposition_time = distance_to_time(dist, speed_kmh=loco.reposition_speed_kmh,
                                                   slope_elevation_diff=elev_diff,
                                                   slope_penalty_coefficient=slope_penalty_coefficient)
            else:
                reposition_time = 0.0
            
            loco_arrival = loco_available_time + reposition_time
            train_dep_time = train_lookup.dep_time[t_id]
            
            if loco_arrival > train_dep_time:
                train_wait_time_sum += (loco_arrival - train_dep_time)
            else:
                loco_wait_time_sum += (train_dep_time - loco_arrival)
            
            empty_time_sum += reposition_time
            actual_departure = max(train_dep_time, loco_arrival)
            loco_available_time = actual_departure + train_lookup.duration[t_id]
            current_loc = train_lookup.arr[t_id]
        
        idle_time_sum += loco_wait_time_sum

    return {
        "idle_time_h": idle_time_sum,
        "empty_time_h": empty_time_sum,
        "train_wait_time_h": train_wait_time_sum,
        "loco_wait_time_h": loco_wait_time_sum,
        "used_locos_count": used_locos
    }


def compute_components(chromosome: Chromosome,
                       locomotives: Dict[int, Locomotive],
                       train_lookup: Optional[TrainLookup] = None,
                       station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None) -> Tuple[float, float, float, float, float]:
    if station_coords is None or train_lookup is None:
        idle_count = sum(1 for lst in chromosome.assignment.values() if not lst)
        empty = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values())
        train_wait = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values()) * 0.5
        loco_wait = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values()) * 0.5
        used_locos = sum(1 for lst in chromosome.assignment.values() if lst)
        return float(idle_count), float(empty), float(train_wait), float(loco_wait), float(used_locos)
    
    metrics = compute_time_components_for_chromosome(chromosome, locomotives, train_lookup, station_coords)
    return (metrics["idle_time_h"], metrics["empty_time_h"], metrics["train_wait_time_h"],
            metrics["loco_wait_time_h"], float(metrics["used_locos_count"]))


def fitness_function_components_based(chromosome: Chromosome,
                                       locomotives: Dict[int, Locomotive],
                                       train_lookup: Optional[TrainLookup] = None,
                                       station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                                       weights: Optional[Tuple[float, float, float, float, float]] = None,
                                       eps: float = 1e-6) -> float:
    comps = compute_components(chromosome, locomotives, train_lookup, station_coords)
    if weights is None:
        weights = (0.2, 0.2, 0.2, 0.2, 0.2)
    comp_norm = [comps[i] / (1.0 + comps[i]) for i in range(5)]
    penalty = sum(weights[i] * comp_norm[i] for i in range(5))
    chromosome.fitness = penalty
    return penalty


# ---------------------
# Helpers: gene conversion
# ---------------------
def ordered_train_ids(trains: Dict[int, Train]) -> List[int]:
    return sorted(trains.keys())


def assignment_to_gene_list(assignment: Dict[int, List[int]], train_ids: List[int]) -> List[int]:
    gene = [None] * len(train_ids)
    train_to_index = {tid: idx for idx, tid in enumerate(train_ids)}
    for loco_id, tlist in assignment.items():
        for t in tlist:
            if t in train_to_index:
                gene[train_to_index[t]] = loco_id
    for i in range(len(gene)):
        if gene[i] is None:
            if assignment:
                gene[i] = next(iter(assignment.keys()))
            else:
                gene[i] = 0
    return gene


def gene_list_to_assignment(gene: List[int], train_ids: List[int], loco_ids: List[int]) -> Dict[int, List[int]]:
    assign = {lid: [] for lid in loco_ids}
    for idx, loco in enumerate(gene):
        tid = train_ids[idx]
        if loco not in assign:
            assign[random.choice(loco_ids)].append(tid)
        else:
            assign[loco].append(tid)
    return assign


# ---------------------
# Genetic operators
# ---------------------
def tournament_selection(population: List[Chromosome], k: int = 3, maximize: bool = False) -> Chromosome:
    if not population:
        raise ValueError("Empty population")
    k = min(k, len(population))
    candidates = random.sample(population, k)
    if maximize:
        return max(candidates, key=lambda c: c.fitness)
    else:
        return min(candidates, key=lambda c: c.fitness)


def crossover_assignments(assign1: Dict[int, List[int]],
                          assign2: Dict[int, List[int]],
                          train_ids: List[int],
                          train_lookup: TrainLookup,
                          method: str = "uniform") -> Dict[int, List[int]]:
    loco_ids = sorted(list(set(list(assign1.keys()) + list(assign2.keys()))))
    g1 = assignment_to_gene_list(assign1, train_ids)
    g2 = assignment_to_gene_list(assign2, train_ids)
    n = len(g1)
    child = [None] * n

    if method == "one_point":
        if n <= 1:
            child = g1[:] if random.random() < 0.5 else g2[:]
        else:
            pt = random.randint(1, n - 1)
            child[:pt] = g1[:pt]
            child[pt:] = g2[pt:]
    elif method == "two_point":
        if n <= 2:
            child = g1[:] if random.random() < 0.5 else g2[:]
        else:
            a = random.randint(0, n - 2)
            b = random.randint(a + 1, n - 1)
            child[:a] = g1[:a]
            child[a:b] = g2[a:b]
            child[b:] = g1[b:]
    elif method == "uniform":
        for i in range(n):
            child[i] = g1[i] if random.random() < 0.5 else g2[i]
    elif method == "priority":
        cats = [train_lookup.category.get(tid, 0) for tid in train_ids]
        max_cat = max(cats) if cats else 1
        for i in range(n):
            cat = cats[i]
            p1 = 0.5 if max_cat == 0 else 0.4 + 0.6 * (cat / max_cat)
            child[i] = g1[i] if random.random() < p1 else g2[i]
    else:
        for i in range(n):
            child[i] = g1[i] if random.random() < 0.5 else g2[i]

    child_assign = gene_list_to_assignment(child, train_ids, loco_ids)
    return child_assign


def mutation_assignment(assignment: Dict[int, List[int]],
                        mutation_rate: float = 0.1,
                        methods: Optional[List[str]] = None):
    if methods is None:
        methods = ["swap_locos", "replace_loco", "range_shuffle"]

    loco_ids = list(assignment.keys())
    if not loco_ids:
        return

    if "swap_locos" in methods and random.random() < mutation_rate:
        if len(loco_ids) >= 2:
            a, b = random.sample(loco_ids, 2)
            assignment[a], assignment[b] = assignment[b], assignment[a]

    if "replace_loco" in methods:
        all_trains = []
        for lid in loco_ids:
            all_trains.extend([(lid, t) for t in assignment[lid]])
        for (from_lid, t) in all_trains:
            if random.random() < mutation_rate:
                to_lid = random.choice(loco_ids)
                if to_lid != from_lid:
                    try:
                        assignment[from_lid].remove(t)
                    except ValueError:
                        continue
                    assignment[to_lid].append(t)

    if "range_shuffle" in methods and random.random() < mutation_rate:
        if len(loco_ids) >= 2:
            num_select = min(len(loco_ids), max(2, random.randint(2, len(loco_ids))))
            idxs = sorted(random.sample(range(len(loco_ids)), num_select))
            if len(idxs) >= 2:
                a = idxs[0]
                b = idxs[-1]
                selected_locos = loco_ids[a:b+1]
                collected = []
                for lid in selected_locos:
                    collected.extend(assignment[lid])
                    assignment[lid] = []
                random.shuffle(collected)
                for i, t in enumerate(collected):
                    assignment[selected_locos[i % len(selected_locos)]].append(t)


# ---------------------
# Multiprocessing workers
# ---------------------
def _fitness_worker_serial(args):
    assignment, locomotives, train_lookup, station_coords, weights, means = args
    chrom = Chromosome(assignment)
    comps = compute_components(chrom, locomotives, train_lookup, station_coords)
    eps = 1e-6
    norm = [comps[i] / (means[i] + eps) for i in range(5)]
    penalty = sum(weights[i] * norm[i] for i in range(5))
    chrom.fitness = penalty
    return chrom.assignment, chrom.fitness


def _child_worker_serial(args):
    (assign_p1, assign_p2, locomotives, train_ids, train_lookup,
     mutation_rate, crossover_operator_probs, crossover_methods, crossover_weights, mutation_methods) = args

    if random.random() < 0.5:
        base = copy.deepcopy(assign_p1)
        other = assign_p2
    else:
        base = copy.deepcopy(assign_p2)
        other = assign_p1

    child_assign = base
    avg_weight = 1.0 / len(crossover_methods) if crossover_methods else 1.0

    for i, method in enumerate(crossover_methods):
        weight = crossover_weights[i] if (crossover_weights and i < len(crossover_weights)) else avg_weight
        base_prob = float(crossover_operator_probs.get(method, 0.5))
        scaling = max(0.0, 1.0 + (weight - avg_weight))
        prob = min(1.0, base_prob * scaling)
        if random.random() < prob:
            try:
                child_assign = crossover_assignments(child_assign, other, train_ids, train_lookup, method=method)
            except Exception:
                child_assign = crossover_assignments(assign_p1, assign_p2, train_ids, train_lookup, method=method)

    mutation_assignment(child_assign, mutation_rate=mutation_rate, methods=mutation_methods)
    child = Chromosome(child_assign)
    if is_feasible_fast(child, locomotives, train_lookup):
        return child_assign
    return None


# ---------------------
# Initial population generation
# ---------------------
def generate_initial_population(population_size: int, locomotives: Dict[int, Locomotive], 
                                trains: Dict[int, Train]) -> List[Chromosome]:
    """Generate initial population"""
    population = []
    train_ids = ordered_train_ids(trains)
    loco_ids = sorted(locomotives.keys())
    
    for _ in range(population_size):
        assignment = {lid: [] for lid in loco_ids}
        for tid in train_ids:
            assignment[random.choice(loco_ids)].append(tid)
        population.append(Chromosome(assignment))
    
    return population


# ---------------------
# GeneticAlgorithm with adaptive crossover selection
# ---------------------
class GeneticAlgorithm:
    def __init__(self, locomotives: Dict[int, Locomotive], trains: Dict[int, Train],
                 population_size: int = 50, generations: int = 100,
                 tournament_k: int = 3, mutation_rate: float = 0.1,
                 weights=(0.2, 0.2, 0.2, 0.2, 0.2),
                 station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                 progress_callback: Optional[callable] = None,
                 use_multiprocessing: bool = True,
                 multiprocessing_threshold: int = 100,
                 crossover_methods_allowed: Optional[List[str]] = None,
                 crossover_operator_probs: Optional[Dict[str, float]] = None,
                 mutation_methods: Optional[List[str]] = None,
                 adaptive_mutation: bool = True,
                 min_mutation_rate: float = 0.01,
                 max_mutation_rate: float = 0.5,
                 maximize: bool = False):
        self.locomotives = locomotives
        self.trains = trains
        self.train_lookup = build_train_lookup(trains)
        self.train_ids = ordered_train_ids(trains)
        self.population_size = max(1, int(population_size))
        self.generations = max(1, int(generations))
        self.tournament_k = max(1, min(int(tournament_k), self.population_size))
        self.initial_mutation_rate = float(mutation_rate)
        self.mutation_rate = float(mutation_rate)
        self.initial_weights = tuple(float(w) for w in weights)
        self.station_coords = station_coords
        self.progress_callback = progress_callback
        self.use_multiprocessing = bool(use_multiprocessing)
        self.multiprocessing_threshold = int(multiprocessing_threshold)
        self.cpu_count = max(1, multiprocessing.cpu_count())
        self.mutation_methods = mutation_methods or ["swap_locos", "replace_loco", "range_shuffle"]
        self.adaptive_mutation = adaptive_mutation
        self.min_mutation_rate = float(min_mutation_rate)
        self.max_mutation_rate = float(max_mutation_rate)

        self.crossover_operator_probs_input = crossover_operator_probs or {}
        self.best_history: List[float] = []
        self.no_improve_generations = 0

        all_ops = ["one_point", "two_point", "uniform", "priority"]
        if crossover_methods_allowed:
            self.crossover_methods = [m for m in all_ops if m in crossover_methods_allowed]
            if not self.crossover_methods:
                self.crossover_methods = all_ops[:]
        else:
            self.crossover_methods = all_ops[:]

        self.crossover_weights = [1.0 / len(self.crossover_methods) for _ in self.crossover_methods]
        self.crossover_operator_probs = {m: float(self.crossover_operator_probs_input.get(m, 0.5)) for m in self.crossover_methods}
        self.maximize = bool(maximize)

    def _tournament_selection(self, population: List[Chromosome], k: int = 3) -> Chromosome:
        k = min(k, len(population))
        candidates = random.sample(population, k)
        if self.maximize:
            return max(candidates, key=lambda c: c.fitness)
        else:
            return min(candidates, key=lambda c: c.fitness)

    def _best(self, population: List[Chromosome]) -> Chromosome:
        if self.maximize:
            return max(population, key=lambda c: c.fitness)
        else:
            return min(population, key=lambda c: c.fitness)

    def _sort_population(self, population: List[Chromosome]) -> List[Chromosome]:
        return sorted(population, key=lambda c: c.fitness, reverse=self.maximize)

    def _evaluate_population_dynamic(self, population: List[Chromosome]) -> Dict[str, Any]:
        comps = [compute_components(ch, self.locomotives, self.train_lookup, self.station_coords) for ch in population]
        if comps:
            means = [statistics.mean([c[i] for c in comps]) for i in range(5)]
        else:
            means = [0.0] * 5
        eps = 1e-6
        raw = [1.0 / (m + eps) for m in means]
        s = sum(raw)
        if s <= 0:
            weights = tuple(self.initial_weights[:5])
        else:
            weights = tuple(r / s for r in raw)

        if self.use_multiprocessing and len(population) >= self.multiprocessing_threshold and self.cpu_count > 1:
            args = [(ch.assignment, self.locomotives, self.train_lookup, self.station_coords, weights, means) for ch in population]
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(args))) as exc:
                for assignment, fitness in exc.map(_fitness_worker_serial, args):
                    for ch in population:
                        if ch.assignment == assignment:
                            ch.fitness = fitness
                            break
        else:
            for ch in population:
                comps_ch = compute_components(ch, self.locomotives, self.train_lookup, self.station_coords)
                norm = [comps_ch[i] / (means[i] + eps) for i in range(5)]
                penalty = sum(weights[i] * norm[i] for i in range(5))
                ch.fitness = penalty

        fitnesses = [ch.fitness for ch in population]
        if self.maximize:
            best = max(fitnesses) if fitnesses else float("-inf")
        else:
            best = min(fitnesses) if fitnesses else float("inf")
        mean_f = statistics.mean(fitnesses) if fitnesses else float("nan")
        std_f = statistics.pstdev(fitnesses) if fitnesses else float("nan")
        mn = min(fitnesses) if fitnesses else float("inf")
        mx = max(fitnesses) if fitnesses else float("-inf")

        stats = {
            "best": best,
            "mean": mean_f,
            "std": std_f,
            "min": mn,
            "max": mx,
            "weights": weights,
            "means_components": means
        }
        return stats

    def _adapt_mutation_rate(self, current_best: float):
        eps = 1e-8
        if not self.best_history:
            self.best_history.append(current_best)
            return
        prev_best = self.best_history[-1]
        improved = (current_best > prev_best + eps) if self.maximize else (current_best < prev_best - eps)
        if improved:
            self.no_improve_generations = 0
            self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.9)
        else:
            self.no_improve_generations += 1
            factor = 1.0 + 0.05 * min(self.no_improve_generations, 10)
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * factor)
        self.best_history.append(current_best)

    def _adapt_crossover_weights(self, current_best: float):
        exploitation = {"one_point", "priority"}
        exploration = {"uniform", "two_point"}
        idx_map = {m: i for i, m in enumerate(self.crossover_methods)}

        eps = 1e-8
        if not self.best_history:
            self.best_history.append(current_best)
            return
        prev_best = self.best_history[-1]
        improved = (current_best > prev_best + eps) if self.maximize else (current_best < prev_best - eps)

        step = 0.15
        min_w = 0.01
        for method in self.crossover_methods:
            i = idx_map[method]
            if improved:
                if method in exploitation:
                    self.crossover_weights[i] = self.crossover_weights[i] * (1.0 + step)
                else:
                    self.crossover_weights[i] = max(min_w, self.crossover_weights[i] * (1.0 - step))
            else:
                if method in exploration:
                    self.crossover_weights[i] = self.crossover_weights[i] * (1.0 + step)
                else:
                    self.crossover_weights[i] = max(min_w, self.crossover_weights[i] * (1.0 - step))
        tot = sum(self.crossover_weights)
        if tot <= 0:
            self.crossover_weights = [1.0 / len(self.crossover_methods) for _ in self.crossover_methods]
        else:
            self.crossover_weights = [w / tot for w in self.crossover_weights]

    def _generate_children(self, population: List[Chromosome], target_count: int) -> List[Chromosome]:
        children: List[Chromosome] = []
        curr_mutation_rate = self.mutation_rate if self.adaptive_mutation else self.initial_mutation_rate
        crossover_usage = {m: 0 for m in self.crossover_methods}

        if self.use_multiprocessing and len(population) >= self.multiprocessing_threshold and self.cpu_count > 1:
            parent_pairs = []
            for _ in range(target_count * 3):
                p1 = random.choice(population)
                p2 = random.choice(population)
                parent_pairs.append((
                    p1.assignment, p2.assignment, self.locomotives, self.train_ids, self.train_lookup,
                    curr_mutation_rate, self.crossover_operator_probs, self.crossover_methods,
                    self.crossover_weights, self.mutation_methods
                ))
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(parent_pairs))) as exc:
                for result in exc.map(_child_worker_serial, parent_pairs):
                    if result is not None:
                        children.append(Chromosome(result))
                        if self.crossover_methods:
                            crossover_usage[random.choice(self.crossover_methods)] += 1
                    if len(children) >= target_count:
                        break
        else:
            attempts = 0
            while len(children) < target_count and attempts < target_count * 50:
                attempts += 1
                p1 = self._tournament_selection(population, self.tournament_k)
                p2 = self._tournament_selection(population, self.tournament_k)

                if random.random() < 0.5:
                    base = copy.deepcopy(p1.assignment)
                    other = p2.assignment
                else:
                    base = copy.deepcopy(p2.assignment)
                    other = p1.assignment

                child_assign = base
                avg = 1.0 / len(self.crossover_methods) if self.crossover_methods else 1.0

                for i, method in enumerate(self.crossover_methods):
                    weight = self.crossover_weights[i] if i < len(self.crossover_weights) else avg
                    base_prob = float(self.crossover_operator_probs.get(method, 0.5))
                    scaling = max(0.0, 1.0 + (weight - avg))
                    prob = min(1.0, base_prob * scaling)
                    if random.random() < prob:
                        try:
                            child_assign = crossover_assignments(child_assign, other, self.train_ids, self.train_lookup, method=method)
                        except Exception:
                            child_assign = crossover_assignments(p1.assignment, p2.assignment, self.train_ids, self.train_lookup, method=method)
                        crossover_usage[method] += 1

                mutation_assignment(child_assign, mutation_rate=curr_mutation_rate, methods=self.mutation_methods)
                child = Chromosome(child_assign)
                if is_feasible_fast(child, self.locomotives, self.train_lookup):
                    children.append(child)

        total_used = sum(crossover_usage.values())
        if total_used > 0:
            usage_norm = {k: v / total_used for k, v in crossover_usage.items()}
        else:
            usage_norm = {k: 0.0 for k in crossover_usage.keys()}
        self.last_crossover_usage = usage_norm

        return children

    def run(self) -> Chromosome:
        population = generate_initial_population(self.population_size, self.locomotives, self.trains)
        generation_stats = []
        total_start = time.time()
        for gen in range(self.generations):
            gen_start = time.time()
            stats = self._evaluate_population_dynamic(population)
            gen_eval_time = time.time() - gen_start
            stats["time_sec"] = round(gen_eval_time, 4)
            stats["gen"] = gen

            best_value = stats.get("best")
            if best_value is None:
                stats["best_signed"] = None
                stats["best_interpretation"] = "n/a"
            else:
                if self.maximize:
                    stats["best_signed"] = f"+{best_value:.6f}"
                    stats["best_interpretation"] = "Максимизация — больше лучше (maximize)"
                else:
                    stats["best_signed"] = f"-{best_value:.6f}"
                    stats["best_interpretation"] = "Минимизация — меньше лучше (minimize)"

            generation_stats.append(stats)

            if self.adaptive_mutation:
                self._adapt_mutation_rate(stats["best"])

            self._adapt_crossover_weights(stats["best"])

            stats["crossover_methods"] = self.crossover_methods[:]
            stats["crossover_weights"] = tuple(self.crossover_weights)
            stats["crossover_operator_probs"] = {m: self.crossover_operator_probs.get(m, 0.5) for m in self.crossover_methods}
            stats["mutation_rate"] = round(self.mutation_rate, 4)

            if self.progress_callback:
                try:
                    self.progress_callback(gen, stats)
                except Exception:
                    try:
                        self.progress_callback(gen, stats["best"])
                    except Exception:
                        pass

            child_start = time.time()
            children = self._generate_children(population, self.population_size)
            child_time = time.time() - child_start
            generation_stats[-1]["children_time_sec"] = round(child_time, 4)

            if hasattr(self, "last_crossover_usage"):
                generation_stats[-1]["crossover_usage"] = self.last_crossover_usage

            new_population = children
            if len(new_population) < self.population_size:
                elites = sorted(population, key=lambda c: c.fitness, reverse=self.maximize)
                i = 0
                while len(new_population) < self.population_size:
                    e = elites[i % len(elites)]
                    new_population.append(Chromosome(copy.deepcopy(e.assignment)))
                    i += 1
            population = new_population

        final_start = time.time()
        final_stats = self._evaluate_population_dynamic(population)
        final_eval_time = time.time() - final_start
        final_stats["time_sec"] = round(final_eval_time, 4)
        final_stats["gen"] = self.generations
        final_best = final_stats.get("best")
        if final_best is None:
            final_stats["best_signed"] = None
            final_stats["best_interpretation"] = "n/a"
        else:
            if self.maximize:
                final_stats["best_signed"] = f"+{final_best:.6f}"
                final_stats["best_interpretation"] = "Максимизация — больше лучше (maximize)"
            else:
                final_stats["best_signed"] = f"-{final_best:.6f}"
                final_stats["best_interpretation"] = "Минимизация — меньше лучше (minimize)"

        generation_stats.append(final_stats)
        total_time = time.time() - total_start
        self.generation_stats = generation_stats
        self.total_time_sec = round(total_time, 4)
        if self.maximize:
            best = max(population, key=lambda c: c.fitness)
        else:
            best = min(population, key=lambda c: c.fitness)
        logger.info("GA finished: best fitness = %.6f; time s: %.4f", best.fitness, self.total_time_sec)
        return best


# ---------------------
# Reporter and plotting
# ---------------------
class GAReporter:
    def __init__(self):
        self.generation_log: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def log_generation(self, gen: int, stats: Dict[str, Any]):
        entry = {
            "поколение": gen,
            "best_raw": stats.get("best"),
            "best_signed": stats.get("best_signed"),
            "best_interpretation": stats.get("best_interpretation"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "weights": stats.get("weights"),
            "means_components": stats.get("means_components"),
            "time_sec": stats.get("time_sec"),
            "mutation_rate": stats.get("mutation_rate"),
            "crossover_methods": stats.get("crossover_methods"),
            "crossover_weights": stats.get("crossover_weights"),
            "crossover_operator_probs": stats.get("crossover_operator_probs"),
            "crossover_usage": stats.get("crossover_usage", {})
        }
        self.generation_log.append(entry)

    def elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def print_summary(self, solution: Chromosome, ga: Optional[GeneticAlgorithm] = None, total_time_sec: Optional[float] = None):
        direction = None
        if ga is not None:
            direction = "maximize" if getattr(ga, "maximize", False) else "minimize"
        print("\n=== Time and final fitness ===")
        print(f"Total computation time, s: {total_time_sec if total_time_sec is not None else self.elapsed():.2f}")
        raw = solution.fitness
        if direction == "maximize":
            print(f"Final objective (raw): {raw:.6f}  (interpretation: '+' bigger better)")
        elif direction == "minimize":
            print(f"Final objective (raw): {raw:.6f}  (interpretation: '-' smaller better)")
        else:
            print(f"Final objective (raw): {raw:.6f}")


def plot_generation_curve(reporter: GAReporter, figsize: Tuple[int, int] = (10, 4), dpi: int = 150) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if not reporter.generation_log:
        ax.set_title("Generation curve (no data)")
        return fig
    df = pd.DataFrame(reporter.generation_log)
    gens = df["поколение"].to_numpy()
    best = df["best_raw"].to_numpy()
    mean = df["mean"].to_numpy()
    std = df["std"].to_numpy()

    ax.plot(gens, best, label="best (raw)", color="#1f77b4", marker="o", linewidth=1.5)
    ax.plot(gens, mean, label="mean", color="#ff7f0e", marker="s", linewidth=1.2)
    ax.fill_between(gens, mean - std, mean + std, color="#ff7f0e", alpha=0.2, label="mean ± std")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Objective (raw)")
    ax.set_title("Evolution of objective")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    return fig


def plot_weights_evolution(reporter: GAReporter, figsize: Tuple[int, int] = (10, 3), dpi: int = 150) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if not reporter.generation_log:
        ax.set_title("Weights evolution (no data)")
        return fig
    ws = [e["weights"] for e in reporter.generation_log]
    gens = [e["поколение"] for e in reporter.generation_log]
    comp_arr = list(zip(*ws))
    labels = ["idle", "empty", "train_wait", "loco_wait", "used_locos"]
    colors = plt.get_cmap("tab10").colors
    for i, lab in enumerate(labels):
        series = comp_arr[i]
        ax.plot(gens, series, label=lab, marker="o", color=colors[i % len(colors)], linewidth=1.2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized weights")
    ax.set_title("Weights dynamics")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    return fig


def plot_crossover_weights_evolution(reporter: GAReporter, figsize: Tuple[int, int] = (10, 3), dpi: int = 150) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if not reporter.generation_log:
        ax.set_title("Crossover weights evolution (no data)")
        return fig
    methods_all = ["one_point", "two_point", "uniform", "priority"]
    gens = [e["поколение"] for e in reporter.generation_log]
    series = {m: [] for m in methods_all}
    for e in reporter.generation_log:
        cw = e.get("crossover_weights", None)
        cms = e.get("crossover_methods", None)
        if cw is not None and cms is not None:
            for m in methods_all:
                if m in cms:
                    idx = cms.index(m)
                    series[m].append(cw[idx])
                else:
                    series[m].append(0.0)
        else:
            for m in methods_all:
                series[m].append(0.0)
    colors = plt.get_cmap("tab10").colors
    for i, m in enumerate(methods_all):
        ax.plot(gens, series[m], label=m, marker="o", color=colors[i % len(colors)])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Adaptive weight")
    ax.set_title("Crossover operator weights evolution")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    return fig


def plot_components_evolution(reporter: GAReporter, figsize: Tuple[int, int] = (10, 4), dpi: int = 150) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if not reporter.generation_log:
        ax.set_title("Components evolution (no data)")
        return fig
    gens = [e["поколение"] for e in reporter.generation_log]
    means = [e.get("means_components", [0, 0, 0, 0, 0]) for e in reporter.generation_log]
    comp_arr = list(zip(*means))
    labels = ["Idle h", "Empty h", "Train wait h", "Loco wait h", "Used locos"]
    colors = plt.get_cmap("tab10").colors
    for i, lab in enumerate(labels):
        ax.plot(gens, comp_arr[i], label=lab, marker="o", color=colors[i % len(colors)])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean component values")
    ax.set_title("Components dynamics")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    return fig


def plot_assignment_matplotlib(solution: Chromosome, trains: Dict[int, Train],
                               figsize: Tuple[int, int] = (12, 6), dpi: int = 150) -> plt.Figure:
    loco_items = sorted(solution.assignment.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n_locos = len(loco_items)
    fig, ax = plt.subplots(figsize=(figsize[0], max(4, n_locos * 0.35)), dpi=dpi)
    y = 0
    ylabels = []
    cmap = plt.get_cmap("tab20")
    for loco_id, train_ids in loco_items:
        for t_id in sorted(train_ids, key=lambda tid: trains.get(tid, Train(0, 0, ("", ""), 0, 0)).departure_time):
            t = trains.get(t_id)
            if t is None:
                continue
            ax.barh(y, t.duration, left=t.departure_time, height=0.6, color=cmap(t_id % 20), edgecolor="k", alpha=0.85)
            ax.text(t.departure_time + t.duration / 2, y, f"{t.id} (cat{t.category})", va="center", ha="center", color="white", fontsize=7)
        ylabels.append(f"{loco_id}")
        y += 1
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time, h")
    ax.set_ylabel("Locomotives (id)")
    ax.set_title("Assignments timeline")
    ax.grid(True, linestyle="--", alpha=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i % 20), edgecolor="k") for i in range(min(6, max(1, len(trains))))]
    labels = [f"train {i}" for i in range(min(6, max(1, len(trains))))]
    ax.legend(handles, labels, loc="upper right", title="Example trains", framealpha=0.9)
    return fig


# ---------------------
# DataFrame builders
# ---------------------
def build_assignment_dataframe(solution: Chromosome, locomotives: Dict[int, Locomotive], 
                               trains: Dict[int, Train]) -> pd.DataFrame:
    """Build assignment dataframe"""
    data = []
    for loco_id, train_ids in solution.assignment.items():
        loco = locomotives.get(loco_id)
        if loco is None:
            continue
        for t_id in train_ids:
            t = trains.get(t_id)
            if t is None:
                continue
            data.append({
                "Locomotive_ID": loco_id,
                "Loco_Type": loco.loco_type,
                "Train_ID": t_id,
                "Train_Weight": t.weight,
                "Route": f"{t.route[0]}-{t.route[1]}",
                "Departure": t.departure_time,
                "Duration": t.duration,
                "Category": t.category
            })
    return pd.DataFrame(data)


def build_locomotive_summary_dataframe(locomotives: Dict[int, Locomotive], 
                                       trains: Dict[int, Train],
                                       solution: Chromosome) -> pd.DataFrame:
    """Build locomotive summary dataframe"""
    data = []
    for loco_id, loco in locomotives.items():
        train_ids = solution.assignment.get(loco_id, [])
        total_weight = sum(trains.get(t_id, Train(0, 0, ("", ""), 0, 0)).weight for t_id in train_ids)
        total_duration = sum(trains.get(t_id, Train(0, 0, ("", ""), 0, 0)).duration for t_id in train_ids)
        
        data.append({
            "Locomotive_ID": loco_id,
            "Type": loco.loco_type,
            "Power": loco.power,
            "Resource": loco.remaining_resource,
            "Depot": loco.home_depot,
            "Assigned_Trains": len(train_ids),
            "Total_Weight": total_weight,
            "Total_Duration": total_duration,
            "Utilization_%": round(100 * total_duration / max(1, loco.remaining_resource), 2)
        })
    return pd.DataFrame(data)


# ---------------------
# Streamlit helpers
# ---------------------
def st_download_button_from_fig(fig: plt.Figure, label: str, filename: str = "figure.png", fmt: str = "png", dpi: int = 300):
    data = fig_to_bytes(fig, fmt=fmt, dpi=dpi)
    href = download_link_bytes(data, filename, mime=f"image/{fmt}")
    st.markdown(f"[Download {label}]({href})")


def st_download_button_from_df(df: pd.DataFrame, label: str, filename: str = "table.csv"):
    data = df_to_csv_bytes(df)
    href = download_link_bytes(data, filename, mime="text/csv")
    st.markdown(f"[Download {label}]({href})")


# ---------------------
# Streamlit UI
# ---------------------
def parse_locomotives_csv(file_buf: io.BytesIO) -> Dict[int, Locomotive]:
    try:
        df = pd.read_csv(file_buf)
    except Exception:
        df = pd.read_excel(file_buf)
    locomotives: Dict[int, Locomotive] = {}
    for _, row in df.iterrows():
        try:
            lid = int(row.get("id", row.get("ID", None)))
        except Exception:
            continue
        loco_type = row.get("loco_type", row.get("type", "unknown"))
        power = float(row.get("power", row.get("Power", 4000)))
        remaining = float(row.get("remaining_resource", row.get("resource", 40)))
        depot = row.get("home_depot", row.get("depot", "A"))
        speed = float(row.get("reposition_speed_kmh", row.get("speed_kmh", 60.0)))
        locomotives[lid] = Locomotive(id=lid, loco_type=str(loco_type), power=power,
                                      remaining_resource=remaining, home_depot=str(depot),
                                      reposition_speed_kmh=float(speed))
    return locomotives


def parse_trains_csv(file_buf: io.BytesIO) -> Dict[int, Train]:
    try:
        df = pd.read_csv(file_buf)
    except Exception:
        df = pd.read_excel(file_buf)
    trains: Dict[int, Train] = {}
    for _, row in df.iterrows():
        try:
            tid = int(row.get("id", row.get("ID", None)))
        except Exception:
            continue
        weight = float(row.get("weight", row.get("Weight", 4000)))
        if "route_from" in row.index or "route_to" in row.index:
            dep = row.get("route_from", row.get("from", None)) or row.get("dep", "A")
            arr = row.get("route_to", row.get("to", None)) or row.get("arr", "B")
        else:
            route_field = row.get("route", None)
            if isinstance(route_field, str) and "-" in route_field:
                dep, arr = route_field.split("-", 1)
                dep = dep.strip()
                arr = arr.strip()
            else:
                dep = row.get("dep", "A")
                arr = row.get("arr", "B")
        departure_time = float(row.get("departure_time", row.get("dep_time", 0.0)))
        duration = float(row.get("duration", row.get("dur", 3.0)))
        category = int(row.get("category", row.get("cat", 0)))
        trains[tid] = Train(id=tid, weight=weight, route=(dep, arr), departure_time=departure_time, duration=duration, category=category)
    return trains


def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Install streamlit to run the UI.")
        return

    st.set_page_config(page_title="Loco_GA", layout="wide")
    st.title("Loco_GA — Locomotive Assignment (Genetic Algorithm)")
    st.sidebar.header("Parameters")

    loco_file = st.sidebar.file_uploader("CSV/Excel: Locomotives", type=["csv", "xls", "xlsx"])
    trains_file = st.sidebar.file_uploader("CSV/Excel: Trains", type=["csv", "xls", "xlsx"])
    use_uploaded = st.sidebar.checkbox("Use uploaded files", value=True)

    num_locomotives = st.sidebar.slider("Number of locomotives (if not uploaded)", 1, 200, 10)
    num_trains = st.sidebar.slider("Number of trains (if not uploaded)", 1, 500, 20)
    depot_names_str = st.sidebar.text_input("Stations/depots (comma separated)", "A,B,C")
    depot_names = tuple(s.strip() for s in depot_names_str.split(",") if s.strip())
    seed = st.sidebar.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
    if seed == 0:
        seed = None

    population_size = st.sidebar.number_input("Population size", min_value=2, max_value=2000, value=60)
    generations = st.sidebar.number_input("Generations", min_value=1, max_value=2000, value=60)
    tournament_k = st.sidebar.number_input("Tournament k", min_value=1, max_value=population_size, value=5)
    mutation_rate = st.sidebar.slider("Initial mutation rate", 0.0, 1.0, 0.15, 0.01)

    st.sidebar.markdown("### Crossover operators")
    use_all_cross = st.sidebar.checkbox("Use all crossover operators", value=True)
    all_ops = ["one_point", "two_point", "uniform", "priority"]
    if use_all_cross:
        crossover_methods_selected = all_ops[:]
    else:
        crossover_methods_selected = st.sidebar.multiselect("Select crossover operators", options=all_ops, default=["uniform", "one_point"])
        if not crossover_methods_selected:
            st.sidebar.warning("No crossover selected — defaulting to 'uniform'")
            crossover_methods_selected = ["uniform"]

    st.sidebar.markdown("Set per-operator base probabilities")
    crossover_operator_probs = {}
    for op in all_ops:
        if op in crossover_methods_selected:
            default = 0.5
            val = st.sidebar.slider(f"p({op})", 0.0, 1.0, float(default), 0.01)
            crossover_operator_probs[op] = float(val)

    
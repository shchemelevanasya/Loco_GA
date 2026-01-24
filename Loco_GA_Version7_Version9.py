# -*- coding: utf-8 -*-
"""
Loco_GA — Genetic Algorithm for locomotive assignment

Версия:
 - Мутации: swap_locos, replace_loco, range_shuffle
 - Кроссоверы: one_point, two_point, uniform, priority
 - Адаптивная вероятность мутации
 - Адаптивный выбор операторов кроссовера (внутри выбранного набора), возможность выбирать операторы и задавать
   индивидуальную базовую вероятность их применения
 - Streamlit UI: загрузка данных, выбор операторов и их вероятностей, выбор Minimize/Maximize
 - fitness хранится как raw objective (penalty). По умолчанию - минимизация (меньше лучше).
 - В отчётах добавлено поле best_signed и пояснение (интерпретация sign/interpretation).
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
import io

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
    reposition_speed_kmh: float  # скорость репозиции (для расчёта времени пустого хода)


@dataclass
class Train:
    id: int
    weight: float
    route: Tuple[str, str]
    departure_time: float
    duration: float
    category: int = 0  # optional priority category (0 = low); используется в priority crossover


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
    """
    Перевод расстояния в часы с учётом средней скорости и штрафа за уклон.
    """
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be > 0")
    if distance <= 0:
        return 0.0
    slope = abs(slope_elevation_diff) / max(distance, 1e-6)
    penalty = 1.0 + slope_penalty_coefficient * slope
    hours = (distance / speed_kmh) * penalty
    return hours


# ---------------------
# Synthetic data with multiple locomotive types
# ---------------------
def generate_synthetic_data(
        num_locomotives: int = 10,
        num_trains: int = 20,
        depot_names: Tuple[str, ...] = ("A", "B", "C"),
        station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
        loco_types: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None):
    """
    Генерация синтетических данных.
    """
    if seed is not None:
        random.seed(seed)

    # default loco types
    if loco_types is None:
        loco_types = {
            "2ЭС6": {"power_range": (4500, 6500), "speed_kmh": 60.0, "resource_range": (20, 50)},
            "ЧС7": {"power_range": (3000, 5000), "speed_kmh": 50.0, "resource_range": (15, 40)},
            "ТЭП70": {"power_range": (5000, 7500), "speed_kmh": 70.0, "resource_range": (25, 60)}
        }

    # station coords if not provided
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
    # create categories for priorities (0=low,1=medium,2=high)
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

    logger.info("Сгенерированы данные: %d локомотивов, %d поездов, %d станций, %d типов локомотивов",
                len(locomotives), len(trains), len(station_coords), len(loco_types))
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
# Feasibility check (fast)
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
    """
    Быстрая проверка: тяга и ресурс по сумме длительностей.
    """
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
# Advanced timeline metrics
# ---------------------
def compute_time_components_for_chromosome(chrom: Chromosome,
                                           locomotives: Dict[int, Locomotive],
                                           trains: Dict[int, Train],
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
        sorted_trains = sorted((trains[t_id] for t_id in train_ids), key=lambda t: t.departure_time)
        loco = locomotives[loco_id]
        loco_available_time = 0.0
        current_loc = loco.home_depot
        for t in sorted_trains:
            dep_station = t.route[0]
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
            if loco_arrival > t.departure_time:
                wait_t = loco_arrival - t.departure_time
                train_wait_time_sum += wait_t
            else:
                wait_l = t.departure_time - loco_arrival
                loco_wait_time_sum += wait_l
            empty_time_sum += reposition_time
            actual_departure = max(t.departure_time, loco_arrival)
            loco_available_time = actual_departure + t.duration
            current_loc = t.route[1]
        idle_time_sum += loco_wait_time_sum

    return {
        "idle_time_h": idle_time_sum,
        "empty_time_h": empty_time_sum,
        "train_wait_time_h": train_wait_time_sum,
        "loco_wait_time_h": loco_wait_time_sum,
        "used_locos_count": used_locos
    }


# ---------------------
# Components & dynamic weights
# ---------------------
def compute_components(chromosome: Chromosome,
                       locomotives: Dict[int, Locomotive],
                       trains: Dict[int, Train],
                       station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None) -> Tuple[float, float, float, float, float]:
    if station_coords is None:
        idle_count = sum(1 for lst in chromosome.assignment.values() if not lst)
        empty = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values())
        train_wait = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values()) * 0.5
        loco_wait = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values()) * 0.5
        used_locos = sum(1 for lst in chromosome.assignment.values() if lst)
        return float(idle_count), float(empty), float(train_wait), float(loco_wait), float(used_locos)
    metrics = compute_time_components_for_chromosome(chromosome, locomotives, trains, station_coords)
    return (metrics["idle_time_h"], metrics["empty_time_h"], metrics["train_wait_time_h"],
            metrics["loco_wait_time_h"], float(metrics["used_locos_count"]))


def derive_dynamic_weights(population: List[Chromosome],
                           locomotives: Dict[int, Locomotive],
                           trains: Dict[int, Train],
                           station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None) -> Tuple[float, float, float, float, float]:
    eps = 1e-6
    comps = [compute_components(chrom, locomotives, trains, station_coords) for chrom in population]
    if not comps:
        return (0.2, 0.2, 0.2, 0.2, 0.2)
    means = [statistics.mean([c[i] for c in comps]) for i in range(5)]
    raw = [1.0 / (m + eps) for m in means]
    s = sum(raw)
    weights = tuple(r / s for r in raw)
    return weights


def fitness_function_components_based(chromosome: Chromosome,
                                      locomotives: Dict[int, Locomotive],
                                      trains: Dict[int, Train],
                                      station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                                      weights: Optional[Tuple[float, float, float, float, float]] = None,
                                      eps: float = 1e-6) -> float:
    """
    Возвращает raw objective (penalty). Меньше — лучше (по умолчанию).
    """
    comps = compute_components(chromosome, locomotives, trains, station_coords)
    if weights is None:
        weights = (0.2, 0.2, 0.2, 0.2, 0.2)
    norm = [comps[i] / (1.0 + comps[i]) for i in range(5)]
    penalty = sum(weights[i] * norm[i] for i in range(5))
    chromosome.fitness = penalty
    return penalty


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
        logger.warning("Начальная допустимая популяция не найдена случайно — использован fallback")
    return population


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
def tournament_selection(population: List[Chromosome], k: int = 3) -> Chromosome:
    if not population:
        raise ValueError("Популяция пуста – нельзя провести отбор")
    k = min(k, len(population))
    candidates = random.sample(population, k)
    return max(candidates, key=lambda c: c.fitness)


def crossover_assignments(assign1: Dict[int, List[int]],
                          assign2: Dict[int, List[int]],
                          trains: Dict[int, Train],
                          method: str = "uniform") -> Dict[int, List[int]]:
    train_ids = ordered_train_ids(trains)
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
        cats = [trains[tid].category if tid in trains else 0 for tid in train_ids]
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
                        trains: Dict[int, Train],
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
            idxs = sorted(random.sample(range(len(loco_ids)), min(len(loco_ids), max(2, random.randint(2, len(loco_ids)))) ))
            a = idxs[0]
            b = idxs[-1]
            selected_locos = loco_ids[a:b+1]
            collected = []
            for lid in selected_locos:
                collected.extend(assignment[lid])
                assignment[lid] = []
            random.shuffle(collected)
            i = 0
            for t in collected:
                assignment[selected_locos[i % len(selected_locos)]].append(t)
                i += 1


# ---------------------
# Multiprocessing workers
# ---------------------
def _fitness_worker_serial(args):
    assignment, locomotives, trains, station_coords, weights, means = args
    chrom = Chromosome(assignment)
    comps = compute_components(chrom, locomotives, trains, station_coords)
    eps = 1e-6
    norm = [comps[i] / (means[i] + eps) for i in range(5)]
    penalty = sum(weights[i] * norm[i] for i in range(5))
    chrom.fitness = penalty
    return chrom.assignment, chrom.fitness


def _child_worker_serial(args):
    (assign_p1, assign_p2, locomotives, trains,
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
                child_assign = crossover_assignments(child_assign, other, trains, method=method)
            except Exception:
                child_assign = crossover_assignments(assign_p1, assign_p2, trains, method=method)

    mutation_assignment(child_assign, trains, mutation_rate=mutation_rate, methods=mutation_methods)
    child = Chromosome(child_assign)
    if is_feasible_fast(child, locomotives, trains):
        return child_assign
    return None


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

    # selection helpers respecting maximize flag
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
        comps = [compute_components(ch, self.locomotives, self.trains, self.station_coords) for ch in population]
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
            args = [(ch.assignment, self.locomotives, self.trains, self.station_coords, weights, means) for ch in population]
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(args))) as exc:
                for assignment, fitness in exc.map(_fitness_worker_serial, args):
                    for ch in population:
                        if ch.assignment == assignment:
                            ch.fitness = fitness
                            break
        else:
            for ch in population:
                comps_ch = compute_components(ch, self.locomotives, self.trains, self.station_coords)
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
                    p1.assignment, p2.assignment, self.locomotives, self.trains,
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
                            child_assign = crossover_assignments(child_assign, other, self.trains, method=method)
                        except Exception:
                            child_assign = crossover_assignments(p1.assignment, p2.assignment, self.trains, method=method)
                        crossover_usage[method] += 1

                mutation_assignment(child_assign, self.trains, mutation_rate=curr_mutation_rate, methods=self.mutation_methods)
                child = Chromosome(child_assign)
                if is_feasible_fast(child, self.locomotives, self.trains):
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
        logger.info("GA завершён: лучшая пригодность = %.6f; время, с: %.4f", best.fitness, self.total_time_sec)
        return best


# ---------------------
# Reporting & DataFrames
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
            "crossover_usage": stats.get("crossover_usage", {}),
            "crossover_apply_rate": stats.get("crossover_apply_rate")
        }
        self.generation_log.append(entry)

    def elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def print_summary(self, solution: Chromosome, ga: Optional[GeneticAlgorithm] = None, total_time_sec: Optional[float] = None):
        direction = None
        if ga is not None:
            direction = "maximize" if getattr(ga, "maximize", False) else "minimize"

        print("\n=== Время расчёта и итоговая пригодность ===")
        print(f"Время расчёта, с: {total_time_sec if total_time_sec is not None else self.elapsed():.2f}")

        raw = solution.fitness
        if direction == "maximize":
            print(f"Итоговая целевая функция (raw): {raw:.6f}  (интерпретация: '+' — больше лучше)")
            print(f"Интерпретация: лучшее значение считается большим (maximize).")
        elif direction == "minimize":
            print(f"Итоговая целевая функция (raw): {raw:.6f}  (интерпретация: '-' — меньше лучше)")
            print(f"Интерпретация: лучшее значение считается меньшим (minimize).")
        else:
            print(f"Итоговая целевая функция (raw): {raw:.6f}")
            print("Интерпретация: direction not specified (use GeneticAlgorithm(maximize=True/False)).")


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
                "Локомотив": loco_id,
                "Тип": loco.loco_type if loco else None,
                "Депо": depot,
                "Остаток_ресурса_ч": remaining,
                "Поезд": None,
                "Откуда": None,
                "Куда": None,
                "Отпр (ч)": None,
                "Приб (ч)": None,
                "Масса_поезда": None,
                "Длительность_рейса_ч": None
            })
        else:
            for t_id in train_ids:
                t = trains[t_id]
                rows.append({
                    "Локомотив": loco_id,
                    "Тип": loco.loco_type if loco else None,
                    "Депо": depot,
                    "Остаток_ресурса_ч": remaining,
                    "Поезд": t.id,
                    "Откуда": t.route[0],
                    "Куда": t.route[1],
                    "Отпр (ч)": round(t.departure_time, 2),
                    "Приб (ч)": round(t.departure_time + t.duration, 2),
                    "Масса_поезда": round(t.weight, 1),
                    "Длительность_рейса_ч": round(t.duration, 2)
                })
    df = pd.DataFrame(rows, columns=[
        "Локомотив", "Тип", "Депо", "Остаток_ресурса_ч", "Поезд", "Откуда", "Куда",
        "Отпр (ч)", "Приб (ч)", "Масса_поезда", "Длительность_рейса_ч"
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
            "Локомотив": loco_id,
            "Тип": loco.loco_type,
            "Депо": loco.home_depot,
            "Текущее_место": current_location,
            "Остаток_ресурса_ч": round(remaining_resource, 2),
            "Назначено_поездов": len(train_ids)
        })
    df = pd.DataFrame(rows, columns=["Локомотив", "Тип", "Депо", "Текущее_место", "Остаток_ресурса_ч", "Назначено_поездов"])
    return df


# ---------------------
# Plotting helpers
# ---------------------
def plot_generation_curve(reporter: GAReporter):
    if not reporter.generation_log:
        fig, ax = plt.subplots()
        ax.set_title("Кривая прогрессии поколений (нет данных)")
        return fig
    df = pd.DataFrame(reporter.generation_log)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["поколение"], df["best_raw"], label="Лучшее (raw)", marker="o")
    ax.plot(df["поколение"], df["mean"], label="Среднее (mean)", marker="o")
    ax.plot(df["поколение"], df["min"], label="Минимум (min)", marker="o")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Значение целевой функции (raw)")
    ax.set_title("Эволюция целевой функции по поколениям")
    ax.legend(loc="best")
    ax.grid(True)
    if "time_sec" in df.columns:
        ax2 = ax.twinx()
        ax2.bar(df["поколение"], df["time_sec"], color="gray", alpha=0.25, label="Время оценки (с)")
        ax2.set_ylabel("Время на поколение, с")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
    return fig


def plot_weights_evolution(reporter: GAReporter):
    if not reporter.generation_log:
        fig, ax = plt.subplots()
        ax.set_title("Эволюция весов (нет данных)")
        return fig
    gens = [e["поколение"] for e in reporter.generation_log]
    ws = [e["weights"] for e in reporter.generation_log]
    w_idle = [w[0] for w in ws]
    w_empty = [w[1] for w in ws]
    w_train_wait = [w[2] for w in ws]
    w_loco_wait = [w[3] for w in ws]
    w_used = [w[4] for w in ws]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(gens, w_idle, label="Вес простоя (idle)", marker="o")
    ax.plot(gens, w_empty, label="Вес порожнего пробега (empty)", marker="o")
    ax.plot(gens, w_train_wait, label="Вес ожидания поездов (train_wait)", marker="o")
    ax.plot(gens, w_loco_wait, label="Вес ожидания локомотивов (loco_wait)", marker="o")
    ax.plot(gens, w_used, label="Вес числа локомотивов (used_locos)", marker="o")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Вес критерия")
    ax.set_title("Динамика весов критериев по поколениям")
    ax.legend(loc="best")
    ax.grid(True)
    return fig


def plot_crossover_weights_evolution(reporter: GAReporter):
    if not reporter.generation_log:
        fig, ax = plt.subplots()
        ax.set_title("Динамика весов операторов кроссовера (нет данных)")
        return fig
    gens = [e["поколение"] for e in reporter.generation_log]
    methods_all = ["one_point", "two_point", "uniform", "priority"]
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
    fig, ax = plt.subplots(figsize=(9, 3))
    for m in methods_all:
        ax.plot(gens, series[m], label=m, marker="o")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Вес оператора")
    ax.set_title("Эволюция весов операторов кроссовера")
    ax.legend(loc="best")
    ax.grid(True)
    return fig


def plot_components_evolution(reporter: GAReporter):
    if not reporter.generation_log:
        fig, ax = plt.subplots()
        ax.set_title("Динамика компонент (нет данных)")
        return fig
    gens = [e["поколение"] for e in reporter.generation_log]
    means = [e.get("means_components", [0, 0, 0, 0, 0]) for e in reporter.generation_log]
    comp_arr = list(zip(*means))  # 5 lists
    labels = ["Простой, ч", "Порожние, ч", "Ожидание поездов, ч", "Ожидание локомотивов, ч", "Используемых локомотивов"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, comp in enumerate(comp_arr):
        ax.plot(gens, comp, label=labels[i], marker="o")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Средние значения компонентов")
    ax.set_title("Динамика компонентов цели по поколениям")
    ax.legend(loc="best")
    ax.grid(True)
    return fig


def plot_assignment_matplotlib(solution: Chromosome, trains: Dict[int, Train]):
    fig, ax = plt.subplots(figsize=(10, max(4, len(solution.assignment) * 0.3)))
    y = 0
    ylabels = []
    for loco_id, train_ids in solution.assignment.items():
        for t_id in train_ids:
            t = trains[t_id]
            ax.barh(y, t.duration, left=t.departure_time, height=0.4, label=f"Поезд {t.id}" if y == 0 else "")
        ylabels.append(str(loco_id))
        y += 1
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Локомотивы (id)")
    ax.set_title("График назначения: временная диаграмма")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    return fig


# ---------------------
# Sensitivity analysis
# ---------------------
def sensitivity_analysis(locomotives: Dict[int, Locomotive],
                         trains: Dict[int, Train],
                         station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                         base_weights: Tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2),
                         deltas: List[float] = (-0.2, -0.1, 0.0, 0.1, 0.2),
                         population_size: int = 40,
                         generations: int = 20,
                         tournament_k: int = 3,
                         mutation_rate: float = 0.1,
                         use_multiprocessing: bool = False) -> pd.DataFrame:
    rows = []
    for d in deltas:
        w0 = base_weights[0] + d
        w1 = base_weights[1] + d
        w2 = base_weights[2] - d
        w3 = base_weights[3] - d
        w4 = base_weights[4]
        ws_raw = [max(0.0, w0), max(0.0, w1), max(0.0, w2), max(0.0, w3), max(0.0, w4)]
        s = sum(ws_raw)
        if s <= 0:
            ws = base_weights
        else:
            ws = tuple(x / s for x in ws_raw)

        ga = GeneticAlgorithm(locomotives, trains,
                              population_size=population_size,
                              generations=generations,
                              tournament_k=tournament_k,
                              mutation_rate=mutation_rate,
                              weights=ws,
                              station_coords=station_coords,
                              use_multiprocessing=use_multiprocessing)
        start = time.time()
        try:
            best = ga.run()
            elapsed = time.time() - start
            fitness_val = best.fitness
        except Exception as e:
            logger.exception("Ошибка в sensitivity run: %s", e)
            elapsed = float("nan")
            fitness_val = float("nan")
        rows.append({
            "Δ": d,
            "w_idle": round(ws[0], 3),
            "w_empty": round(ws[1], 3),
            "w_train_wait": round(ws[2], 3),
            "w_loco_wait": round(ws[3], 3),
            "w_used_locos": round(ws[4], 3),
            "fitness": fitness_val,
            "time_sec": round(elapsed, 4)
        })
    return pd.DataFrame(rows)


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
        print("Streamlit не установлен в окружении. Установите streamlit для запуска UI.")
        return

    st.set_page_config(page_title="Loco_GA", layout="wide")
    st.title("Loco_GA — Назначение локомотивов (Генетический алгоритм)")
    st.sidebar.header("Параметры")

    loco_file = st.sidebar.file_uploader("CSV/Excel: Локомотивы", type=["csv", "xls", "xlsx"])
    trains_file = st.sidebar.file_uploader("CSV/Excel: Поезда", type=["csv", "xls", "xlsx"])
    use_uploaded = st.sidebar.checkbox("Использовать загруженные данные если есть", value=True)

    num_locomotives = st.sidebar.slider("Число локомотивов (если не загружено)", 1, 200, 10)
    num_trains = st.sidebar.slider("Число поездов (если не загружено)", 1, 500, 20)
    depot_names_str = st.sidebar.text_input("Станции/депо (через запятую)", "A,B,C")
    depot_names = tuple(s.strip() for s in depot_names_str.split(",") if s.strip())
    seed = st.sidebar.number_input("Random seed (0 = произвольно)", min_value=0, value=0, step=1)
    if seed == 0:
        seed = None

    population_size = st.sidebar.number_input("Размер популяции", min_value=2, max_value=2000, value=60)
    generations = st.sidebar.number_input("Поколений", min_value=1, max_value=2000, value=60)
    tournament_k = st.sidebar.number_input("Размер турнира", min_value=1, max_value=population_size, value=5)
    mutation_rate = st.sidebar.slider("Вероятность мутации (начальная)", 0.0, 1.0, 0.15, 0.01)

    st.sidebar.markdown("### Операторы кроссовера")
    use_all_cross = st.sidebar.checkbox("Использовать все операторы кроссовера", value=True)
    all_ops = ["one_point", "two_point", "uniform", "priority"]
    if use_all_cross:
        crossover_methods_selected = all_ops[:]
    else:
        crossover_methods_selected = st.sidebar.multiselect("Выберите операторы кроссовера (подмножество)",
                                                            options=all_ops,
                                                            default=["uniform", "one_point"])
        if not crossover_methods_selected:
            st.sidebar.warning("Не выбрано ни одного оператора — будет использован 'uniform'")
            crossover_methods_selected = ["uniform"]

    st.sidebar.markdown("Задайте вероятность применения для каждого выбранного оператора кроссовера")
    crossover_operator_probs = {}
    for op in all_ops:
        if op in crossover_methods_selected:
            default = 0.5
            val = st.sidebar.slider(f"p({op})", 0.0, 1.0, float(default), 0.01)
            crossover_operator_probs[op] = float(val)

    st.sidebar.markdown("Мутации (выбрать применимые):")
    m_swap = st.sidebar.checkbox("swap_locos (обмен назначениями между локомотивами)", value=True)
    m_replace = st.sidebar.checkbox("replace_loco (перенос отдельного поезда на другой локомотив)", value=True)
    m_range = st.sidebar.checkbox("range_shuffle (перемешивание по диапазону локомотивов)", value=True)
    mutation_methods = []
    if m_swap:
        mutation_methods.append("swap_locos")
    if m_replace:
        mutation_methods.append("replace_loco")
    if m_range:
        mutation_methods.append("range_shuffle")
    adaptive_mutation = st.sidebar.checkbox("Адаптивная вероятность мутации (увеличивается при стагнации)", value=True)
    min_mut = st.sidebar.slider("Мин вероятность мутации (adaptive)", 0.0, 1.0, 0.01, 0.01)
    max_mut = st.sidebar.slider("Макс вероятность мутации (adaptive)", 0.0, 1.0, 0.5, 0.01)

    st.sidebar.markdown("### Направление оптимизации")
    maximize = st.sidebar.checkbox("Maximize (если снято — Minimize по умолчанию)", value=False)

    st.sidebar.markdown("### Типы локомотивов (пример)")
    st.sidebar.markdown("По умолчанию используются 3 типа: 2ЭС6, ЧС7, ТЭП70.")
    use_mp = st.sidebar.checkbox("Использовать multiprocessing для тяжёлых задач", value=True)
    mp_threshold = st.sidebar.number_input("Порог для multiprocessing (размер популяции)", min_value=2, max_value=500, value=100)

    st.sidebar.markdown("---")
    run_sens_button = st.sidebar.button("Запустить анализ чувствительности (быстро)")

    if st.sidebar.button("Сгенерировать данные"):
        with st.spinner("Генерация синтетических данных..."):
            locomotives, trains, station_coords = generate_synthetic_data(
                num_locomotives=num_locomotives,
                num_trains=num_trains,
                depot_names=depot_names,
                seed=seed
            )
            st.session_state["locomotives"] = locomotives
            st.session_state["trains"] = trains
            st.session_state["station_coords"] = station_coords
        st.success("Данные сгенерированы")

    if use_uploaded and loco_file is not None:
        try:
            locomotives = parse_locomotives_csv(loco_file)
            st.session_state["locomotives"] = locomotives
            st.success("Локомотивы загружены")
        except Exception as e:
            st.error(f"Не удалось распарсить файл локомотивов: {e}")

    if use_uploaded and trains_file is not None:
        try:
            trains = parse_trains_csv(trains_file)
            st.session_state["trains"] = trains
            st.success("Поезда загружены")
        except Exception as e:
            st.error(f"Не удалось распарсить файл поездов: {e}")

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

    left, right = st.columns((2, 1))
    with right:
        st.markdown("### Статистика по поколению")
        gen_stats_placeholder = st.empty()
        st.markdown("### Динамика весов")
        weights_plot_placeholder = st.empty()
        st.markdown("### Веса операторов кроссовера")
        crossover_weights_placeholder = st.empty()
        st.markdown("### Время и итог")
        time_summary_placeholder = st.empty()

    with left:
        st.markdown("### Таблица назначений")
        assignment_placeholder = st.empty()
        st.markdown("### Сводная таблица локомотивов")
        loco_summary_placeholder = st.empty()
        st.markdown("### График назначений (временная диаграмма)")
        timeline_placeholder = st.empty()
        st.markdown("### Кривая эволюции (fitness по поколениям)")
        evolution_placeholder = st.empty()
        st.markdown("### Динамика компонентов")
        components_placeholder = st.empty()
        st.markdown("### Результаты анализа чувствительности")
        sens_placeholder = st.empty()

    reporter = GAReporter()
    reporter.start()

    run_button = st.button("Запустить GA (синхронно)")

    if run_button:
        def progress_callback(gen: int, stats: Dict[str, Any]):
            best_signed = stats.get("best_signed", "")
            mean = stats.get("mean", 0.0)
            gen_stats_placeholder.metric(label=f"Поколение {gen}", value=f"Лучшее: {best_signed}", delta=f"Среднее: {mean:.6f}")
            reporter.log_generation(gen, stats)

        try:
            ga = GeneticAlgorithm(
                locomotives, trains,
                population_size=int(population_size),
                generations=int(generations),
                tournament_k=int(tournament_k),
                mutation_rate=float(mutation_rate),
                weights=(0.2, 0.2, 0.2, 0.2, 0.2),
                station_coords=station_coords,
                progress_callback=progress_callback,
                use_multiprocessing=bool(use_mp),
                multiprocessing_threshold=int(mp_threshold),
                crossover_methods_allowed=crossover_methods_selected,
                crossover_operator_probs=crossover_operator_probs,
                mutation_methods=mutation_methods,
                adaptive_mutation=bool(adaptive_mutation),
                min_mutation_rate=float(min_mut),
                max_mutation_rate=float(max_mut),
                maximize=bool(maximize)
            )
            with st.spinner("Запуск GA... подождите"):
                t0 = time.time()
                best = ga.run()
                total_time = time.time() - t0
            st.success("GA завершён")
            st.session_state["solution"] = best
            st.session_state["reporter"] = reporter
            st.session_state["generation_stats"] = ga.generation_stats
            st.session_state["total_time"] = ga.total_time_sec if hasattr(ga, "total_time_sec") else round(total_time, 4)
        except Exception as e:
            logger.exception("Ошибка при запуске GA: %s", e)
            st.error(f"Ошибка GA: {e}")

    if run_sens_button:
        with st.spinner("Запуск анализа чувствительности..."):
            df_sens = sensitivity_analysis(locomotives, trains, station_coords,
                                           base_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
                                           deltas=[-0.2, -0.1, 0, 0.1, 0.2],
                                           population_size=30,
                                           generations=15,
                                           tournament_k=int(tournament_k),
                                           mutation_rate=float(mutation_rate),
                                           use_multiprocessing=False)
            st.session_state["sens_df"] = df_sens
        st.success("Анализ чувствительности завершён")

    if "solution" in st.session_state:
        solution: Chromosome = st.session_state["solution"]
        reporter: GAReporter = st.session_state.get("reporter", reporter)
        gen_stats = st.session_state.get("generation_stats", [])

        df_assign = build_assignment_dataframe(solution, locomotives, trains)
        assignment_placeholder.dataframe(df_assign, height=350)

        df_loco = build_locomotive_summary_dataframe(locomotives, trains, solution)
        loco_summary_placeholder.table(df_loco)

        fig_timeline = plot_assignment_matplotlib(solution, trains)
        timeline_placeholder.pyplot(fig_timeline)

        fig_gc = plot_generation_curve(reporter)
        evolution_placeholder.pyplot(fig_gc)

        fig_w = plot_weights_evolution(reporter)
        weights_plot_placeholder.pyplot(fig_w)

        fig_cw = plot_crossover_weights_evolution(reporter)
        crossover_weights_placeholder.pyplot(fig_cw)

        fig_c = plot_components_evolution(reporter)
        components_placeholder.pyplot(fig_c)

        total_time = st.session_state.get("total_time", None)
        time_summary_placeholder.markdown(f"**Общее время расчёта:** {total_time:.4f} с")
        time_summary_placeholder.markdown(f"**Итоговая пригодность (raw):** {solution.fitness:.6f}")

    if "sens_df" in st.session_state:
        df_sens: pd.DataFrame = st.session_state["sens_df"]
        sens_placeholder.subheader("Результаты анализа чувствительности")
        sens_placeholder.dataframe(df_sens, height=300)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df_sens["Δ"], df_sens["fitness"], marker="o", label="fitness")
        ax.set_xlabel("Δ (смещение весов)")
        ax.set_ylabel("Итоговая пригодность (fitness)")
        ax.set_title("Чувствительность итоговой пригодности к смещению весов")
        ax.legend(loc="best")
        ax.grid(True)
        sens_placeholder.pyplot(fig)


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
    reporter.print_summary(best, ga=ga, total_time_sec=getattr(ga, "total_time_sec", None))
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
        logger.info("Streamlit не доступен или отключён — запускаю CLI демо")
        run_cli_demo()
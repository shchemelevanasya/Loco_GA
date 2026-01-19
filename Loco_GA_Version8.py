# -*- coding: utf-8 -*-
"""
Loco_GA — Genetic Algorithm for locomotive assignment

Добавлены новые операторы кроссовера и мутации:

Кроссоверы:
  - one_point_crossover
  - two_point_crossover
  - uniform_crossover
  - priority_crossover  (учёт категорий поездов)

Мутации:
  - swap_locomotives_mutation (перестановка локомотивов между двумя поездами)
  - replace_locomotive_mutation (замена локомотива для поезда на подходящий свободный)
  - shuffle_within_window_mutation (перемешивание локомотивов в выбранной части расписания)

Реализация:
 - Для операций кроссовера внутри алгоритма используем представление train->loco (gene per train).
 - После кроссовера формируем обратно assignment: loco -> ordered list of train ids (по departure_time).
 - Все операции защищены проверкой is_feasible_fast — при недопустимости откатываем изменения.
 - Streamlit UI расширен для выбора типа кроссовера и включения/настройки мутаций.
 - Поезда получили поле category (категория/приоритет) для priority_crossover.
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
# Data classes (train category added)
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
    category: str = "normal"  # категория / приоритет: e.g. "express","passenger","freight"


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
# Synthetic data with categories and multiple loco types
# ---------------------
def generate_synthetic_data(
        num_locomotives: int = 10,
        num_trains: int = 20,
        depot_names: Tuple[str, ...] = ("A", "B", "C"),
        station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
        loco_types: Optional[Dict[str, Dict[str, Any]]] = None,
        categories: Optional[List[str]] = None,
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

    if categories is None:
        categories = ["express", "passenger", "freight"]

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
        cat = random.choices(categories, weights=[0.2, 0.6, 0.2])[0]
        trains[j] = Train(
            id=j,
            weight=random.uniform(3000, 6000),
            route=(dep, arr),
            departure_time=random.uniform(0, 24),
            duration=random.uniform(2, 6),
            category=cat
        )

    logger.info("Generated %d locos, %d trains, %d stations, categories=%s", len(locomotives), len(trains), len(station_coords), categories)
    return locomotives, trains, station_coords


# ---------------------
# Chromosome (assignment loco->ordered train list)
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
# Feasibility helpers
# ---------------------
def build_lookup_tables(trains: Dict[int, Train]):
    train_weight = {tid: t.weight for tid, t in trains.items()}
    train_duration = {tid: t.duration for tid, t in trains.items()}
    train_dep = {tid: t.route[0] for tid, t in trains.items()}
    train_arr = {tid: t.route[1] for tid, t in trains.items()}
    train_dep_time = {tid: t.departure_time for tid, t in trains.items()}
    return {"weight": train_weight, "duration": train_duration, "dep": train_dep, "arr": train_arr, "dep_time": train_dep_time}


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
        # traction
        for tid in train_ids:
            if weight[tid] > loco.power:
                return False
        # resource
        total_dur = 0.0
        for tid in train_ids:
            total_dur += duration[tid]
            if total_dur > loco.remaining_resource:
                return False
    return True


# ---------------------
# Conversion helpers between representations
# ---------------------
def assignment_to_train_map(assignment: Dict[int, List[int]]) -> Dict[int, int]:
    """
    Convert loco->list(trains) to train->loco map (latest loco if multiple, but assignment should be unique).
    """
    train_map: Dict[int, int] = {}
    for loco_id, trains in assignment.items():
        for t in trains:
            train_map[t] = loco_id
    return train_map


def train_map_to_assignment(train_map: Dict[int, int], locomotives: Dict[int, Locomotive], trains: Dict[int, Train]) -> Dict[int, List[int]]:
    """
    Convert train->loco mapping to assignment loco->ordered list of trains by departure_time.
    Ensure every loco appears as key.
    """
    assignment: Dict[int, List[int]] = {loco_id: [] for loco_id in locomotives.keys()}
    # group trains by loco
    for t_id, loco_id in train_map.items():
        if loco_id not in assignment:
            assignment[loco_id] = []
        assignment[loco_id].append(t_id)
    # order each list by departure_time
    dep_times = {tid: trains[tid].departure_time for tid in trains}
    for loco_id, tlist in assignment.items():
        tlist.sort(key=lambda x: dep_times.get(x, 0.0))
    return assignment


# ---------------------
# Crossover operators (operate on train->loco mapping)
# ---------------------
def one_point_crossover_map(parent1_map: Dict[int, int], parent2_map: Dict[int, int], train_order: List[int]) -> Dict[int, int]:
    """
    One-point crossover on linearized gene array (ordered list of train ids).
    Child gets prefix from parent1, suffix from parent2.
    """
    if not train_order:
        return parent1_map.copy()
    cut = random.randint(0, len(train_order))
    child = {}
    for i, tid in enumerate(train_order):
        if i < cut:
            child[tid] = parent1_map.get(tid, parent2_map.get(tid))
        else:
            child[tid] = parent2_map.get(tid, parent1_map.get(tid))
    return child


def two_point_crossover_map(parent1_map: Dict[int, int], parent2_map: Dict[int, int], train_order: List[int]) -> Dict[int, int]:
    if not train_order:
        return parent1_map.copy()
    a = random.randint(0, len(train_order) - 1)
    b = random.randint(a, len(train_order) - 1)
    child = {}
    for i, tid in enumerate(train_order):
        if a <= i <= b:
            child[tid] = parent2_map.get(tid, parent1_map.get(tid))
        else:
            child[tid] = parent1_map.get(tid, parent2_map.get(tid))
    return child


def uniform_crossover_map(parent1_map: Dict[int, int], parent2_map: Dict[int, int], train_order: List[int], p: float = 0.5) -> Dict[int, int]:
    child = {}
    for tid in train_order:
        if random.random() < p:
            child[tid] = parent1_map.get(tid, parent2_map.get(tid))
        else:
            child[tid] = parent2_map.get(tid, parent1_map.get(tid))
    return child


def priority_crossover_map(parent1_map: Dict[int, int], parent2_map: Dict[int, int],
                           trains: Dict[int, Train], train_order: List[int],
                           priority_categories: List[str]) -> Dict[int, int]:
    """
    Priority crossover:
      - choose which parent is better for priority trains by evaluating simple proxy:
        count of priority trains already assigned to locos with sufficient power in that parent.
      - child inherits priority trains assignments from best parent; other trains from other parent.
    """
    def priority_score(train_map, trains, priority_categories):
        score = 0
        for tid, t in trains.items():
            if t.category in priority_categories:
                # count if parent assigns loco with enough power (proxy)
                loco_id = train_map.get(tid)
                if loco_id is not None:
                    # proxy: +1 (we don't know loco power here; caller will filter later)
                    score += 1
        return score

    score1 = priority_score(parent1_map, trains, priority_categories)
    score2 = priority_score(parent2_map, trains, priority_categories)
    # choose parent for priority
    pri_parent = parent1_map if score1 >= score2 else parent2_map
    other_parent = parent2_map if pri_parent is parent1_map else parent1_map
    child = {}
    for tid in train_order:
        if trains[tid].category in priority_categories:
            child[tid] = pri_parent.get(tid, other_parent.get(tid))
        else:
            child[tid] = other_parent.get(tid, pri_parent.get(tid))
    return child


# ---------------------
# Mutation operators working on train_map (train->loco)
# ---------------------
def swap_locomotives_mutation(train_map: Dict[int, int], trains: Dict[int, Train], locomotives: Dict[int, Locomotive]) -> Dict[int, int]:
    """
    Choose two trains assigned to different locos and swap their loco assignments.
    """
    t_ids = list(train_map.keys())
    if len(t_ids) < 2:
        return train_map
    for _ in range(10):
        a, b = random.sample(t_ids, 2)
        loco_a = train_map[a]
        loco_b = train_map[b]
        if loco_a != loco_b:
            new_map = train_map.copy()
            new_map[a], new_map[b] = loco_b, loco_a
            return new_map
    return train_map


def replace_locomotive_mutation(train_map: Dict[int, int], trains: Dict[int, Train], locomotives: Dict[int, Locomotive]) -> Dict[int, int]:
    """
    For a randomly selected train try to replace its loco by another loco that is free in that time window.
    Simple heuristic: choose an alternative loco from same depot if possible, else random.
    Note: We don't perform full temporal conflict detection here — rely on feasibility check after mutation.
    """
    t_ids = list(train_map.keys())
    if not t_ids:
        return train_map
    tid = random.choice(t_ids)
    current_loco = train_map[tid]
    # candidate locos not equal current
    candidates = [lid for lid in locomotives.keys() if lid != current_loco]
    if not candidates:
        return train_map
    # prefer same depot locos
    preferred = []
    train = trains[tid]
    for lid in candidates:
        loco = locomotives[lid]
        if loco.home_depot == locomotives[current_loco].home_depot:
            preferred.append(lid)
    choices = preferred if preferred else candidates
    new_loco = random.choice(choices)
    new_map = train_map.copy()
    new_map[tid] = new_loco
    return new_map


def shuffle_within_window_mutation(train_map: Dict[int, int], trains: Dict[int, Train], window_size: int = 5) -> Dict[int, int]:
    """
    Select contiguous segment in train_order of length window_size and randomly permute the assigned locos among them.
    """
    train_order = sorted(trains.keys(), key=lambda x: trains[x].departure_time)
    if len(train_order) <= 1:
        return train_map
    ws = min(window_size, len(train_order))
    start = random.randint(0, len(train_order) - ws)
    segment = train_order[start:start + ws]
    locos_segment = [train_map[t] for t in segment]
    random.shuffle(locos_segment)
    new_map = train_map.copy()
    for t, l in zip(segment, locos_segment):
        new_map[t] = l
    return new_map


# ---------------------
# High-level helpers to perform crossover+mutation and convert back to assignment
# ---------------------
def produce_child_by_crossover(parent1: Chromosome, parent2: Chromosome,
                               trains: Dict[int, Train],
                               locomotives: Dict[int, Locomotive],
                               crossover_type: str = "one_point",
                               priority_categories: Optional[List[str]] = None,
                               uniform_p: float = 0.5) -> Chromosome:
    """
    Produce child chromosome using selected crossover type. Operates on train_map representation.
    """
    parent1_map = assignment_to_train_map(parent1.assignment)
    parent2_map = assignment_to_train_map(parent2.assignment)
    train_order = sorted(list(trains.keys()), key=lambda x: trains[x].departure_time)

    if crossover_type == "one_point":
        child_map = one_point_crossover_map(parent1_map, parent2_map, train_order)
    elif crossover_type == "two_point":
        child_map = two_point_crossover_map(parent1_map, parent2_map, train_order)
    elif crossover_type == "uniform":
        child_map = uniform_crossover_map(parent1_map, parent2_map, train_order, p=uniform_p)
    elif crossover_type == "priority":
        if priority_categories is None:
            priority_categories = ["express"]
        child_map = priority_crossover_map(parent1_map, parent2_map, trains, train_order, priority_categories)
    else:
        # fallback to uniform
        child_map = uniform_crossover_map(parent1_map, parent2_map, train_order, p=uniform_p)

    # convert back to assignment
    child_assignment = train_map_to_assignment(child_map, locomotives, trains)
    return Chromosome(child_assignment)


def apply_mutation_ops(chrom: Chromosome, trains: Dict[int, Train], locomotives: Dict[int, Locomotive],
                       swap_prob: float = 0.05, replace_prob: float = 0.05, shuffle_prob: float = 0.05,
                       shuffle_window: int = 5) -> Chromosome:
    """
    Apply mutation operators probabilistically to chromosome. Work on train_map and then convert back.
    """
    train_map = assignment_to_train_map(chrom.assignment)
    if random.random() < swap_prob:
        train_map = swap_locomotives_mutation(train_map, trains, locomotives)
    if random.random() < replace_prob:
        train_map = replace_locomotive_mutation(train_map, trains, locomotives)
    if random.random() < shuffle_prob:
        train_map = shuffle_within_window_mutation(train_map, trains, window_size=shuffle_window)
    new_assignment = train_map_to_assignment(train_map, locomotives, trains)
    return Chromosome(new_assignment)


# ---------------------
# Remaining algorithm logic (dynamic weights and GA) - adapted to new operators
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


# fitness evaluation across population using derived weights
def evaluate_population_and_set_fitness(population: List[Chromosome],
                                        locomotives: Dict[int, Locomotive],
                                        trains: Dict[int, Train],
                                        station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                                        use_mp: bool = False):
    # compute comps for all
    comps = [compute_components(ch, locomotives, trains, station_coords) for ch in population]
    if comps:
        means = [statistics.mean([c[i] for c in comps]) for i in range(5)]
    else:
        means = [0.0] * 5
    eps = 1e-6
    raw = [1.0 / (m + eps) for m in means]
    s = sum(raw)
    if s <= 0:
        weights = tuple(0.2 for _ in range(5))
    else:
        weights = tuple(r / s for r in raw)

    # compute fitness per chrom (normalized by means)
    if use_mp and len(population) >= 50 and multiprocessing.cpu_count() > 1:
        args = [(ch.assignment, locomotives, trains, station_coords, weights, means) for ch in population]
        def _worker(args):
            assign, locos, trs, sc, ws, means_loc = args
            ch_local = Chromosome(assign)
            comps_ch = compute_components(ch_local, locos, trs, sc)
            norm = [comps_ch[i] / (means_loc[i] + eps) for i in range(5)]
            penalty = sum(ws[i] * norm[i] for i in range(5))
            ch_local.fitness = -penalty
            return ch_local.assignment, ch_local.fitness
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(args))) as exc:
            for assignment, fitness in exc.map(_worker, args):
                for ch in population:
                    if ch.assignment == assignment:
                        ch.fitness = fitness
                        break
    else:
        for ch in population:
            comps_ch = compute_components(ch, locomotives, trains, station_coords)
            norm = [comps_ch[i] / (means[i] + eps) for i in range(5)]
            penalty = sum(weights[i] * norm[i] for i in range(5))
            ch.fitness = -penalty

    # return derived weights and means for reporting
    return weights, means


# ---------------------
# GA class with choice of crossover and mutation operators
# ---------------------
class GeneticAlgorithm:
    def __init__(self, locomotives: Dict[int, Locomotive], trains: Dict[int, Train],
                 population_size: int = 50, generations: int = 100,
                 tournament_k: int = 3, mutation_rate: float = 0.1,
                 crossover_type: str = "one_point",
                 priority_categories: Optional[List[str]] = None,
                 swap_prob: float = 0.05, replace_prob: float = 0.03, shuffle_prob: float = 0.03,
                 shuffle_window: int = 5,
                 use_multiprocessing: bool = True,
                 multiprocessing_threshold: int = 100,
                 station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                 progress_callback: Optional[callable] = None):
        self.locomotives = locomotives
        self.trains = trains
        self.population_size = max(1, int(population_size))
        self.generations = max(1, int(generations))
        self.tournament_k = max(1, min(int(tournament_k), self.population_size))
        self.mutation_rate = float(mutation_rate)
        self.crossover_type = crossover_type
        self.priority_categories = priority_categories or ["express"]
        self.swap_prob = swap_prob
        self.replace_prob = replace_prob
        self.shuffle_prob = shuffle_prob
        self.shuffle_window = shuffle_window
        self.use_multiprocessing = use_multiprocessing
        self.multiprocessing_threshold = multiprocessing_threshold
        self.station_coords = station_coords
        self.progress_callback = progress_callback
        self.cpu_count = max(1, multiprocessing.cpu_count())

    def _initial_population(self) -> List[Chromosome]:
        return generate_initial_population(self.population_size, self.locomotives, self.trains)

    def _select_parents(self, population: List[Chromosome]) -> Tuple[Chromosome, Chromosome]:
        p1 = tournament_selection(population, self.tournament_k)
        p2 = tournament_selection(population, self.tournament_k)
        return p1, p2

    def run(self) -> Chromosome:
        population = self._initial_population()
        generation_stats = []
        total_start = time.time()
        train_order = sorted(list(self.trains.keys()), key=lambda x: self.trains[x].departure_time)

        for gen in range(self.generations):
            gen_start = time.time()
            # evaluate and set fitness; also get weights/means
            weights, means = evaluate_population_and_set_fitness(population, self.locomotives, self.trains,
                                                                 self.station_coords, use_mp=self.use_multiprocessing)
            fitnesses = [ch.fitness for ch in population]
            stats = {
                "best": max(fitnesses),
                "mean": statistics.mean(fitnesses),
                "std": statistics.pstdev(fitnesses) if len(fitnesses) > 1 else 0.0,
                "min": min(fitnesses),
                "weights": weights,
                "means_components": means
            }
            gen_eval_time = time.time() - gen_start
            stats["time_sec"] = round(gen_eval_time, 4)
            stats["gen"] = gen
            generation_stats.append(stats)
            if self.progress_callback:
                try:
                    self.progress_callback(gen, stats)
                except Exception:
                    try:
                        self.progress_callback(gen, stats["best"])
                    except Exception:
                        pass

            # create next generation
            new_population: List[Chromosome] = []
            # keep elitism: carry best 2
            elites = sorted(population, key=lambda c: c.fitness, reverse=True)[:2]
            new_population.extend(Chromosome(copy.deepcopy(e.assignment)) for e in elites)

            while len(new_population) < self.population_size:
                p1, p2 = self._select_parents(population)
                # produce child by chosen crossover
                child = produce_child_by_crossover(p1, p2, self.trains, self.locomotives,
                                                   crossover_type=self.crossover_type,
                                                   priority_categories=self.priority_categories,
                                                   uniform_p=0.5)
                # apply mutation with probability
                if random.random() < self.mutation_rate:
                    child = apply_mutation_ops(child, self.trains, self.locomotives,
                                               swap_prob=self.swap_prob, replace_prob=self.replace_prob,
                                               shuffle_prob=self.shuffle_prob, shuffle_window=self.shuffle_window)
                # if feasible, accept; else try simple repair (fallback: keep one of parents)
                if is_feasible_fast(child, self.locomotives, self.trains):
                    new_population.append(child)
                else:
                    # try alternate parent clones or parents themselves
                    if is_feasible_fast(p1, self.locomotives, self.trains):
                        new_population.append(Chromosome(copy.deepcopy(p1.assignment)))
                    elif is_feasible_fast(p2, self.locomotives, self.trains):
                        new_population.append(Chromosome(copy.deepcopy(p2.assignment)))
                    else:
                        # fallback: random assignment
                        new_population.append(generate_initial_population(1, self.locomotives, self.trains)[0])

            population = new_population

        # final evaluation
        weights, means = evaluate_population_and_set_fitness(population, self.locomotives, self.trains, self.station_coords, use_mp=self.use_multiprocessing)
        fitnesses = [ch.fitness for ch in population]
        final_stats = {
            "best": max(fitnesses),
            "mean": statistics.mean(fitnesses),
            "std": statistics.pstdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            "min": min(fitnesses),
            "weights": weights,
            "means_components": means
        }
        generation_stats.append({"gen": self.generations, **final_stats})
        self.generation_stats = generation_stats
        self.total_time_sec = round(time.time() - total_start, 4)
        best = max(population, key=lambda c: c.fitness)
        logger.info("GA finished: best fitness=%.6f, total_time=%.4f s", best.fitness, self.total_time_sec)
        return best


# ---------------------
# Existing helpers for population, reporting, plotting etc.
# (unchanged or adapted for categories)
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
        logger.warning("Initial feasible population not found; fallback used")
    return population


def tournament_selection(population: List[Chromosome], k: int = 3) -> Chromosome:
    k = min(k, len(population))
    if k <= 0:
        raise ValueError("Популяция пуста")
    candidates = random.sample(population, k)
    return max(candidates, key=lambda c: c.fitness)


# Reporting and plotting (Russian labels) - reuse functions from previous versions adapted to categories
class GAReporter:
    def __init__(self):
        self.generation_log: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def log_generation(self, gen: int, stats: Dict[str, Any]):
        entry = {
            "поколение": gen,
            "best": stats.get("best"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "weights": stats.get("weights"),
            "means_components": stats.get("means_components"),
            "time_sec": stats.get("time_sec")
        }
        self.generation_log.append(entry)

    def elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def print_summary(self, solution: Chromosome, total_time_sec: Optional[float] = None):
        print("\n=== Время расчёта и итоговая пригодность ===")
        print(f"Время расчёт��, с: {total_time_sec if total_time_sec is not None else self.elapsed():.2f}")
        print(f"Итоговая целевая функция (fitness): {solution.fitness:.6f}")


def build_assignment_dataframe(solution: Chromosome,
                               locomotives: Dict[int, Locomotive],
                               trains: Dict[int, Train]) -> pd.DataFrame:
    rows = []
    for loco_id, train_ids in solution.assignment.items():
        loco = locomotives.get(loco_id)
        for t_id in train_ids or [None]:
            if t_id is None:
                rows.append({
                    "Локомотив": loco_id,
                    "Тип": loco.loco_type if loco else None,
                    "Депо": loco.home_depot if loco else None,
                    "Остаток_ресурса_ч": loco.remaining_resource if loco else None,
                    "Поезд": None,
                    "Категория": None,
                    "Откуда": None,
                    "Куда": None,
                    "Отпр (ч)": None,
                    "Приб (ч)": None
                })
            else:
                t = trains[t_id]
                rows.append({
                    "Локомотив": loco_id,
                    "Тип": loco.loco_type if loco else None,
                    "Депо": loco.home_depot if loco else None,
                    "Остаток_ресурса_ч": loco.remaining_resource if loco else None,
                    "Поезд": t.id,
                    "Категория": t.category,
                    "Откуда": t.route[0],
                    "Куда": t.route[1],
                    "Отпр (ч)": round(t.departure_time, 2),
                    "Приб (ч)": round(t.departure_time + t.duration, 2)
                })
    df = pd.DataFrame(rows, columns=["Локомотив", "��ип", "Депо", "Остаток_ресурса_ч", "Поезд", "Категория", "Откуда", "Куда", "Отпр (ч)", "Приб (ч)"])
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


def plot_assignment_matplotlib(solution: Chromosome, trains: Dict[int, Train]):
    fig, ax = plt.subplots(figsize=(10, max(4, len(solution.assignment) * 0.3)))
    y = 0
    ylabels = []
    for loco_id, train_ids in solution.assignment.items():
        for t_id in train_ids:
            t = trains[t_id]
            ax.barh(y, t.duration, left=t.departure_time, height=0.4, label=f"Поезд {t.id} ({t.category})" if y == 0 else "")
        ylabels.append(f"{loco_id}")
        y += 1
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Локомотивы (id)")
    ax.set_title("График назначений — временная диаграмма")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_generation_curve(reporter: GAReporter):
    if not reporter.generation_log:
        fig, ax = plt.subplots()
        ax.set_title("Кривая прогрессии поколений (нет данных)")
        return fig
    df = pd.DataFrame(reporter.generation_log)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["поколение"], df["best"], label="Лучшее (best)", marker="o")
    ax.plot(df["поколение"], df["mean"], label="Среднее (mean)", marker="o")
    ax.plot(df["поколение"], df["min"], label="Минимум (min)", marker="o")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Значение целевой функции (fitness)")
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
        ax.set_title("Эволюция весов (нет да��ных)")
        return fig
    gens = [e["поколение"] for e in reporter.generation_log]
    ws = [e["weights"] for e in reporter.generation_log]
    if not ws:
        return plt.figure()
    comp_names = ["Простой", "Порожние", "Ожидание поездов", "Ожидание локомотивов", "Используемых локомотивов"]
    fig, ax = plt.subplots(figsize=(9, 3))
    for i in range(5):
        ax.plot(gens, [w[i] for w in ws], marker="o", label=comp_names[i])
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Вес критерия")
    ax.set_title("Динамика весов критериев по поколениям")
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
    comp_arr = list(zip(*means))
    labels = ["Простой, ч", "Порожние, ч", "Ожидание поездов, ч", "Ожидание локомотивов, ч", "Используемых локомотивов"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, comp in enumerate(comp_arr):
        ax.plot(gens, comp, label=labels[i], marker="o")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Средние значения компонентов")
    ax.set_title("Динамика компонентов по поколениям")
    ax.legend(loc="best")
    ax.grid(True)
    return fig


# ---------------------
# Streamlit UI additions: choose crossover & mutation options
# ---------------------
def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit не установлен в окружении.")
        return

    st.set_page_config(page_title="Loco_GA", layout="wide")
    st.title("Loco_GA — Назначение локомотивов (Генетический алгоритм)")

    st.sidebar.header("Параметры GA")
    population_size = st.sidebar.number_input("Размер популяции", 2, 2000, 60)
    generations = st.sidebar.number_input("Поколений", 1, 1000, 60)
    tournament_k = st.sidebar.number_input("Размер турнира", 1, population_size, 5)
    mutation_rate = st.sidebar.slider("Вероятность мутации (общая)", 0.0, 1.0, 0.15)

    crossover_type = st.sidebar.selectbox("Тип кроссовера", ["one_point", "two_point", "uniform", "priority"])
    priority_cats_input = st.sidebar.text_input("Категории приоритета (для priority, через запятую)", "express")
    priority_cats = [s.strip() for s in priority_cats_input.split(",") if s.strip()]

    st.sidebar.markdown("Мутации (вероятности отдельных операторов)")
    swap_prob = st.sidebar.slider("Перестановка локомотивов (swap)", 0.0, 1.0, 0.05)
    replace_prob = st.sidebar.slider("Замена локомотива (replace)", 0.0, 1.0, 0.03)
    shuffle_prob = st.sidebar.slider("Перемешивание в окне (shuffle)", 0.0, 1.0, 0.03)
    shuffle_window = st.sidebar.number_input("Окно перемешивания (число поездов)", 2, 50, 5)

    st.sidebar.markdown("---")
    num_locomotives = st.sidebar.slider("Число локомотивов", 1, 200, 10)
    num_trains = st.sidebar.slider("Число поездов", 1, 500, 20)
    depots = st.sidebar.text_input("Депо/станции (через запятую)", "A,B,C")
    depot_names = tuple(s.strip() for s in depots.split(",") if s.strip())
    seed = st.sidebar.number_input("Random seed (0=произвольно)", 0, 999999, 0)
    if seed == 0:
        seed = None

    if st.button("Сгенерировать данные"):
        locomotives, trains, station_coords = generate_synthetic_data(num_locomotives, num_trains, depot_names, seed=seed)
        st.session_state["locomotives"] = locomotives
        st.session_state["trains"] = trains
        st.session_state["station_coords"] = station_coords
        st.success("Данные сгенерированы")

    if "locomotives" not in st.session_state:
        locomotives, trains, station_coords = generate_synthetic_data(num_locomotives, num_trains, depot_names, seed=seed)
        st.session_state["locomotives"] = locomotives
        st.session_state["trains"] = trains
        st.session_state["station_coords"] = station_coords

    locomotives = st.session_state["locomotives"]
    trains = st.session_state["trains"]
    station_coords = st.session_state["station_coords"]

    left, right = st.columns((2, 1))
    with left:
        st.markdown("### Таблица назначений")
        assignment_ph = st.empty()
        st.markdown("### Сводная таблица локомотивов")
        loco_ph = st.empty()
        st.markdown("### График назначений")
        timeline_ph = st.empty()
        st.markdown("### Кривая эволюции")
        evolution_ph = st.empty()
        st.markdown("### Динамика компонентов")
        components_ph = st.empty()
        st.markdown("### Анализ чувствительности (по весам)")
        sens_ph = st.empty()

    with right:
        st.markdown("### Параметры и статус")
        status_ph = st.empty()
        gen_stats_ph = st.empty()
        weights_ph = st.empty()
        time_ph = st.empty()

    run_button = st.button("Запустить GA")

    reporter = GAReporter()
    reporter.start()

    if run_button:
        ga = GeneticAlgorithm(locomotives, trains,
                              population_size=int(population_size),
                              generations=int(generations),
                              tournament_k=int(tournament_k),
                              mutation_rate=float(mutation_rate),
                              crossover_type=crossover_type,
                              priority_categories=priority_cats,
                              swap_prob=float(swap_prob),
                              replace_prob=float(replace_prob),
                              shuffle_prob=float(shuffle_prob),
                              shuffle_window=int(shuffle_window),
                              use_multiprocessing=True,
                              multiprocessing_threshold=50,
                              station_coords=station_coords,
                              progress_callback=lambda gen, stats: (gen_stats_ph.metric(label=f"Поколение {gen}", value=f"Лучшее: {stats['best']:.6f}", delta=f"Среднее: {stats['mean']:.6f}"),
                                                                    reporter.log_generation(gen, stats)))
        with st.spinner("Запуск GA..."):
            t0 = time.time()
            best = ga.run()
            total_time = time.time() - t0
        st.session_state["solution"] = best
        st.session_state["reporter"] = reporter
        st.session_state["total_time"] = round(total_time, 4)

    if "solution" in st.session_state:
        solution = st.session_state["solution"]
        reporter = st.session_state["reporter"]
        df_assign = build_assignment_dataframe(solution, locomotives, trains)
        assignment_ph.dataframe(df_assign, height=400)
        df_loco = build_locomotive_summary_dataframe(locomotives, trains, solution)
        loco_ph.table(df_loco)
        fig = plot_assignment_matplotlib(solution, trains)
        timeline_ph.pyplot(fig)
        fig_gc = plot_generation_curve(reporter)
        evolution_ph.pyplot(fig_gc)
        fig_comp = plot_components_evolution(reporter)
        components_ph.pyplot(fig_comp)
        time_ph.markdown(f"**Общее время расчёта:** {st.session_state.get('total_time', 0.0)} с")
        time_ph.markdown(f"**Итоговая пригодность (fitness):** {solution.fitness:.6f}")

    st.sidebar.markdown("Логи: файл loco_ga.log")

# ---------------------
# CLI fallback
# ---------------------
def run_cli_demo():
    locomotives, trains, station_coords = generate_synthetic_data()
    ga = GeneticAlgorithm(locomotives, trains, population_size=60, generations=40)
    reporter = GAReporter()
    reporter.start()
    best = ga.run()
    reporter.print_summary(best, total_time_sec=getattr(ga, "total_time_sec", None))
    print(build_assignment_dataframe(best, locomotives, trains))
    fig = plot_assignment_matplotlib(best, trains)
    plt.show()


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE and os.environ.get("RUN_STREAMLIT", "1") == "1":
        run_streamlit_app()
    else:
        logger.info("Streamlit not available — running CLI demo")
        run_cli_demo()
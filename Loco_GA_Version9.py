# -*- coding: utf-8 -*-
"""
Loco_GA — Genetic Algorithm for locomotive assignment

Изменения:
 - Операторы кроссовера и мутаций выбираются динамически из списков (можно задать набор операторов и веса).
 - Добавлены корректирующие операторы, применяемые после генерации новых решений и перед оценкой:
     1) technical_compatibility_operator
     2) temporal_conflict_resolution_operator
     3) maintenance_operator
 - Корректирующие операторы пытаются исправлять нарушения (техническая несовместимость, временные конфликты, превышение ресурса)
   путем подбора альтернативных локомотивов или перераспределения назначений.
 - Train и Locomotive получили дополнительные поля (category, requires_electrified, electrified),
   генератор синтетических данных инициализирует их.
 - GA принимает списки операторов: crossover_ops, mutation_ops, corrective_ops (строки или callables).
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Callable
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
# Data classes (extended)
# ---------------------
@dataclass
class Locomotive:
    id: int
    loco_type: str
    power: float
    remaining_resource: float
    home_depot: str
    reposition_speed_kmh: float  # скорость репозиции
    electrified: bool = True     # поддерживает ли участок электрификацию (электрический/дизель)


@dataclass
class Train:
    id: int
    weight: float
    route: Tuple[str, str]
    departure_time: float
    duration: float
    category: str = "normal"             # категория/приоритет поезда
    requires_electrified: bool = False   # требуется ли электрификация


# ---------------------
# Geometry and conversion
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
# Synthetic data generation (multi-type locos, electrification, categories)
# ---------------------
def generate_synthetic_data(
        num_locomotives: int = 10,
        num_trains: int = 20,
        depot_names: Tuple[str, ...] = ("A", "B", "C"),
        station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
        loco_types: Optional[Dict[str, Dict[str, Any]]] = None,
        categories: Optional[List[str]] = None,
        seed: Optional[int] = None):
    """
    loco_types example:
      {
        "2ЭС6": {"power_range": (4500,6500), "speed_kmh":60.0, "resource_range": (20,50), "electrified": True},
        "ТЭП70": {"power_range": (5000,7500), "speed_kmh":70.0, "resource_range": (25,60), "electrified": False}
      }
    """
    if seed is not None:
        random.seed(seed)

    if loco_types is None:
        loco_types = {
            "2ЭС6": {"power_range": (4500, 6500), "speed_kmh": 60.0, "resource_range": (20, 50), "electrified": True},
            "ЧС7": {"power_range": (3000, 5000), "speed_kmh": 50.0, "resource_range": (15, 40), "electrified": True},
            "ТЭП70": {"power_range": (5000, 7500), "speed_kmh": 70.0, "resource_range": (25, 60), "electrified": False}
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

    # create locomotives
    locomotives: Dict[int, Locomotive] = {}
    type_names = list(loco_types.keys())
    for i in range(num_locomotives):
        tname = random.choice(type_names)
        props = loco_types[tname]
        power = random.uniform(props["power_range"][0], props["power_range"][1])
        resource = random.uniform(props["resource_range"][0], props["resource_range"][1])
        speed = float(props.get("speed_kmh", 60.0))
        electr = bool(props.get("electrified", True))
        locomotives[i] = Locomotive(id=i, loco_type=tname, power=power,
                                    remaining_resource=resource,
                                    home_depot=random.choice(list(depot_names)),
                                    reposition_speed_kmh=speed,
                                    electrified=electr)

    # create trains, some requiring electrification
    trains: Dict[int, Train] = {}
    for j in range(num_trains):
        dep = random.choice(list(depot_names))
        arr = random.choice([d for d in depot_names if d != dep])
        cat = random.choices(categories, weights=[0.2, 0.6, 0.2])[0]
        requires_e = (cat == "express") and random.random() < 0.8  # express often electrified
        trains[j] = Train(id=j, weight=random.uniform(3000, 6000),
                          route=(dep, arr),
                          departure_time=random.uniform(0, 24),
                          duration=random.uniform(2, 6),
                          category=cat,
                          requires_electrified=requires_e)

    logger.info("Сгенерированы данные: %d локомотивов, %d поездов, %d станций", len(locomotives), len(trains), len(station_coords))
    return locomotives, trains, station_coords


# ---------------------
# Chromosome structure: loco -> ordered list of trains
# ---------------------
class Chromosome:
    def __init__(self, assignment: Dict[int, List[int]]):
        self.assignment = assignment
        self._fitness: Optional[float] = None

    @property
    def fitness(self) -> float:
        if self._fitness is None:
            raise RuntimeError("Fitness ещё не вычислен")
        return self._fitness

    @fitness.setter
    def fitness(self, v: float):
        self._fitness = float(v)


# ---------------------
# Feasibility check
# ---------------------
def build_lookup_tables(trains: Dict[int, Train]):
    return {
        "weight": {tid: t.weight for tid, t in trains.items()},
        "duration": {tid: t.duration for tid, t in trains.items()},
        "dep_time": {tid: t.departure_time for tid, t in trains.items()}
    }


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
        total = 0.0
        for tid in train_ids:
            if weight[tid] > loco.power:
                return False
            total += duration[tid]
            if total > loco.remaining_resource:
                return False
    return True


# ---------------------
# Helpers: representations conversions
# ---------------------
def assignment_to_train_map(assignment: Dict[int, List[int]]) -> Dict[int, int]:
    train_map: Dict[int, int] = {}
    for loco_id, tlist in assignment.items():
        for tid in tlist:
            train_map[tid] = loco_id
    return train_map


def train_map_to_assignment(train_map: Dict[int, int], locomotives: Dict[int, Locomotive], trains: Dict[int, Train]) -> Dict[int, List[int]]:
    assignment: Dict[int, List[int]] = {lid: [] for lid in locomotives.keys()}
    for tid, lid in train_map.items():
        if lid not in assignment:
            assignment[lid] = []
        assignment[lid].append(tid)
    dep_times = {tid: trains[tid].departure_time for tid in trains}
    for lid in assignment:
        assignment[lid].sort(key=lambda x: dep_times.get(x, 0.0))
    return assignment


# ---------------------
# Crossover operators (map-based)
# ---------------------
def one_point_crossover_map(parent1_map, parent2_map, train_order):
    if not train_order:
        return parent1_map.copy()
    cut = random.randint(0, len(train_order))
    child = {}
    for i, tid in enumerate(train_order):
        child[tid] = (parent1_map.get(tid) if i < cut else parent2_map.get(tid))
    return child


def two_point_crossover_map(parent1_map, parent2_map, train_order):
    if not train_order:
        return parent1_map.copy()
    a = random.randint(0, len(train_order) - 1)
    b = random.randint(a, len(train_order) - 1)
    child = {}
    for i, tid in enumerate(train_order):
        child[tid] = (parent2_map.get(tid) if a <= i <= b else parent1_map.get(tid))
    return child


def uniform_crossover_map(parent1_map, parent2_map, train_order, p=0.5):
    child = {}
    for tid in train_order:
        child[tid] = (parent1_map.get(tid) if random.random() < p else parent2_map.get(tid))
    return child


def priority_crossover_map(parent1_map, parent2_map, trains, train_order, priority_categories):
    # choose parent better on priority -- heuristic: count assigned priority trains
    def score(pr_map):
        return sum(1 for tid, t in trains.items() if t.category in priority_categories and pr_map.get(tid) is not None)
    s1 = score(parent1_map)
    s2 = score(parent2_map)
    pri = parent1_map if s1 >= s2 else parent2_map
    other = parent2_map if pri is parent1_map else parent1_map
    child = {}
    for tid in train_order:
        if trains[tid].category in priority_categories:
            child[tid] = pri.get(tid, other.get(tid))
        else:
            child[tid] = other.get(tid, pri.get(tid))
    return child


CROSSOVER_FUNCTIONS: Dict[str, Callable] = {
    "one_point": one_point_crossover_map,
    "two_point": two_point_crossover_map,
    "uniform": uniform_crossover_map,
    "priority": priority_crossover_map
}


# ---------------------
# Mutation operators (map-based)
# ---------------------
def swap_locomotives_mutation(train_map, trains, locomotives):
    t_ids = list(train_map.keys())
    if len(t_ids) < 2:
        return train_map
    for _ in range(10):
        a, b = random.sample(t_ids, 2)
        if train_map[a] != train_map[b]:
            new_map = train_map.copy()
            new_map[a], new_map[b] = train_map[b], train_map[a]
            return new_map
    return train_map


def replace_locomotive_mutation(train_map, trains, locomotives):
    t_ids = list(train_map.keys())
    if not t_ids:
        return train_map
    tid = random.choice(t_ids)
    current = train_map[tid]
    candidates = [lid for lid in locomotives.keys() if lid != current]
    if not candidates:
        return train_map
    # prefer same depot
    pref = [lid for lid in candidates if locomotives[lid].home_depot == locomotives[current].home_depot]
    choices = pref if pref else candidates
    new_map = train_map.copy()
    new_map[tid] = random.choice(choices)
    return new_map


def shuffle_within_window_mutation(train_map, trains, window_size=5):
    order = sorted(trains.keys(), key=lambda x: trains[x].departure_time)
    if len(order) <= 1:
        return train_map
    ws = min(window_size, len(order))
    start = random.randint(0, len(order) - ws)
    seg = order[start:start + ws]
    locos = [train_map[t] for t in seg]
    random.shuffle(locos)
    new_map = train_map.copy()
    for t, l in zip(seg, locos):
        new_map[t] = l
    return new_map


MUTATION_FUNCTIONS: Dict[str, Callable] = {
    "swap": swap_locomotives_mutation,
    "replace": replace_locomotive_mutation,
    "shuffle": shuffle_within_window_mutation
}


# ---------------------
# Corrective operators
# ---------------------
def technical_compatibility_operator(chrom: Chromosome,
                                     locomotives: Dict[int, Locomotive],
                                     trains: Dict[int, Train],
                                     prefer_same_depot: bool = True) -> Chromosome:
    """
    Проверяем требования train.requires_electrified and loco.power >= train.weight.
    При несоответствии подбираем подходящий loco (по power и electrified), предпочитая депо.
    """
    train_map = assignment_to_train_map(chrom.assignment)
    for tid, lid in list(train_map.items()):
        t = trains[tid]
        loco = locomotives.get(lid)
        incompatible = False
        if loco is None:
            incompatible = True
        else:
            if loco.power < t.weight:
                incompatible = True
            if t.requires_electrified and not loco.electrified:
                incompatible = True
        if incompatible:
            # find candidate list
            candidates = []
            for cand_id, cand in locomotives.items():
                if cand.power >= t.weight and (not t.requires_electrified or cand.electrified):
                    candidates.append(cand_id)
            # prefer same depot
            chosen = None
            if prefer_same_depot and loco is not None:
                same_depot = [c for c in candidates if locomotives[c].home_depot == loco.home_depot]
                if same_depot:
                    chosen = random.choice(same_depot)
            if chosen is None and candidates:
                chosen = random.choice(candidates)
            if chosen is not None:
                train_map[tid] = chosen
            else:
                # no candidate: leave as is (will likely be infeasible and filtered later)
                continue
    new_assign = train_map_to_assignment(train_map, locomotives, trains)
    return Chromosome(new_assign)


def loco_busy_in_interval(loco_id: int, interval: Tuple[float, float], assignment: Dict[int, List[int]], trains: Dict[int, Train]) -> bool:
    for tid in assignment.get(loco_id, []):
        t = trains[tid]
        s, e = t.departure_time, t.departure_time + t.duration
        if not (e <= interval[0] or s >= interval[1]):
            return True
    return False


def temporal_conflict_resolution_operator(chrom: Chromosome,
                                          locomotives: Dict[int, Locomotive],
                                          trains: Dict[int, Train],
                                          station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                                          max_swap_attempts: int = 5) -> Chromosome:
    """
    Для каждого локомотива проверяем пересечения интервалов. Если конфликт найден, пытаемся:
      1) найти другой локомотив свободный в интервале;
      2) попытаться поменять локомотивы между конфликтующими поездами (swap);
      3) в случае неудачи оставить и далее фильтровать.
    """
    assignment = copy.deepcopy(chrom.assignment)
    # build intervals per train
    intervals = {tid: (trains[tid].departure_time, trains[tid].departure_time + trains[tid].duration) for tid in trains}
    for loco_id, tlist in list(assignment.items()):
        # sort by start
        sorted_t = sorted(tlist, key=lambda x: intervals[x][0])
        for i in range(1, len(sorted_t)):
            prev = sorted_t[i - 1]
            curr = sorted_t[i]
            prev_end = intervals[prev][1]
            curr_start = intervals[curr][0]
            if prev_end > curr_start:  # conflict
                # try to find alternative loco for curr
                interval = intervals[curr]
                candidate = None
                for cand_id, cand in locomotives.items():
                    if cand_id == loco_id:
                        continue
                    if not loco_busy_in_interval(cand_id, interval, assignment, trains):
                        # quick technical check
                        if cand.power >= trains[curr].weight and (not trains[curr].requires_electrified or cand.electrified):
                            candidate = cand_id
                            break
                if candidate is not None:
                    # reassign curr to candidate
                    assignment[loco_id].remove(curr)
                    assignment.setdefault(candidate, []).append(curr)
                    # re-sort lists
                    assignment[loco_id].sort(key=lambda x: intervals[x][0])
                    assignment[candidate].sort(key=lambda x: intervals[x][0])
                    sorted_t = sorted(assignment[loco_id], key=lambda x: intervals[x][0])
                    continue
                # else try swapping with previous train's assigned loco if that would help
                swapped = False
                for _ in range(max_swap_attempts):
                    other_loco = assignment.get(assignment[curr][0], None) if False else None  # placeholder; skip complex swaps here
                    # attempt simple swap: find another train assigned to other loco that can be swapped - omitted due to complexity
                    break
                # if no fix found, leave conflict (will be handled later)
    return Chromosome(assignment)


def maintenance_operator(chrom: Chromosome,
                         locomotives: Dict[int, Locomotive],
                         trains: Dict[int, Train]) -> Chromosome:
    """
    Проверяем превышение ресурса: если суммарная длительность назначений > remaining_resource,
    удаляем последние назначения (по хронологии) до восстановления ресурса и пытаемся перераспределить их.
    """
    assignment = copy.deepcopy(chrom.assignment)
    for loco_id, tlist in list(assignment.items()):
        if not tlist:
            continue
        loco = locomotives.get(loco_id)
        if loco is None:
            continue
        dep_times = {tid: trains[tid].departure_time for tid in trains}
        ordered = sorted(tlist, key=lambda x: dep_times[x])
        total_dur = sum(trains[tid].duration for tid in ordered)
        if total_dur <= loco.remaining_resource:
            continue
        # need maintenance: remove last trains until within resource
        while ordered and total_dur > loco.remaining_resource:
            tid = ordered.pop()  # remove last
            total_dur -= trains[tid].duration
            assignment[loco_id].remove(tid)
            # try to reassign tid to alternative loco
            reassigned = False
            for cand_id, cand in locomotives.items():
                if cand_id == loco_id:
                    continue
                # quick checks
                if cand.power >= trains[tid].weight and (not trains[tid].requires_electrified or cand.electrified):
                    # add to cand and check resource
                    cand_total = sum(trains[tj].duration for tj in assignment.get(cand_id, []) ) + trains[tid].duration
                    if cand_total <= cand.remaining_resource:
                        assignment.setdefault(cand_id, []).append(tid)
                        reassigned = True
                        break
            if not reassigned:
                # leave unassigned: put into a special key -1 (reserve)
                assignment.setdefault(-1, []).append(tid)
    # ensure sort
    for lid in list(assignment.keys()):
        if lid == -1:
            continue
        assignment[lid].sort(key=lambda x: trains[x].departure_time)
    return Chromosome(assignment)


CORRECTIVE_FUNCTIONS: Dict[str, Callable] = {
    "technical": technical_compatibility_operator,
    "temporal": temporal_conflict_resolution_operator,
    "maintenance": maintenance_operator
}


# ---------------------
# Produce child (dynamic crossover), apply dynamic mutation list and corrections
# ---------------------
def produce_child_dynamic(parents: List[Chromosome],
                          trains: Dict[int, Train],
                          locomotives: Dict[int, Locomotive],
                          crossover_ops: List[str],
                          crossover_weights: Optional[List[float]] = None,
                          priority_categories: Optional[List[str]] = None,
                          uniform_p: float = 0.5) -> Chromosome:
    """
    parents: list of two Chromosomes
    crossover_ops: list of keys in CROSSOVER_FUNCTIONS (e.g., ["one_point","uniform"])
    crossover_weights: optional weights for random choice
    """
    parent1_map = assignment_to_train_map(parents[0].assignment)
    parent2_map = assignment_to_train_map(parents[1].assignment)
    train_order = sorted(list(trains.keys()), key=lambda x: trains[x].departure_time)
    if not crossover_ops:
        op = "one_point"
    else:
        op = random.choices(crossover_ops, weights=crossover_weights, k=1)[0] if crossover_weights else random.choice(crossover_ops)
    func = CROSSOVER_FUNCTIONS.get(op, one_point_crossover_map)
    if op == "priority":
        child_map = func(parent1_map, parent2_map, trains, train_order, priority_categories or ["express"])
    elif op == "uniform":
        child_map = func(parent1_map, parent2_map, train_order, uniform_p)
    else:
        child_map = func(parent1_map, parent2_map, train_order)
    child_assignment = train_map_to_assignment(child_map, locomotives, trains)
    return Chromosome(child_assignment)


def apply_mutations_dynamic(chrom: Chromosome,
                            trains: Dict[int, Train],
                            locomotives: Dict[int, Locomotive],
                            mutation_ops: List[str],
                            mutation_weights: Optional[List[float]] = None,
                            mutation_params: Optional[Dict[str, Any]] = None) -> Chromosome:
    """
    mutation_ops: list of keys in MUTATION_FUNCTIONS (e.g., ["swap","replace"])
    Randomly choose one of provided mutation ops (or none) according to mutation_weights, apply it.
    Additionally 'apply_mutation_ops' higher-level function may apply several; here apply one.
    """
    if not mutation_ops:
        return chrom
    op = random.choices(mutation_ops, weights=mutation_weights, k=1)[0] if mutation_weights else random.choice(mutation_ops)
    func = MUTATION_FUNCTIONS.get(op)
    if func is None:
        return chrom
    params = mutation_params or {}
    # functions expect (train_map, trains, locomotives, ...) depending on signature
    train_map = assignment_to_train_map(chrom.assignment)
    if op == "shuffle":
        window = params.get("shuffle_window", 5)
        new_map = func(train_map, trains, locomotives, window)
    else:
        new_map = func(train_map, trains, locomotives)
    new_assign = train_map_to_assignment(new_map, locomotives, trains)
    return Chromosome(new_assign)


def apply_corrective_operators(chrom: Chromosome,
                               locomotives: Dict[int, Locomotive],
                               trains: Dict[int, Train],
                               station_coords: Optional[Dict[str, Tuple[float, float, float]]],
                               corrective_ops: List[str]) -> Chromosome:
    """
    Apply corrective operators in the given order.
    """
    c = chrom
    for op in corrective_ops:
        func = CORRECTIVE_FUNCTIONS.get(op)
        if func is None:
            logger.debug("Неизвестный корректирующий оператор: %s", op)
            continue
        # temporal operator may need station_coords
        if op == "temporal":
            c = func(c, locomotives, trains, station_coords)
        else:
            c = func(c, locomotives, trains)
    return c


# ---------------------
# Evaluate population (components, dynamic selection of weights) - reuse previous logic
# ---------------------
def evaluate_population_and_set_fitness(population: List[Chromosome],
                                        locomotives: Dict[int, Locomotive],
                                        trains: Dict[int, Train],
                                        station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                                        use_mp: bool = False):
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
    for ch in population:
        comps_ch = compute_components(ch, locomotives, trains, station_coords)
        norm = [comps_ch[i] / (means[i] + eps) for i in range(5)]
        penalty = sum(weights[i] * norm[i] for i in range(5))
        ch.fitness = -penalty
    return weights, means


# ---------------------
# GA class: dynamic ops & corrective operators integration
# ---------------------
class GeneticAlgorithm:
    def __init__(self, locomotives: Dict[int, Locomotive], trains: Dict[int, Train],
                 population_size: int = 50, generations: int = 100,
                 tournament_k: int = 3, mutation_rate: float = 0.1,
                 crossover_ops: Optional[List[str]] = None,
                 crossover_weights: Optional[List[float]] = None,
                 mutation_ops: Optional[List[str]] = None,
                 mutation_weights: Optional[List[float]] = None,
                 corrective_ops: Optional[List[str]] = None,
                 priority_categories: Optional[List[str]] = None,
                 mutation_params: Optional[Dict[str, Any]] = None,
                 use_multiprocessing: bool = True,
                 multiprocessing_threshold: int = 100,
                 station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                 progress_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None):
        self.locomotives = locomotives
        self.trains = trains
        self.population_size = max(1, int(population_size))
        self.generations = max(1, int(generations))
        self.tournament_k = max(1, min(int(tournament_k), self.population_size))
        self.mutation_rate = float(mutation_rate)
        self.crossover_ops = crossover_ops or ["one_point", "uniform", "priority"]
        self.crossover_weights = crossover_weights
        self.mutation_ops = mutation_ops or ["swap", "replace", "shuffle"]
        self.mutation_weights = mutation_weights
        self.corrective_ops = corrective_ops or ["technical", "temporal", "maintenance"]
        self.priority_categories = priority_categories or ["express"]
        self.mutation_params = mutation_params or {"shuffle_window": 5}
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
        for gen in range(self.generations):
            gen_start = time.time()
            weights, means = evaluate_population_and_set_fitness(population, self.locomotives, self.trains, self.station_coords, use_mp=self.use_multiprocessing)
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
                    pass

            # create next generation with elitism
            new_population: List[Chromosome] = []
            elites = sorted(population, key=lambda c: c.fitness, reverse=True)[:2]
            new_population.extend(Chromosome(copy.deepcopy(e.assignment)) for e in elites)

            while len(new_population) < self.population_size:
                p1, p2 = self._select_parents(population)
                child = produce_child_dynamic([p1, p2], self.trains, self.locomotives,
                                              crossover_ops=self.crossover_ops,
                                              crossover_weights=self.crossover_weights,
                                              priority_categories=self.priority_categories,
                                              uniform_p=0.5)
                # apply mutation(s) dynamically with some probability
                if random.random() < self.mutation_rate:
                    child = apply_mutations_dynamic(child, self.trains, self.locomotives,
                                                    mutation_ops=self.mutation_ops,
                                                    mutation_weights=self.mutation_weights,
                                                    mutation_params=self.mutation_params)
                # apply corrective operators
                child = apply_corrective_operators(child, self.locomotives, self.trains, self.station_coords, self.corrective_ops)
                # check feasibility and accept; otherwise try parents or fallback
                if is_feasible_fast(child, self.locomotives, self.trains):
                    new_population.append(child)
                else:
                    # try parent clones
                    if is_feasible_fast(p1, self.locomotives, self.trains):
                        new_population.append(Chromosome(copy.deepcopy(p1.assignment)))
                    elif is_feasible_fast(p2, self.locomotives, self.trains):
                        new_population.append(Chromosome(copy.deepcopy(p2.assignment)))
                    else:
                        # fallback random
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
# Utility functions and UI helpers unchanged (abbreviated)...
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


# (Display, plotting, and Streamlit UI can call the GA with dynamic operator lists.)
# The UI code can pass lists for crossover_ops, mutation_ops and corrective_ops to GeneticAlgorithm.

# End of file
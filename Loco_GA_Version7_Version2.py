# -*- coding: utf-8 -*-
"""
Loco_GA — Genetic Algorithm for locomotive assignment

Изменения в этой версии:
 - Добавлены адаптивные операторы кроссовера: стратегия выбора оператора меняется в зависимости от
   изменений лучшей пригодности в истории популяции (стагнация -> больше исследовательских операторов,
   улучшение -> больше эксплуатационных операторов).
 - Адаптивная скорость мутации (как было ранее) сохранена.
 - Streamlit UI показывает распределение весов кроссоверов по поколениям.
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
    Параметр loco_types: словарь вида:
       {
         "typeA": {"power_range": (4000,6000), "speed_kmh": 60, "resource_range": (20,50)},
         "typeB": {...}
       }
    Если loco_types не задан — создаются 3 базовых типа.
    Возвращает locomotives, trains, station_coords.
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
# Advanced timeline metrics: idle_time, empty_time, wait times, used locos
# ---------------------
def compute_time_components_for_chromosome(chrom: Chromosome,
                                           locomotives: Dict[int, Locomotive],
                                           trains: Dict[int, Train],
                                           station_coords: Dict[str, Tuple[float, float, float]],
                                           slope_penalty_coefficient: float = 0.05) -> Dict[str, float]:
    """
    Для одной хромосомы считаем:
      - idle_time_sum: суммарное время, когда локомотив простаивал в ожидании рейса (локомотив раньше прибыл)
      - empty_time_sum: суммарное время репозиции (порожний ход), когда локомотив ехал без поезда
      - train_wait_time_sum: суммарное времени ожидания поездов (когда локомотив приходит позже отправления)
      - loco_wait_time_sum: суммарное времени ожидания локомотивов (когда локомотив прибывает раньше и ждёт отправлен[...]
      - used_locos: число локомотивов, имеющих хотя бы одно назначение
    Алгоритм: для каждого локомотива сортируем назначенные по departure_time, моделируем доступность локомотива.
    """
    idle_time_sum = 0.0
    empty_time_sum = 0.0
    train_wait_time_sum = 0.0
    loco_wait_time_sum = 0.0
    used_locos = 0

    for loco_id, train_ids in chrom.assignment.items():
        if not train_ids:
            continue
        used_locos += 1
        # sort assigned trains by departure_time (assume service order should respect time)
        sorted_trains = sorted((trains[t_id] for t_id in train_ids), key=lambda t: t.departure_time)
        # loco starts at depot and is available at time 0
        loco = locomotives[loco_id]
        loco_available_time = 0.0
        # current loco location coordinates (start from depot)
        current_loc = loco.home_depot
        for t in sorted_trains:
            dep_station = t.route[0]
            # get coords presence
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
            # loco arrives at loco_available_time + reposition_time
            loco_arrival = loco_available_time + reposition_time
            # compare with train departure
            if loco_arrival > t.departure_time:
                # train waits for loco
                wait_t = loco_arrival - t.departure_time
                train_wait_time_sum += wait_t
            else:
                # loco waits for train
                wait_l = t.departure_time - loco_arrival
                loco_wait_time_sum += wait_l
            # sum empty reposition time
            empty_time_sum += reposition_time
            # after train runs, loco becomes available at departure + duration + possible lateness
            # if train waited, actual departure = loco_arrival, else it's t.departure_time
            actual_departure = max(t.departure_time, loco_arrival)
            loco_available_time = actual_departure + t.duration
            # update current location to arrival station
            current_loc = t.route[1]
        # idle_time for this loco could be considered as time loco waited for trains (loco_wait_time)
        idle_time_sum += loco_wait_time_sum

    # normalize/return
    return {
        "idle_time_h": idle_time_sum,
        "empty_time_h": empty_time_sum,
        "train_wait_time_h": train_wait_time_sum,
        "loco_wait_time_h": loco_wait_time_sum,
        "used_locos_count": used_locos
    }


# ---------------------
# Components & dynamic weights (now for 5 components)
# ---------------------
def compute_components(chromosome: Chromosome,
                       locomotives: Dict[int, Locomotive],
                       trains: Dict[int, Train],
                       station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None) -> Tuple[float, float, float, float, float]:
    """
    Возвращает 5-компонентный вектор:
      (idle_time_h, empty_time_h, train_wait_time_h, loco_wait_time_h, used_locos_count)
    """
    if station_coords is None:
        # fallback: use simple proxies
        idle_count = sum(1 for lst in chromosome.assignment.values() if not lst)
        empty = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values())
        train_wait = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values()) * 0.5
        loco_wait = sum(max(0, len(lst) - 1) for lst in chromosome.assignment.values()) * 0.5
        used_locos = sum(1 for lst in chromosome.assignment.values() if lst)
        return float(idle_count), float(empty), float(train_wait), float(loco_wait), float(used_locos)
    # otherwise compute timeline-based metrics
    metrics = compute_time_components_for_chromosome(chromosome, locomotives, trains, station_coords)
    return (metrics["idle_time_h"], metrics["empty_time_h"], metrics["train_wait_time_h"],
            metrics["loco_wait_time_h"], float(metrics["used_locos_count"]))


def derive_dynamic_weights(population: List[Chromosome],
                           locomotives: Dict[int, Locomotive],
                           trains: Dict[int, Train],
                           station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None) -> Tuple[float, float, float, float, float]:
    """
    Для 5 компонентов считаем средние значения и задаём веса, обратно пропорциональные среднему.
    Возвращаем нормированные веса (сумма = 1).
    """
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
    Вычисляет fitness на основе 5 компонентов. Если weights не заданы, используются равные веса.
    """
    comps = compute_components(chromosome, locomotives, trains, station_coords)
    if weights is None:
        weights = (0.2, 0.2, 0.2, 0.2, 0.2)
    comp_norm = [comps[i] / (1.0 + comps[i]) for i in range(5)]
    penalty = sum(weights[i] * comp_norm[i] for i in range(5))
    fitness = -penalty
    chromosome.fitness = fitness
    return fitness


# ---------------------
# Helpers: conversion between assignment dict and gene list
# ---------------------
def ordered_train_ids(trains: Dict[int, Train]) -> List[int]:
    # consistent train order across crossovers/mutations
    return sorted(trains.keys())


def assignment_to_gene_list(assignment: Dict[int, List[int]], train_ids: List[int]) -> List[int]:
    # gene i -> loco_id assigned to train train_ids[i]
    # find loco for each train (train appears exactly once)
    gene = [None] * len(train_ids)
    train_to_index = {tid: idx for idx, tid in enumerate(train_ids)}
    for loco_id, tlist in assignment.items():
        for t in tlist:
            if t in train_to_index:
                gene[train_to_index[t]] = loco_id
    # if some trains unassigned (None) fill with random loco (shouldn't happen normally)
    for i in range(len(gene)):
        if gene[i] is None:
            # choose any loco present in assignment
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
            # if loco id not present (rare), map to random existing loco
            assign[random.choice(loco_ids)].append(tid)
        else:
            assign[loco].append(tid)
    return assign


# ---------------------
# Genetic operators (expanded)
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
    """
    Поддерживаем: "one_point", "two_point", "uniform", "priority"
    Для crossover используем представление в виде списка генов (по train order).
    priority: использует Train.category (int). Если категории присутствуют — шанс взять ген у parent1
              пропорционален относительной важности категории.
    """
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
        # priority crossover: use Train.category as weight to choose parent1
        cats = [trains[tid].category if tid in trains else 0 for tid in train_ids]
        max_cat = max(cats) if cats else 1
        for i in range(n):
            cat = cats[i]
            # probability to take from parent1 increases with category
            p1 = 0.5 if max_cat == 0 else 0.4 + 0.6 * (cat / max_cat)  # p1 in [0.4, 1.0]
            child[i] = g1[i] if random.random() < p1 else g2[i]
    else:
        # default fallback: uniform
        for i in range(n):
            child[i] = g1[i] if random.random() < 0.5 else g2[i]

    # map to assignment dict
    child_assign = gene_list_to_assignment(child, train_ids, loco_ids)
    return child_assign


def mutation_assignment(assignment: Dict[int, List[int]],
                        trains: Dict[int, Train],
                        mutation_rate: float = 0.1,
                        methods: Optional[List[str]] = None):
    """
    Поддерживаем следующие методы:
      - 'swap_locos' : перестановка локомотивов (обмен полных списков назначений между двумя локомотивами)
      - 'replace_loco' : выбор одного поезда и перераспределение его на другой локомотив
      - 'range_shuffle' : выбрать диапазон локомотивов и перемешать поезда между ними
    mutation_rate — базовая вероятность применения операции на отдельном элементе, но для некоторых
    операторов мы применяем 1 вызов с вероятностью mutation_rate.
    """
    if methods is None:
        methods = ["swap_locos", "replace_loco", "range_shuffle"]

    loco_ids = list(assignment.keys())
    if not loco_ids:
        return

    # Apply swap_locos: with probability mutation_rate perform one swap of two locomotives' assignments
    if "swap_locos" in methods and random.random() < mutation_rate:
        if len(loco_ids) >= 2:
            a, b = random.sample(loco_ids, 2)
            assignment[a], assignment[b] = assignment[b], assignment[a]

    # Apply replace_loco: iterate trains and with small probability move to another loco
    if "replace_loco" in methods:
        # flatten trains
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

    # Apply range_shuffle: with probability mutation_rate choose a random contiguous range of loco_ids
    # (by index in loco_ids list) and shuffle all trains between them
    if "range_shuffle" in methods and random.random() < mutation_rate:
        if len(loco_ids) >= 2:
            idxs = sorted(random.sample(range(len(loco_ids)), min(len(loco_ids), max(2, random.randint(2, len(loco_ids)))) ))
            a = idxs[0]
            b = idxs[-1]
            selected_locos = loco_ids[a:b+1]
            # collect trains
            collected = []
            for lid in selected_locos:
                collected.extend(assignment[lid])
                assignment[lid] = []
            random.shuffle(collected)
            # redistribute roughly evenly
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
    # normalized components by means (avoid division by zero)
    comps = compute_components(chrom, locomotives, trains, station_coords)
    # normalized comp = comp / (mean + eps)
    eps = 1e-6
    norm = [comps[i] / (means[i] + eps) for i in range(5)]
    # penalty = sum weights * norm
    penalty = sum(weights[i] * norm[i] for i in range(5))
    chrom.fitness = -penalty
    return chrom.assignment, chrom.fitness


def _child_worker_serial(args):
    """
    args:
      assign_p1, assign_p2, locomotives, trains, crossover_method, mutation_rate, mutation_methods
    """
    assign_p1, assign_p2, locomotives, trains, crossover_method, mutation_rate, mutation_methods = args
    child_assign = crossover_assignments(assign_p1, assign_p2, trains, method=crossover_method)
    mutation_assignment(child_assign, trains, mutation_rate=mutation_rate, methods=mutation_methods)
    child = Chromosome(child_assign)
    if is_feasible_fast(child, locomotives, trains):
        return child_assign
    return None


# ---------------------
# GeneticAlgorithm with dynamic weights + per-generation stats
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
                 crossover_method: str = "uniform",
                 mutation_methods: Optional[List[str]] = None,
                 adaptive_mutation: bool = True,
                 min_mutation_rate: float = 0.01,
                 max_mutation_rate: float = 0.5):
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
        self.crossover_method = crossover_method
        self.mutation_methods = mutation_methods or ["swap_locos", "replace_loco", "range_shuffle"]
        self.adaptive_mutation = adaptive_mutation
        self.min_mutation_rate = float(min_mutation_rate)
        self.max_mutation_rate = float(max_mutation_rate)

        # adaptive tracking
        self.best_history: List[float] = []
        self.no_improve_generations = 0

        # Crossover adaptation: maintain methods and weights
        self.crossover_methods = ["one_point", "two_point", "uniform", "priority"]
        # initial uniform weights
        self.crossover_weights = [1.0 for _ in self.crossover_methods]

    def _evaluate_population_dynamic(self, population: List[Chromosome]) -> Dict[str, Any]:
        """
        1) compute components per individual
        2) compute means
        3) derive weights (inversely proportional to means)
        4) compute normalized penalty per individual and set fitness = -penalty
        5) return stats: best/mean/std/min, weights, means_of_components
        """
        comps = [compute_components(ch, self.locomotives, self.trains, self.station_coords) for ch in population]
        # means per component
        if comps:
            means = [statistics.mean([c[i] for c in comps]) for i in range(5)]
        else:
            means = [0.0] * 5
        # derive weights from means
        eps = 1e-6
        raw = [1.0 / (m + eps) for m in means]
        s = sum(raw)
        if s <= 0:
            weights = tuple(self.initial_weights[:5])
        else:
            weights = tuple(r / s for r in raw)

        # Evaluate fitness (parallel if useful)
        if self.use_multiprocessing and len(population) >= self.multiprocessing_threshold and self.cpu_count > 1:
            args = [(ch.assignment, self.locomotives, self.trains, self.station_coords, weights, means) for ch in population]
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(args))) as exc:
                for assignment, fitness in exc.map(_fitness_worker_serial, args):
                    for ch in population:
                        if ch.assignment == assignment:
                            ch.fitness = fitness
                            break
        else:
            # serial evaluation
            for ch in population:
                comps_ch = compute_components(ch, self.locomotives, self.trains, self.station_coords)
                norm = [comps_ch[i] / (means[i] + eps) for i in range(5)]
                penalty = sum(weights[i] * norm[i] for i in range(5))
                ch.fitness = -penalty

        fitnesses = [ch.fitness for ch in population]
        best = max(fitnesses) if fitnesses else float("-inf")
        mean_f = statistics.mean(fitnesses) if fitnesses else float("nan")
        std_f = statistics.pstdev(fitnesses) if fitnesses else float("nan")
        mn = min(fitnesses) if fitnesses else float("inf")

        stats = {
            "best": best,
            "mean": mean_f,
            "std": std_f,
            "min": mn,
            "weights": weights,
            "means_components": means
        }
        return stats

    def _adapt_mutation_rate(self, current_best: float):
        """
        Простая адаптивная логика:
         - если best улучшился (на eps) — уменьшаем mutation_rate (��сследование не требуется так сильно)
         - если не улучшился — увеличиваем mutation_rate (чтобы выходить из лок. минимума)
         - ограничиваем между min_mutation_rate и max_mutation_rate
        """
        eps = 1e-8
        if not self.best_history:
            self.best_history.append(current_best)
            return
        prev_best = self.best_history[-1]
        if current_best > prev_best + eps:
            # улучшение — снизить мутацию
            self.no_improve_generations = 0
            self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.9)
        else:
            # нет улучшения — увеличить мутацию
            self.no_improve_generations += 1
            # more stagnation -> stronger increase
            factor = 1.0 + 0.05 * min(self.no_improve_generations, 10)
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * factor)
        # store history
        self.best_history.append(current_best)

    def _adapt_crossover_weights(self, current_best: float):
        """
        Адаптируем веса операторов кроссовера на основе улучшения best:
        - при улучшении: повышаем вероятность эксплуатационных операторов (one_point, priority)
        - при стагнации: повышаем вероятность исследовател��ских операторов (uniform, two_point)
        Поддерживаем нормировку и ограничения.
        """
        # define which are exploitation vs exploration
        exploitation = {"one_point", "priority"}
        exploration = {"uniform", "two_point"}
        # map methods to indices
        idx_map = {m: i for i, m in enumerate(self.crossover_methods)}

        # compute improvement
        eps = 1e-8
        if not self.best_history:
            self.best_history.append(current_best)
            return
        prev_best = self.best_history[-1]
        improved = current_best > prev_best + eps

        # adapt step
        step = 0.15  # how strongly weights change per generation of improvement/stagnation
        min_w = 0.05
        # adjust
        for method in self.crossover_methods:
            i = idx_map[method]
            if improved:
                # reward exploitation, penalize exploration
                if method in exploitation:
                    self.crossover_weights[i] = self.crossover_weights[i] * (1.0 + step)
                else:
                    self.crossover_weights[i] = max(min_w, self.crossover_weights[i] * (1.0 - step))
            else:
                # stagnation -> reward exploration
                if method in exploration:
                    self.crossover_weights[i] = self.crossover_weights[i] * (1.0 + step)
                else:
                    self.crossover_weights[i] = max(min_w, self.crossover_weights[i] * (1.0 - step))
        # normalize to avoid overflow or diminishment
        tot = sum(self.crossover_weights)
        if tot <= 0:
            # reset to uniform
            self.crossover_weights = [1.0 for _ in self.crossover_methods]
        else:
            self.crossover_weights = [w / tot for w in self.crossover_weights]

    def _generate_children(self, population: List[Chromosome], target_count: int) -> List[Chromosome]:
        children: List[Chromosome] = []
        # current mutation_rate (may be adapted externally)
        curr_mutation_rate = self.mutation_rate if self.adaptive_mutation else self.initial_mutation_rate

        # For logging how many times each crossover method used this generation
        crossover_usage = {m: 0 for m in self.crossover_methods}

        if self.use_multiprocessing and len(population) >= self.multiprocessing_threshold and self.cpu_count > 1:
            parent_pairs = []
            # produce more candidates than needed to increase chance of feasible children
            for _ in range(target_count * 3):
                p1 = random.choice(population)
                p2 = random.choice(population)
                # sample crossover method according to adaptive weights
                method = random.choices(self.crossover_methods, weights=self.crossover_weights, k=1)[0]
                crossover_usage[method] += 1
                parent_pairs.append((p1.assignment, p2.assignment, self.locomotives, self.trains,
                                     method, curr_mutation_rate, self.mutation_methods))
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(parent_pairs))) as exc:
                for result in exc.map(_child_worker_serial, parent_pairs):
                    if result is not None:
                        children.append(Chromosome(result))
                    if len(children) >= target_count:
                        break
        else:
            attempts = 0
            while len(children) < target_count and attempts < target_count * 50:
                attempts += 1
                p1 = tournament_selection(population, self.tournament_k)
                p2 = tournament_selection(population, self.tournament_k)
                method = random.choices(self.crossover_methods, weights=self.crossover_weights, k=1)[0]
                crossover_usage[method] += 1
                child_assign = crossover_assignments(p1.assignment, p2.assignment, self.trains, method=method)
                mutation_assignment(child_assign, self.trains, mutation_rate=curr_mutation_rate, methods=self.mutation_methods)
                child = Chromosome(child_assign)
                if is_feasible_fast(child, self.locomotives, self.trains):
                    children.append(child)

        # attach last usage stats for external inspection if needed
        # convert to normalized distribution
        total_used = sum(crossover_usage.values())
        if total_used > 0:
            usage_norm = {k: v / total_used for k, v in crossover_usage.items()}
        else:
            usage_norm = {k: 0.0 for k in crossover_usage.keys()}
        # store as attribute for retrieval (last generation)
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
            generation_stats.append(stats)

            # adapt mutation rate based on best history
            if self.adaptive_mutation:
                self._adapt_mutation_rate(stats["best"])

            # adapt crossover weights based on improvement/stagnation
            self._adapt_crossover_weights(stats["best"])

            # record current crossover weights in stats
            stats["crossover_weights"] = tuple(self.crossover_weights)

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

            # include observed crossover usage if available
            if hasattr(self, "last_crossover_usage"):
                generation_stats[-1]["crossover_usage"] = self.last_crossover_usage

            new_population = children
            if len(new_population) < self.population_size:
                elites = sorted(population, key=lambda c: c.fitness, reverse=True)
                i = 0
                while len(new_population) < self.population_size:
                    e = elites[i % len(elites)]
                    new_population.append(Chromosome(copy.deepcopy(e.assignment)))
                    i += 1
            population = new_population

        # final evaluation & total time
        final_start = time.time()
        final_stats = self._evaluate_population_dynamic(population)
        final_eval_time = time.time() - final_start
        final_stats["time_sec"] = round(final_eval_time, 4)
        final_stats["gen"] = self.generations
        generation_stats.append(final_stats)
        total_time = time.time() - total_start
        self.generation_stats = generation_stats
        self.total_time_sec = round(total_time, 4)
        best = max(population, key=lambda c: c.fitness)
        logger.info("GA завершён: лучшая пригодность = %.6f; время, с: %.4f", best.fitness, self.total_time_sec)
        return best


# ---------------------
# Reporting & DataFrames (Russian labels)
# ---------------------
class GAReporter:
    def __init__(self):
        self.generation_log: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def log_generation(self, gen: int, stats: Dict[str, Any]):
        # stats может содержать: best,mean,std,min,weights,means_components,time_sec,mutation_rate,crossover_weights,crossover_usage
        entry = {
            "поколение": gen,
            "best": stats.get("best"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "weights": stats.get("weights"),
            "means_components": stats.get("means_components"),
            "time_sec": stats.get("time_sec"),
            "mutation_rate": stats.get("mutation_rate"),
            "crossover_weights": stats.get("crossover_weights"),
            "crossover_usage": stats.get("crossover_usage", {})
        }
        self.generation_log.append(entry)

    def elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def print_summary(self, solution: Chromosome, total_time_sec: Optional[float] = None):
        print("\n=== Время расчёта и итоговая пригодность ===")
        print(f"Время расчёта, с: {total_time_sec if total_time_sec is not None else self.elapsed():.2f}")
        print(f"Итоговая целевая функция (fitness): {solution.fitness:.6f}")


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
# Plotting helpers (with Russian labels and legends)
# ---------------------
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
    cw = [e.get("crossover_weights", (0, 0, 0, 0)) for e in reporter.generation_log]
    methods = ["one_point", "two_point", "uniform", "priority"]
    fig, ax = plt.subplots(figsize=(9, 3))
    for i, m in enumerate(methods):
        ax.plot(gens, [w[i] for w in cw], label=m, marker="o")
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
# Sensitivity analysis (по весам) — быстрый режим, русские метки
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
    """
    Быстрый анализ чувствительности: для каждой дельты вариируем веса и запускаем GA.
    Возвращаем DataFrame с колонками: delta, w1..w5, fitness, time_sec
    """
    rows = []
    for d in deltas:
        # распределим d по первым двум весам и компенсируем остальное
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
# Streamlit UI (русский) с новыми метриками и графиками
# ---------------------
def parse_locomotives_csv(file_buf: io.BytesIO) -> Dict[int, Locomotive]:
    try:
        df = pd.read_csv(file_buf)
    except Exception:
        df = pd.read_excel(file_buf)
    # try to interpret columns
    required = {"id", "loco_type", "power", "remaining_resource", "home_depot"}
    df_cols = set(c.lower() for c in df.columns)
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
        # route might be two columns or single "route" like "A-B"
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

    # data loaders
    st.sidebar.markdown("### Загрузка данных")
    loco_file = st.sidebar.file_uploader("CSV/Excel: Локомотивы (колонки id,loco_type,power,remaining_resource,home_depot,reposition_speed_kmh)", type=["csv", "xls", "xlsx"])
    trains_file = st.sidebar.file_uploader("CSV/Excel: Поезда (id,weight,route_from,route_to,departure_time,duration,category)", type=["csv", "xls", "xlsx"])
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

    st.sidebar.markdown("### Операторы")
    crossover_method = st.sidebar.selectbox("Кроссовер (начальный)", options=["uniform", "one_point", "two_point", "priority"], index=0)
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

    st.sidebar.markdown("### Типы локомотивов (пример)")
    st.sidebar.markdown("По умолчанию используются 3 типа: 2ЭС6, ЧС7, ТЭП70.")
    use_mp = st.sidebar.checkbox("Использовать multiprocessing для тяжёлых задач", value=True)
    mp_threshold = st.sidebar.number_input("Порог для multiprocessing (размер популяции)", min_value=2, max_value=500, value=100)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Анализ чувствительности (быстрый)")
    sens_deltas_str = st.sidebar.text_input("Дельты (через запятую)", "-0.2,-0.1,0,0.1,0.2")
    sens_deltas = [float(x.strip()) for x in sens_deltas_str.split(",") if x.strip()]
    sens_pop = st.sidebar.number_input("Популяция (анализ чувствительности)", min_value=5, max_value=500, value=30)
    sens_gens = st.sidebar.number_input("Поколений (анализ чувствительности)", min_value=1, max_value=200, value=15)
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

    # If uploaded files present and user chooses to use them, parse them
    if use_uploaded and loco_file is not None:
        try:
            locomotives = parse_locomotives_csv(loco_file)
            st.session_state["locomotives"] = locomotives
            st.success("Локомотивы загружены")
        except Exception as e:
            st.error(f"Не удалось распарсить фай�� локомотивов: {e}")

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

    st.sidebar.markdown(f"Локомотивов: {len(locomotives)}")
    st.sidebar.markdown(f"Поездов: {len(trains)}")
    st.sidebar.markdown(f"Станций: {len(station_coords)}")

    # layout
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
            best = stats.get("best")
            mean = stats.get("mean")
            time_s = stats.get("time_sec", 0.0)
            mut = stats.get("mutation_rate", 0.0)
            gen_stats_placeholder.metric(label=f"Поколение {gen}", value=f"Лучшее: {best:.6f}", delta=f"Среднее: {mean:.6f}")
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
                crossover_method=crossover_method,
                mutation_methods=mutation_methods,
                adaptive_mutation=bool(adaptive_mutation),
                min_mutation_rate=float(min_mut),
                max_mutation_rate=float(max_mut)
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

    # sensitivity
    if run_sens_button:
        with st.spinner("Запуск анализа чувствительности..."):
            df_sens = sensitivity_analysis(locomotives, trains, station_coords,
                                           base_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
                                           deltas=sens_deltas,
                                           population_size=int(sens_pop),
                                           generations=int(sens_gens),
                                           tournament_k=int(tournament_k),
                                           mutation_rate=float(mutation_rate),
                                           use_multiprocessing=False)
            st.session_state["sens_df"] = df_sens
        st.success("Анализ чувствительности завершён")

    # отображение результатов
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

        # generation curve
        fig_gc = plot_generation_curve(reporter)
        evolution_placeholder.pyplot(fig_gc)

        # weights evolution
        fig_w = plot_weights_evolution(reporter)
        weights_plot_placeholder.pyplot(fig_w)

        # crossover weights evolution
        fig_cw = plot_crossover_weights_evolution(reporter)
        crossover_weights_placeholder.pyplot(fig_cw)

        # components evolution
        fig_c = plot_components_evolution(reporter)
        components_placeholder.pyplot(fig_c)

        total_time = st.session_state.get("total_time", None)
        time_summary_placeholder.markdown(f"**Общее время расчёта:** {total_time:.4f} с")
        time_summary_placeholder.markdown(f"**Итоговая пригодность (fitness):** {solution.fitness:.6f}")

    if "sens_df" in st.session_state:
        df_sens: pd.DataFrame = st.session_state["sens_df"]
        sens_placeholder.subheader("Результаты анализа чувствительности")
        sens_placeholder.dataframe(df_sens, height=300)
        # график
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
    reporter.print_summary(best, total_time_sec=getattr(ga, "total_time_sec", None))
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
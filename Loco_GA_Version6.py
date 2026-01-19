# -*- coding: utf-8 -*-
"""
Loco_GA — Genetic Algorithm for locomotive assignment

Добавлены:
 - адаптивные веса критериев (динамически вычисляются на каждом поколении)
 - сбор и отображение статистики по поколениям: best/mean/min/std, использованные веса, время генерации
 - итого��ый анализ: общее время работы GA и итоговая пригодность
 - анализ чувствительности целевой функции к весам (быстрый режим)
 - все подписи в интерфейсе и на графиках — на русском языке, добавлены легенды
 - улучшенный вывод таблиц в Streamlit (pandas.DataFrame, русские заголовки)
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

    logger.info("Сгенерированы синтетические данные: %d локомотивов, %d поездов, %d станций",
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
# Components and dynamic weights
# ---------------------
def calculate_train_mass(chromosome: Chromosome, trains: Dict[int, Train]) -> float:
    return sum(trains[t_id].weight for train_ids in chromosome.assignment.values() for t_id in train_ids)


def compute_components(chromosome: Chromosome,
                       locomotives: Dict[int, Locomotive],
                       trains: Dict[int, Train],
                       station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                       reposition_speed_kmh: float = 60.0,
                       slope_penalty_coefficient: float = 0.05) -> Tuple[float, float, float]:
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
    eps = 1e-6
    comp_list = []
    for chrom in population:
        idle, empty_time, mass = compute_components(chrom, locomotives, trains, station_coords)
        comp_list.append((idle, empty_time, mass))

    if not comp_list:
        return (0.33, 0.33, 0.34)

    means = [statistics.mean([c[i] for c in comp_list]) for i in range(3)]
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
        logger.warning("Начальная допустимая популяция не найдена случайно — использован fallback")
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
# Multiprocessing workers
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
# GeneticAlgorithm with dynamic weights and timing
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
        comps = []
        for chrom in population:
            idle, empty_time, mass = compute_components(chrom, self.locomotives, self.trains, self.station_coords)
            comps.append((idle, empty_time, mass))

        weights = derive_dynamic_weights(population, self.locomotives, self.trains, self.station_coords)

        # Evaluate fitness (parallel if beneficial)
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
        total_start = time.time()
        for gen in range(self.generations):
            gen_start = time.time()
            stats = self._evaluate_population_with_dynamic_weights(population)
            gen_eval_time = time.time() - gen_start

            # add timing to stats
            stats["time_sec"] = round(gen_eval_time, 4)
            generation_stats.append({"gen": gen, **stats})

            if self.progress_callback:
                try:
                    # progress_callback expects (gen, stats_dict)
                    self.progress_callback(gen, stats)
                except Exception:
                    try:
                        self.progress_callback(gen, stats["best"])
                    except Exception:
                        pass

            # children generation also consumes time; measure it too
            child_start = time.time()
            children = self._generate_children(population, self.population_size)
            child_time = time.time() - child_start
            # include child_time in last generation record (augment)
            generation_stats[-1]["children_time_sec"] = round(child_time, 4)

            new_population = children
            if len(new_population) < self.population_size:
                elites = sorted(population, key=lambda c: c.fitness, reverse=True)
                i = 0
                while len(new_population) < self.population_size:
                    e = elites[i % len(elites)]
                    new_population.append(Chromosome(copy.deepcopy(e.assignment)))
                    i += 1
            population = new_population

        # final evaluation and total time
        final_start = time.time()
        final_stats = self._evaluate_population_with_dynamic_weights(population)
        final_eval_time = time.time() - final_start
        final_stats["time_sec"] = round(final_eval_time, 4)
        generation_stats.append({"gen": self.generations, **final_stats})
        total_time = time.time() - total_start
        self.generation_stats = generation_stats
        self.total_time_sec = round(total_time, 4)
        best = max(population, key=lambda c: c.fitness)
        logger.info("GA завершён: лучшая пригодность = %.6f; время, с: %.4f", best.fitness, self.total_time_sec)
        return best


# ---------------------
# Reporting & DataFrames for nicer tables (Russian)
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
            "best": stats.get("best"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "weights": stats.get("weights"),
            "time_sec": stats.get("time_sec")
        }
        self.generation_log.append(entry)

    def elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def print_summary(self, solution: Chromosome, total_time_sec: Optional[float] = None):
        print("\n=== 1. Время расчёта и итоговая пригодность ===")
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
        "Локомотив", "Депо", "Остаток_ресурса_ч", "Поезд", "Откуда", "Куда",
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
            "Депо": loco.home_depot,
            "Текущее_место": current_location,
            "Остаток_ресурса_ч": round(remaining_resource, 2),
            "Назначено_поездов": len(train_ids)
        })
    df = pd.DataFrame(rows, columns=["Локомотив", "Депо", "Текущее_место", "Остаток_ресурса_ч", "Назначено_поездов"])
    return df


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

    # вторичная ось: время на поколение (если есть)
    if "time_sec" in df.columns:
        ax2 = ax.twinx()
        ax2.bar(df["поколение"], df["time_sec"], color="gray", alpha=0.25, label="Время на поколение (с)")
        ax2.set_ylabel("Время на поколение, с")
        # объединённая легенда
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
    w_mass = [w[2] for w in ws]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(gens, w_idle, label="Вес простоя (idle)", marker="o")
    ax.plot(gens, w_empty, label="Вес порожнего пробега (empty)", marker="o")
    ax.plot(gens, w_mass, label="Вес массы поездов (mass)", marker="o")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Вес критерия")
    ax.set_title("Динамика весов критериев по поколениям")
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
    # легенда: показываем поезда отдельно, если есть метки
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    return fig


# ---------------------
# Анализ чувствительности к весам
# ---------------------
def sensitivity_analysis(locomotives: Dict[int, Locomotive],
                         trains: Dict[int, Train],
                         station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
                         base_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
                         deltas: List[float] = (-0.2, -0.1, 0.0, 0.1, 0.2),
                         population_size: int = 40,
                         generations: int = 30,
                         tournament_k: int = 3,
                         mutation_rate: float = 0.1,
                         use_multiprocessing: bool = False) -> pd.DataFrame:
    """
    Быстрый анализ чувствительности: для набора дельт модифицируем базовые веса и запускаем GA.
    Возвращаем pandas.DataFrame с колонками: delta, w_idle, w_empty, w_mass, fitness, time_sec
    """
    rows = []
    for d in deltas:
        # простой способ варьирования: прибавляем d к idle и empty, компенсируем mass
        w_idle = base_weights[0] + d
        w_empty = base_weights[1] + d
        w_mass = base_weights[2] - 2 * d
        # нормируем и корректируем отрицательные
        ws_raw = [max(0.0, w_idle), max(0.0, w_empty), max(0.0, w_mass)]
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
            "delta": d,
            "w_idle": round(ws[0], 3),
            "w_empty": round(ws[1], 3),
            "w_mass": round(ws[2], 3),
            "fitness": fitness_val,
            "time_sec": round(elapsed, 4)
        })
    df = pd.DataFrame(rows)
    return df


# ---------------------
# Streamlit UI (русский) с расширенными выводами
# ---------------------
def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit не установлен в окружении. Установите streamlit для запуска UI.")
        return

    st.set_page_config(page_title="Loco_GA", layout="wide")
    st.title("Loco_GA — Назначение локомотивов (Генетический алгоритм)")
    st.sidebar.header("Параметры")

    num_locomotives = st.sidebar.slider("Число локомотивов", 1, 200, 10)
    num_trains = st.sidebar.slider("Число поездов", 1, 500, 20)
    depot_names_str = st.sidebar.text_input("Станции/депо (через запятую)", "A,B,C")
    depot_names = tuple(s.strip() for s in depot_names_str.split(",") if s.strip())
    seed = st.sidebar.number_input("Random seed (0 = произвольно)", min_value=0, value=0, step=1)
    if seed == 0:
        seed = None

    population_size = st.sidebar.number_input("Размер популяции", min_value=2, max_value=2000, value=60)
    generations = st.sidebar.number_input("Поколений", min_value=1, max_value=2000, value=60)
    tournament_k = st.sidebar.number_input("Размер турнира", min_value=1, max_value=population_size, value=5)
    mutation_rate = st.sidebar.slider("Вероятность му��ации", 0.0, 1.0, 0.15, 0.01)

    st.sidebar.markdown("### Начальные веса (будут адаптироваться)")
    weights_idle = st.sidebar.slider("Вес: простой (idle)", 0.0, 1.0, 0.4, 0.05)
    weights_empty = st.sidebar.slider("Вес: порожний пробег (empty)", 0.0, 1.0, 0.3, 0.05)
    weights_mass = st.sidebar.slider("Вес: масса поездов (mass)", 0.0, 1.0, 0.3, 0.05)
    w_sum = max(1e-6, weights_idle + weights_empty + weights_mass)
    init_weights = (weights_idle / w_sum, weights_empty / w_sum, weights_mass / w_sum)

    use_mp = st.sidebar.checkbox("Использовать multiprocessing для тяжёлых задач", value=True)
    mp_threshold = st.sidebar.number_input("Порог для multiprocessing (размер популяции)", min_value=2, max_value=500, value=100)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Анализ чувствительности")
    sens_deltas_str = st.sidebar.text_input("Дельты (через запятую)", "-0.2,-0.1,0,0.1,0.2")
    sens_deltas = [float(x.strip()) for x in sens_deltas_str.split(",") if x.strip()]
    sens_pop = st.sidebar.number_input("Популяция (для анализа чувствите��ьности)", min_value=5, max_value=500, value=30)
    sens_gens = st.sidebar.number_input("Поколений (для анализа чувствительности)", min_value=1, max_value=200, value=20)
    run_sens_button = st.sidebar.button("Запустить анализ чувствительности (быстрый)")

    # Генерация данных
    if st.button("Сгенерировать данные"):
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

    # Main layout
    left, right = st.columns((2, 1))
    with right:
        st.markdown("### Статистика по поколению")
        gen_stats_placeholder = st.empty()
        st.markdown("### Динамика весов")
        weights_plot_placeholder = st.empty()
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
        st.markdown("### Результаты анализа чувствительности")
        sens_placeholder = st.empty()

    reporter = GAReporter()
    reporter.start()

    run_button = st.button("Запустить GA (синхронно)")

    if run_button:
        def progress_callback(gen: int, stats: Dict[str, Any]):
            # Показ краткой статистики по поколению в правой панели
            best = stats.get("best")
            mean = stats.get("mean")
            time_s = stats.get("time_sec", 0.0)
            gen_stats_placeholder.metric(label=f"Поколение {gen}", value=f"Лучшее: {best:.6f}", delta=f"Среднее: {mean:.6f}")
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
            with st.spinner("Запуск GA... подождите, процесс может быть длительным"):
                t0 = time.time()
                best = ga.run()
                total_time = time.time() - t0
            st.success("GA завершён")
            st.session_state["solution"] = best
            st.session_state["reporter"] = reporter
            # сохранить также статистику из GA
            st.session_state["generation_stats"] = ga.generation_stats
            st.session_state["total_time"] = ga.total_time_sec if hasattr(ga, "total_time_sec") else round(total_time, 4)
        except Exception as e:
            logger.exception("Ошибка при запуске GA: %s", e)
            st.error(f"Ошибка GA: {e}")

    # Sensitivity analysis
    if run_sens_button:
        with st.spinner("Запуск анализа чувствительности..."):
            df_sens = sensitivity_analysis(locomotives, trains, station_coords,
                                           base_weights=init_weights,
                                           deltas=sens_deltas,
                                           population_size=int(sens_pop),
                                           generations=int(sens_gens),
                                           tournament_k=int(tournament_k),
                                           mutation_rate=float(mutation_rate),
                                           use_multiprocessing=False)
            st.session_state["sens_df"] = df_sens
        st.success("Анализ чувствительности завершён")

    # Отображение результатов если есть решение
    if "solution" in st.session_state:
        solution: Chromosome = st.session_state["solution"]
        reporter: GAReporter = st.session_state.get("reporter", reporter)
        gen_stats = st.session_state.get("generation_stats", [])

        # Таблица назначений
        df_assign = build_assignment_dataframe(solution, locomotives, trains)
        assignment_placeholder.dataframe(df_assign, height=350)

        # Сводная таблица локомотивов
        df_loco = build_locomotive_summary_dataframe(locomotives, trains, solution)
        loco_summary_placeholder.table(df_loco)

        # Временная диаграмма назначений
        fig_timeline = plot_assignment_matplotlib(solution, trains)
        timeline_placeholder.pyplot(fig_timeline)

        # Кривая эволюции
        evolution_fig = plot_generation_curve(reporter)
        evolution_placeholder.pyplot(evolution_fig)

        # Динамика весов
        weights_fig = plot_weights_evolution(reporter)
        weights_plot_placeholder.pyplot(weights_fig)

        # Время и итоговая пригодность
        total_time = st.session_state.get("total_time", None)
        time_summary_placeholder.markdown(f"**Общее время расчёта:** {total_time:.4f} с")
        time_summary_placeholder.markdown(f"**Итоговая пригодность (fitness):** {solution.fitness:.6f}")

    # Отображение результатов анализа чувствительности
    if "sens_df" in st.session_state:
        df_sens: pd.DataFrame = st.session_state["sens_df"]
        sens_placeholder.subheader("Результаты анализа чувствительности")
        sens_placeholder.dataframe(df_sens, height=300)
        # график чувствительности: fitness vs delta
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df_sens["delta"], df_sens["fitness"], marker="o", label="fitness")
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
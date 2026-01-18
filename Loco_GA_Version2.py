# -*- coding: utf-8 -*-
"""
Loco_GA — Genetic Algorithm for locomotive assignment
Features:
 - Streamlit UI for interactive runs
 - Performance-oriented feasibility checks and caching
 - Improved empty-run model using station coordinates (distance-based)
 - Logging to file and stdout
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import copy
import time
import math
import logging
import threading

# optional imports (matplotlib for plotting)
import matplotlib.pyplot as plt

# Try to import streamlit, otherwise provide a CLI fallback
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# ---------------------
# Logging configuration
# ---------------------
logger = logging.getLogger("Loco_GA")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)
# File handler
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
    power: float                 # тяговая мощность (условные единицы)
    remaining_resource: float    # ресурс до ТО (часы)
    home_depot: str

@dataclass
class Train:
    id: int
    weight: float
    route: Tuple[str, str]       # (откуда, куда)
    departure_time: float
    duration: float


# ---------------------
# Utilities & Caching
# ---------------------
def compute_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------
# Synthetic data generator
# ---------------------
def generate_synthetic_data(
        num_locomotives: int = 10,
        num_trains: int = 20,
        depot_names: Tuple[str, ...] = ("A", "B", "C"),
        station_coords: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None):
    """
    Генерация синтетических данных и координат станций.

    Возвращает:
      locomotives: Dict[id, Locomotive]
      trains: Dict[id, Train]
      coords: Dict[station, (x,y)]
    """
    if seed is not None:
        random.seed(seed)

    # If not provided, place depots/stations on a circle to avoid degeneracy
    if station_coords is None:
        station_coords = {}
        n = len(depot_names)
        radius = max(5, n) * 10.0
        for i, name in enumerate(depot_names):
            angle = 2 * math.pi * i / max(1, n)
            station_coords[name] = (radius * math.cos(angle), radius * math.sin(angle))

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
# Chromosome & helpers
# ---------------------
class Chromosome:
    def __init__(self, assignment: Dict[int, List[int]]):
        """
        assignment:
          key: locomotive id
          value: list of train ids in service order
        """
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
# Performance-optimized feasibility & metrics
# ---------------------
def build_lookup_tables(trains: Dict[int, Train]):
    """
    Build dictionaries for very fast lookup (used by many hot paths).
    """
    train_weight = {tid: t.weight for tid, t in trains.items()}
    train_duration = {tid: t.duration for tid, t in trains.items()}
    train_dep_station = {tid: t.route[0] for tid, t in trains.items()}
    train_arr_station = {tid: t.route[1] for tid, t in trains.items()}
    train_departure = {tid: t.departure_time for tid, t in trains.items()}
    return {
        "weight": train_weight,
        "duration": train_duration,
        "dep": train_dep_station,
        "arr": train_arr_station,
        "dep_time": train_departure
    }


def is_feasible_fast(chromosome: Chromosome,
                     locomotives: Dict[int, Locomotive],
                     trains: Dict[int, Train]) -> bool:
    """
    Быстрая проверка ограничений без создания лишних объектов:
      - тяга: локомотив способен тянуть каждый назначенный поезд
      - ресурс: суммарная длительность назначений <= remaining_resource
    """
    lookup = build_lookup_tables(trains)
    weight = lookup["weight"]
    duration = lookup["duration"]

    for loco_id, train_ids in chromosome.assignment.items():
        loco = locomotives.get(loco_id)
        if loco is None:
            logger.debug("Locomotive %s missing in locomotives", loco_id)
            return False

        # Check traction quickly
        for tid in train_ids:
            if weight[tid] > loco.power:
                logger.debug("Traction fail: loco %s power %.1f < train %s weight %.1f",
                             loco_id, loco.power, tid, weight[tid])
                return False

        # Check resource: sum durations
        total_dur = 0.0
        for tid in train_ids:
            total_dur += duration[tid]
            if total_dur > loco.remaining_resource:
                logger.debug("Resource fail: loco %s total_dur %.1f > remaining %.1f",
                             loco_id, total_dur, loco.remaining_resource)
                return False

    return True


# ---------------------
# Empty-run: more realistic model
# ---------------------
def calculate_empty_run_distance(chromosome: Chromosome,
                                 trains: Dict[int, Train],
                                 station_coords: Dict[str, Tuple[float, float]],
                                 reposition_speed_kmh: float = 60.0) -> float:
    """
    Рассчитывает суммарную дистанцию (или время) порожних пробегов для каждого локомотива.
    Подход:
      - для каждого локомотива берём его депо (если задано в locomotives) либо считаем 0
      - считаем дистанцию от депо до первой станции отправления первого поезда (если есть)
      - между последовательными рейсами считаем дистанцию от станции назначения предыдущего до станции отправления следующего
      - дистанцию из последнего пункта в депо не считаем (не требуется)
    Возвращаем суммарную дистанцию (в км).
    (Координаты принимаются в условных единицах; можно нормализовать)
    """
    total_distance = 0.0
    # Build quick maps
    train_dep = {tid: t.route[0] for tid, t in trains.items()}
    train_arr = {tid: t.route[1] for tid, t in trains.items()}

    # We need locomotives' home_depot; caller must provide access or closure.
    # To keep function generic, caller will compute per-loco using passed mapping in chromosome.context.
    # Here we expect each train's station to exist in station_coords.
    for loco_id, train_ids in chromosome.assignment.items():
        if not train_ids:
            continue
        # first train
        first = train_ids[0]
        first_dep_station = train_dep[first]
        # If loco home depot not passed here, try to retrieve from trains or set (0,0)
        # We'll assume caller uses locomotives mapping to compute start coords when needed.
        # For now, compute intra-service repositioning
        # distance from prev arrival to next departure
        prev_arr = train_arr[first]
        # But we will compute depot->first_dep if chromosome has attribute start_depot_map
        start_depot = getattr(chromosome, "start_depot_map", {}).get(loco_id)
        if start_depot is not None and start_depot in station_coords:
            total_distance += compute_distance(station_coords[start_depot], station_coords[first_dep_station])
        # between successive
        for a, b in zip(train_ids, train_ids[1:]):
            prev = train_arr[a]
            nxt = train_dep[b]
            total_distance += compute_distance(station_coords[prev], station_coords[nxt])
    return total_distance


# ---------------------
# Fitness function (uses distance-based empty runs)
# ---------------------
def calculate_train_mass(chromosome: Chromosome, trains: Dict[int, Train]) -> float:
    return sum(trains[t_id].weight for train_ids in chromosome.assignment.values() for t_id in train_ids)


def fitness_function(chromosome: Chromosome,
                     locomotives: Dict[int, Locomotive],
                     trains: Dict[int, Train],
                     station_coords: Optional[Dict[str, Tuple[float, float]]] = None,
                     weights=(0.4, 0.3, 0.3)) -> float:
    """
    weights = (idle_weight, empty_run_weight, mass_weight)
    idle: number of idle locos (we minimize)
    empty_run: sum of empty distances (we minimize)
    mass: total mass of assigned trains (we maximize)
    Fitness: -w_idle * idle_norm - w_empty * empty_norm + w_mass * mass_norm
    For stability, we normalize each component to comparable scale using simple heuristics.
    """
    # idle count
    idle = sum(1 for trains_list in chromosome.assignment.values() if not trains_list)
    mass = calculate_train_mass(chromosome, trains)

    # empty distance (if we have coords)
    if station_coords is not None:
        # Inject start_depot_map so calculate_empty_run_distance can use it
        start_depot_map = {loco.id: loco.home_depot for loco in locomotives.values()}
        setattr(chromosome, "start_depot_map", start_depot_map)
        empty = calculate_empty_run_distance(chromosome, trains, station_coords)
    else:
        empty = sum(max(0, len(trains) - 1) for trains in chromosome.assignment.values())

    # Normalization heuristics:
    # mass typical scale ~ sum of train weights; empty distance typical ~ 0..1000; idle 0..num_locos
    mass_scale = max(1.0, sum(t.weight for t in trains.values()))
    empty_scale = max(1.0, max(1.0, empty))
    idle_scale = max(1.0, len(locomotives))
    # Convert to normalized measures
    mass_n = mass / mass_scale
    empty_n = empty / empty_scale
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
        train_ids = train_ids_master[:]  # local copy
        random.shuffle(train_ids)
        for t in train_ids:
            assignment[random.choice(loco_ids)].append(t)
        chrom = Chromosome(assignment)
        if is_feasible_fast(chrom, locomotives, trains):
            population.append(chrom)

    if not population:
        # fallback deterministic assignment (round-robin)
        assignment = {lid: [] for lid in loco_ids}
        for i, t in enumerate(train_ids_master):
            assignment[loco_ids[i % len(loco_ids)]].append(t)
        population.append(Chromosome(assignment))
        logger.warning("Initial feasible population not found stochastically, used fallback")

    logger.debug("Initial population size: %d", len(population))
    return population


# ---------------------
# Genetic operators
# ---------------------
def tournament_selection(population: List[Chromosome], k: int = 3) -> Chromosome:
    if not population:
        raise ValueError("Популяция пуста – нельзя провести отбор")
    k = min(k, len(population))
    candidates = random.sample(population, k)
    # fitness must be computed already
    return max(candidates, key=lambda c: c.fitness)


def crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    # uniform crossover per-loco
    child_assignment: Dict[int, List[int]] = {}
    for loco_id in parent1.assignment.keys():
        if random.random() < 0.5:
            child_assignment[loco_id] = copy.deepcopy(parent1.assignment[loco_id])
        else:
            child_assignment[loco_id] = copy.deepcopy(parent2.assignment.get(loco_id, []))
    return Chromosome(child_assignment)


def mutation(chromosome: Chromosome, mutation_rate: float = 0.1):
    loco_ids = list(chromosome.assignment.keys())
    # move single train with some probability
    for loco_id, trains_list in list(chromosome.assignment.items()):
        if trains_list and random.random() < mutation_rate:
            t = random.choice(trains_list)
            trains_list.remove(t)
            target = random.choice(loco_ids)
            chromosome.assignment[target].append(t)


# ---------------------
# GeneticAlgorithm with progress callback
# ---------------------
class GeneticAlgorithm:
    def __init__(self, locomotives: Dict[int, Locomotive], trains: Dict[int, Train],
                 population_size: int = 50,
                 generations: int = 100,
                 tournament_k: int = 3,
                 mutation_rate: float = 0.1,
                 weights=(0.4, 0.3, 0.3),
                 station_coords: Optional[Dict[str, Tuple[float, float]]] = None,
                 progress_callback: Optional[callable] = None):
        self.locomotives = locomotives
        self.trains = trains
        self.population_size = max(1, int(population_size))
        self.generations = max(1, int(generations))
        self.tournament_k = max(1, min(int(tournament_k), self.population_size))
        self.mutation_rate = float(mutation_rate)
        self.weights = tuple(float(w) for w in weights)
        self.station_coords = station_coords
        self.progress_callback = progress_callback  # fn(gen, best_fitness)

    def run(self) -> Chromosome:
        population = generate_initial_population(self.population_size, self.locomotives, self.trains)
        # Evolution
        for gen in range(self.generations):
            # Evaluate fitness
            for chrom in population:
                fitness_function(chrom, self.locomotives, self.trains, station_coords=self.station_coords, weights=self.weights)

            best = max(population, key=lambda c: c.fitness)
            if self.progress_callback:
                try:
                    self.progress_callback(gen, best.fitness)
                except Exception:
                    pass

            new_pop: List[Chromosome] = []

            # produce children
            attempts = 0
            while len(new_pop) < self.population_size and attempts < self.population_size * 10:
                attempts += 1
                p1 = tournament_selection(population, self.tournament_k)
                p2 = tournament_selection(population, self.tournament_k)
                child = crossover(p1, p2)
                mutation(child, self.mutation_rate)
                if is_feasible_fast(child, self.locomotives, self.trains):
                    new_pop.append(child)

            # If new_pop is too small, fill with elites
            if len(new_pop) < self.population_size:
                elites = sorted(population, key=lambda c: c.fitness, reverse=True)
                i = 0
                while len(new_pop) < self.population_size:
                    # clone elite to avoid aliasing
                    e = elites[i % len(elites)]
                    new_pop.append(Chromosome(copy.deepcopy(e.assignment)))
                    i += 1

            population = new_pop

        # final evaluation and return best
        for chrom in population:
            fitness_function(chrom, self.locomotives, self.trains, station_coords=self.station_coords, weights=self.weights)

        best = max(population, key=lambda c: c.fitness)
        logger.info("GA finished: best fitness = %.6f", best.fitness)
        return best


# ---------------------
# Reporting & printing
# ---------------------
class GAReporter:
    def __init__(self):
        self.generation_log: List[Tuple[int, float]] = []
        self.start_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def log_generation(self, gen: int, best_fitness: float):
        self.generation_log.append((gen, best_fitness))

    def elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def print_summary(self, solution: Chromosome):
        print("\n=== 1. Время расчёта и итоговая пригодность ===")
        print(f"Время расчёта, с: {self.elapsed():.2f}")
        print(f"Итоговая целевая функция: {solution.fitness:.6f}")

    def print_generation_curve(self):
        if not self.generation_log:
            print("(нет данных о поколениях)")
            return
        for gen, fit in self.generation_log:
            print(f"Gen {gen:4d}: fitness = {fit:.6f}")


def print_assignment_table(solution: Chromosome, locomotives: Dict[int, Locomotive], trains: Dict[int, Train]):
    lines = []
    for loco_id, train_ids in solution.assignment.items():
        loco = locomotives.get(loco_id)
        if loco is None:
            lines.append(f"Locomotive {loco_id}: no data")
            continue
        lines.append(f"Locomotive {loco_id} | Depot {loco.home_depot} | Remaining resource: {loco.remaining_resource:.1f}")
        for t_id in train_ids:
            t = trains[t_id]
            lines.append(f"  Train {t.id}: {t.route[0]} → {t.route[1]}, dep {t.departure_time:.1f}, dur {t.duration:.1f}")
    output = "\n".join(lines)
    print(output)
    return output


# ---------------------
# Visualization helpers
# ---------------------
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
# Streamlit UI
# ---------------------
def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not available in this environment. Install streamlit to use UI.")
        return

    st.set_page_config(page_title="Loco_GA", layout="wide")
    st.title("Loco_GA — Genetic Algorithm for Locomotive Assignment")
    st.sidebar.header("Settings")

    # Data generation settings
    num_locomotives = st.sidebar.slider("Number of locomotives", 1, 200, 10)
    num_trains = st.sidebar.slider("Number of trains", 1, 500, 20)
    depot_names_str = st.sidebar.text_input("Stations/Depots (comma-separated)", "A,B,C")
    depot_names = tuple(s.strip() for s in depot_names_str.split(",") if s.strip())
    seed = st.sidebar.number_input("Random seed (0 means random)", min_value=0, value=0, step=1)
    if seed == 0:
        seed = None

    # GA params
    st.sidebar.subheader("GA parameters")
    population_size = st.sidebar.number_input("Population size", min_value=2, max_value=2000, value=60)
    generations = st.sidebar.number_input("Generations", min_value=1, max_value=2000, value=120)
    tournament_k = st.sidebar.number_input("Tournament size", min_value=1, max_value=population_size, value=5)
    mutation_rate = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.15, 0.01)
    weights_idle = st.sidebar.slider("Weight: idle", 0.0, 1.0, 0.4, 0.05)
    weights_empty = st.sidebar.slider("Weight: empty", 0.0, 1.0, 0.3, 0.05)
    weights_mass = st.sidebar.slider("Weight: mass", 0.0, 1.0, 0.3, 0.05)
    # Normalize weights
    w_sum = max(1e-6, weights_idle + weights_empty + weights_mass)
    weights = (weights_idle / w_sum, weights_empty / w_sum, weights_mass / w_sum)

    # Button to generate data
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

    # Show summary
    st.sidebar.markdown(f"Locomotives: {len(locomotives)}")
    st.sidebar.markdown(f"Trains: {len(trains)}")
    st.sidebar.markdown(f"Stations: {len(station_coords)}")

    # Run GA
    run_button = st.button("Run Genetic Algorithm")
    stop_run = False

    # area for progress and logs
    progress_bar = st.progress(0)
    progress_text = st.empty()
    logs_box = st.empty()

    reporter = GAReporter()
    reporter.start()

    def progress_callback(gen, best_fitness):
        # Called from GA thread
        frac = (gen + 1) / generations
        if frac < 0: frac = 0
        if frac > 1: frac = 1
        try:
            progress_bar.progress(int(frac * 100))
            progress_text.text(f"Generation {gen+1}/{generations} — best fitness {best_fitness:.6f}")
        except Exception:
            pass
        reporter.log_generation(gen, best_fitness)

    if run_button:
        # run GA in a thread so Streamlit UI remains responsive
        def ga_thread():
            nonlocal locomotives, trains, station_coords, weights
            try:
                ga = GeneticAlgorithm(
                    locomotives, trains,
                    population_size=int(population_size),
                    generations=int(generations),
                    tournament_k=int(tournament_k),
                    mutation_rate=float(mutation_rate),
                    weights=weights,
                    station_coords=station_coords,
                    progress_callback=progress_callback
                )
                best = ga.run()
                st.session_state["solution"] = best
                st.session_state["reporter"] = reporter
                logger.info("GA run completed via Streamlit")
            except Exception as e:
                logger.exception("Error during GA run: %s", e)
                st.error(f"GA error: {e}")

        thread = threading.Thread(target=ga_thread, daemon=True)
        thread.start()
        st.info("GA started in background thread. Progress shown above.")

    # When solution available, show tables & plots
    if "solution" in st.session_state:
        solution = st.session_state["solution"]
        reporter = st.session_state.get("reporter", reporter)
        st.subheader("Best solution")
        st.markdown(f"**Fitness:** {solution.fitness:.6f}")
        # assignment table
        text = print_assignment_table(solution, locomotives, trains)
        st.text(text)

        # plot timeline
        fig = plot_assignment_matplotlib(solution, trains)
        st.pyplot(fig)

        # generation curve
        st.subheader("Generation curve")
        if reporter.generation_log:
            gens, fits = zip(*reporter.generation_log)
            plt.figure(figsize=(8, 3))
            plt.plot(gens, fits, "-o")
            plt.xlabel("Generation")
            plt.ylabel("Best fitness")
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.write("No generation log available")

        # sensitivity quick run
        if st.button("Run sensitivity (quick)"):
            with st.spinner("Running sensitivity..."):
                sens = []
                deltas = (-0.1, 0.0, 0.1)
                for d in deltas:
                    base = (weights[0] + d, weights[1] + d, weights[2] - 2 * d)
                    ssum = max(sum(base), 1e-6)
                    ws = (base[0] / ssum, base[1] / ssum, base[2] / ssum)
                    ga = GeneticAlgorithm(locomotives, trains,
                                          population_size=max(10, min(200, population_size//2)),
                                          generations=max(5, generations//4),
                                          tournament_k=tournament_k,
                                          mutation_rate=mutation_rate,
                                          weights=ws,
                                          station_coords=station_coords)
                    try:
                        solx = ga.run()
                        sens.append((d, ws, solx.fitness))
                    except Exception as e:
                        sens.append((d, ws, float("nan")))
                # display
                st.table([{"delta": d, "w_idle": round(ws[0], 3), "w_empty": round(ws[1], 3), "w_mass": round(ws[2], 3), "fitness": f} for d, ws, f in sens])

    st.sidebar.markdown("---")
    st.sidebar.markdown("Logs are written to loco_ga.log")


# ---------------------
# CLI fallback
# ---------------------
def run_cli_demo():
    locomotives, trains, station_coords = generate_synthetic_data()
    ga = GeneticAlgorithm(locomotives, trains,
                          population_size=60, generations=120,
                          tournament_k=5, mutation_rate=0.15,
                          station_coords=station_coords)
    reporter = GAReporter()
    reporter.start()
    best = ga.run()
    reporter.print_summary(best)
    print_assignment_table(best, locomotives, trains)
    fig = plot_assignment_matplotlib(best, trains)
    plt.show()


# ---------------------
# Entrypoint
# ---------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        logger.info("Streamlit not available — running CLI demo")
        run_cli_demo()
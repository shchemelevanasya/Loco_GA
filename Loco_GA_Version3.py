from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import copy
import time
import math
import logging
import threading
import concurrent.futures
import multiprocessing
import os

# optional imports (matplotlib for plotting)
import matplotlib.pyplot as plt

# Try to import streamlit, otherwise provide a CLI fallback
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# ---------------------
# Logging configuration
# ---------------------
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
# Helpers: geometry, distance->time
# ---------------------
def compute_distance_3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance in 3D (same units as coords)"""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.hypot(math.hypot(dx, dy), dz)


def distance_to_time(distance: float,
                     speed_kmh: float = 60.0,
                     slope_elevation_diff: float = 0.0,
                     slope_penalty_coefficient: float = 0.05) -> float:
    """
    Convert distance (in same linear units; assume 1 unit = 1 km for simplicity or provide coords accordingly)
    into hours considering average repositioning speed and slope penalty.

    slope_penalty_coefficient: how much time increases per unit relative slope (dimensionless).
      For a slope expressed as abs(elev_diff)/distance, penalty multiplier = 1 + slope_penalty_coefficient * slope.
    """
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be > 0")
    if distance <= 0:
        return 0.0
    # slope (rise/run). If distance very small, treat slope gracefully
    slope = abs(slope_elevation_diff) / max(distance, 1e-6)
    penalty = 1.0 + slope_penalty_coefficient * slope
    hours = (distance / speed_kmh) * penalty
    return hours


# ---------------------
# Synthetic data generator (with elevation)
# ---------------------
def generate_synthetic_data(
        num_locomotives: int = 10,
        num_trains: int = 20,
        depot_names: Tuple[str, ...] = ("A", "B", "C"),
        station_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
        seed: Optional[int] = None):
    """
    Generate synthetic locomotives/trains and 3D station coords (x, y, elevation).
    If station_coords not provided, stations are placed on a circle with small random elevation.

    Returns:
      locomotives: Dict[id, Locomotive]
      trains: Dict[id, Train]
      station_coords: Dict[station_name, (x, y, elevation)]
    """
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
            elev = random.uniform(-50, 200)  # meters (or arbitrary units)
            station_coords[name] = (x, y, elev)

    locomotives: Dict[int, Locomotive] = {}
    for i in range(num_locomotives):
        locomotives[i] = Locomotive(
            id=i,
            loco_type="2ЭС6",
            power=random.uniform(4000, 7000),
            remaining_resource=random.uniform(20, 50),  # hours
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
        """
        assignment: mapping loco_id -> ordered list of train_ids
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
# Fast feasibility check
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
# Empty-run time using distance->time and elevation
# ---------------------
def calculate_empty_run_time(chromosome: Chromosome,
                             locomotives: Dict[int, Locomotive],
                             trains: Dict[int, Train],
                             station_coords: Dict[str, Tuple[float, float, float]],
                             reposition_speed_kmh: float = 60.0,
                             slope_penalty_coefficient: float = 0.05) -> float:
    """
    For each locomotive:
      - time from home_depot -> first train departure station
      - times between successive assignments (prev_arrival -> next_departure)
    Returns total repositioning time in hours.
    """
    total_time = 0.0
    lookup_dep = {tid: trains[tid].route[0] for tid in trains}
    lookup_arr = {tid: trains[tid].route[1] for tid in trains}

    for loco_id, train_ids in chromosome.assignment.items():
        if not train_ids:
            continue
        loco = locomotives.get(loco_id)
        if loco is None:
            continue
        # depot -> first departure
        first_dep = lookup_dep[train_ids[0]]
        depot_name = loco.home_depot
        if depot_name in station_coords and first_dep in station_coords:
            depot_coord = station_coords[depot_name]
            first_coord = station_coords[first_dep]
            dist = compute_distance_3d(depot_coord, first_coord)
            elev_diff = first_coord[2] - depot_coord[2]
            total_time += distance_to_time(dist, reposition_speed_kmh, elev_diff, slope_penalty_coefficient)
        # between successive
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
# Fitness function (distance->time considered)
# ---------------------
def calculate_train_mass(chromosome: Chromosome, trains: Dict[int, Train]) -> float:
    return sum(trains[t_id].weight for train_ids in chromosome.assignment.values() for t_id in train_ids)


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
# Multiprocessing helpers (workers)
# ---------------------
def _fitness_worker_serial(args):
    """Worker for ProcessPoolExecutor: args is tuple"""
    assignment, locomotives, trains, station_coords, weights, reposition_speed_kmh, slope_penalty_coefficient = args
    chrom = Chromosome(assignment)
    fitness_function(chrom, locomotives, trains, station_coords, weights,
                     reposition_speed_kmh, slope_penalty_coefficient)
    # return assignment and fitness
    return chrom.assignment, chrom.fitness


def _child_worker_serial(args):
    """Worker that builds a child from two parent assignments and returns assignment if feasible else None"""
    assign_p1, assign_p2, locomotives, trains, mutation_rate = args
    child_assign = crossover_assignment(assign_p1, assign_p2)
    mutation_assignment(child_assign, mutation_rate)
    child = Chromosome(child_assign)
    if is_feasible_fast(child, locomotives, trains):
        return child_assign
    return None


# ---------------------
# GeneticAlgorithm with optional multiprocessing
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
        """
        use_multiprocessing: enable parallel evaluation/generation when population is large
        multiprocessing_threshold: minimal population size to start using parallelism
        """
        self.locomotives = locomotives
        self.trains = trains
        self.population_size = max(1, int(population_size))
        self.generations = max(1, int(generations))
        self.tournament_k = max(1, min(int(tournament_k), self.population_size))
        self.mutation_rate = float(mutation_rate)
        self.weights = tuple(float(w) for w in weights)
        self.station_coords = station_coords
        self.progress_callback = progress_callback
        self.use_multiprocessing = bool(use_multiprocessing)
        self.multiprocessing_threshold = int(multiprocessing_threshold)
        self.cpu_count = max(1, multiprocessing.cpu_count())

    def _evaluate_population(self, population: List[Chromosome]):
        """Evaluate fitness for the whole population; may use multiprocessing."""
        if self.use_multiprocessing and len(population) >= self.multiprocessing_threshold and self.cpu_count > 1:
            # prepare args
            args = [
                (chrom.assignment, self.locomotives, self.trains, self.station_coords, self.weights,
                 60.0, 0.05)
                for chrom in population
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.cpu_count, len(args))) as exc:
                for assignment, fitness in exc.map(_fitness_worker_serial, args):
                    # find corresponding chromosome and set fitness
                    # To avoid lookup complexity, we match by assignment identity via comparing contents
                    for chrom in population:
                        if chrom.assignment == assignment:
                            chrom.fitness = fitness
                            break
        else:
            for chrom in population:
                fitness_function(chrom, self.locomotives, self.trains, self.station_coords, self.weights)

    def _generate_children(self, population: List[Chromosome], target_count: int) -> List[Chromosome]:
        """
        Create children until target_count reached. May use multiprocessing to produce children faster.
        Returns list of Chromosome objects.
        """
        children: List[Chromosome] = []
        if self.use_multiprocessing and len(population) >= self.multiprocessing_threshold and self.cpu_count > 1:
            # prepare parent pairs
            parent_pairs = []
            for _ in range(target_count * 2):  # attempt more pairs to improve chance of feasible children
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
        for gen in range(self.generations):
            # Evaluate
            self._evaluate_population(population)
            best = max(population, key=lambda c: c.fitness)
            if self.progress_callback:
                try:
                    self.progress_callback(gen, best.fitness)
                except Exception:
                    pass
            # Create new population
            new_population: List[Chromosome] = []
            # generate children
            children = self._generate_children(population, self.population_size)
            new_population.extend(children)
            # if not enough children, fill with elites
            if len(new_population) < self.population_size:
                elites = sorted(population, key=lambda c: c.fitness, reverse=True)
                i = 0
                while len(new_population) < self.population_size:
                    e = elites[i % len(elites)]
                    new_population.append(Chromosome(copy.deepcopy(e.assignment)))
                    i += 1
            population = new_population
        # final evaluation
        self._evaluate_population(population)
        best = max(population, key=lambda c: c.fitness)
        logger.info("GA finished: best fitness = %.6f", best.fitness)
        return best


# ---------------------
# Reporting & visualization (unchanged)
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


def print_assignment_table(solution: Chromosome, locomotives: Dict[int, Locomotive], trains: Dict[int, Train]) -> str:
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
# Streamlit UI (unchanged core, uses GA.run which may use multiprocessing)
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
    weights_idle = st.sidebar.slider("Weight: idle", 0.0, 1.0, 0.4, 0.05)
    weights_empty = st.sidebar.slider("Weight: empty", 0.0, 1.0, 0.3, 0.05)
    weights_mass = st.sidebar.slider("Weight: mass", 0.0, 1.0, 0.3, 0.05)
    w_sum = max(1e-6, weights_idle + weights_empty + weights_mass)
    weights = (weights_idle / w_sum, weights_empty / w_sum, weights_mass / w_sum)

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

    run_button = st.button("Run Genetic Algorithm")
    progress_bar = st.progress(0)
    progress_text = st.empty()
    reporter = GAReporter()
    reporter.start()

    def progress_callback(gen, best_fitness):
        frac = (gen + 1) / max(1, generations)
        progress_bar.progress(int(frac * 100))
        progress_text.text(f"Generation {gen+1}/{generations} — best fitness {best_fitness:.6f}")
        reporter.log_generation(gen, best_fitness)

    if run_button:
        def ga_thread():
            try:
                ga = GeneticAlgorithm(
                    locomotives, trains,
                    population_size=int(population_size),
                    generations=int(generations),
                    tournament_k=int(tournament_k),
                    mutation_rate=float(mutation_rate),
                    weights=weights,
                    station_coords=station_coords,
                    progress_callback=progress_callback,
                    use_multiprocessing=bool(use_mp),
                    multiprocessing_threshold=int(mp_threshold)
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

    if "solution" in st.session_state:
        solution = st.session_state["solution"]
        reporter = st.session_state.get("reporter", reporter)
        st.subheader("Best solution")
        st.markdown(f"**Fitness:** {solution.fitness:.6f}")
        text = print_assignment_table(solution, locomotives, trains)
        st.text(text)
        fig = plot_assignment_matplotlib(solution, trains)
        st.pyplot(fig)

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
    print_assignment_table(best, locomotives, trains)
    fig = plot_assignment_matplotlib(best, trains)
    plt.show()


# ---------------------
# Entrypoint
# ---------------------
if __name__ == "__main__":
    # If environment variable RUN_STREAMLIT=1, run streamlit app (useful for local dev)
    if STREAMLIT_AVAILABLE and os.environ.get("RUN_STREAMLIT", "0") == "1":
        run_streamlit_app()
    elif STREAMLIT_AVAILABLE and "PYCHARM_HOSTED" not in os.environ:
        # default to streamlit UI when interactive and streamlit installed
        run_streamlit_app()
    else:
        logger.info("Streamlit not available or not selected — running CLI demo")
        run_cli_demo()

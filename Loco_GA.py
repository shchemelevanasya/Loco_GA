from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
import copy

@dataclass
class Locomotive:
    id: int
    loco_type: str
    power: float                 # тяговая мощность
    remaining_resource: float    # ресурс до ТО
    home_depot: str

class Train:
    id: int
    weight: float
    route: Tuple[str, str]       # (откуда, куда)
    departure_time: float
    duration: float

class Chromosome:
    def __init__(self, assignment: Dict[int, List[int]]):
        """
        assignment:
        key   = id локомотива
        value = список id поездов в порядке обслуживания
        """
        self.assignment = assignment
        self.fitness = None

def check_traction(loco: Locomotive, train: Train) -> bool:
    return loco.power >= train.weight

def check_resource(loco: Locomotive, trains: List[Train]) -> bool:
    total_duration = sum(t.duration for t in trains)
    return total_duration <= loco.remaining_resource

def is_feasible(chromosome: Chromosome,
                locomotives: Dict[int, Locomotive],
                trains: Dict[int, Train]) -> bool:
    for loco_id, train_ids in chromosome.assignment.items():
        loco = locomotives[loco_id]
        assigned_trains = [trains[t_id] for t_id in train_ids]

        # Тяга
        for t in assigned_trains:
            if not check_traction(loco, t):
                return False

        # Ресурс
        if not check_resource(loco, assigned_trains):
            return False

    return True

def calculate_idle_time(chromosome: Chromosome) -> float:
    return sum(len(trains) == 0 for trains in chromosome.assignment.values())

def calculate_empty_run(chromosome: Chromosome) -> float:
    # прокси-оценка: количество разрывов маршрута
    return sum(max(0, len(trains) - 1) for trains in chromosome.assignment.values())

def calculate_train_mass(chromosome: Chromosome,
                         trains: Dict[int, Train]) -> float:
    return sum(trains[t_id].weight
               for train_ids in chromosome.assignment.values()
               for t_id in train_ids)

def fitness_function(chromosome: Chromosome,
                     locomotives: Dict[int, Locomotive],
                     trains: Dict[int, Train],
                     weights=(0.4, 0.3, 0.3)) -> float:
    """
    weights = (idle_weight, empty_run_weight, mass_weight)
    """

    idle = calculate_idle_time(chromosome)
    empty = calculate_empty_run(chromosome)
    mass = calculate_train_mass(chromosome, trains)

    # минимизируем простои и порожние пробеги,
    # максимизируем массу поездов
    fitness = (
        -weights[0] * idle
        -weights[1] * empty
        +weights[2] * mass
    )

    chromosome.fitness = fitness
    return fitness

def generate_initial_population(pop_size: int,
                                locomotives: Dict[int, Locomotive],
                                trains: Dict[int, Train]) -> List[Chromosome]:
    population = []

    loco_ids = list(locomotives.keys())
    train_ids = list(trains.keys())

    for _ in range(pop_size):
        assignment = {loco_id: [] for loco_id in loco_ids}

        random.shuffle(train_ids)
        for t_id in train_ids:
            loco_id = random.choice(loco_ids)
            assignment[loco_id].append(t_id)

        chromosome = Chromosome(assignment)

        if is_feasible(chromosome, locomotives, trains):
            population.append(chromosome)

    return population

def tournament_selection(population: List[Chromosome],
                          k: int = 3) -> Chromosome:
    candidates = random.sample(population, k)
    return max(candidates, key=lambda c: c.fitness)

def crossover(parent1: Chromosome,
              parent2: Chromosome) -> Chromosome:
    child_assignment = {}

    for loco_id in parent1.assignment.keys():
        if random.random() < 0.5:
            child_assignment[loco_id] = copy.deepcopy(parent1.assignment[loco_id])
        else:
            child_assignment[loco_id] = copy.deepcopy(parent2.assignment[loco_id])

    return Chromosome(child_assignment)

def mutation(chromosome: Chromosome,
             mutation_rate: float = 0.1):
    loco_ids = list(chromosome.assignment.keys())

    for loco_id, trains in chromosome.assignment.items():
        if trains and random.random() < mutation_rate:
            t_id = random.choice(trains)
            trains.remove(t_id)
            new_loco = random.choice(loco_ids)
            chromosome.assignment[new_loco].append(t_id)

class GeneticAlgorithm:
    def __init__(self, locomotives, trains,
                 population_size=50,
                 generations=100):
        self.locomotives = locomotives
        self.trains = trains
        self.population_size = population_size
        self.generations = generations

    def run(self):
        population = generate_initial_population(
            self.population_size,
            self.locomotives,
            self.trains
        )

        for gen in range(self.generations):
            for chrom in population:
                fitness_function(chrom, self.locomotives, self.trains)

            new_population = []

            while len(new_population) < self.population_size:
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)

                child = crossover(p1, p2)
                mutation(child)

                if is_feasible(child, self.locomotives, self.trains):
                    new_population.append(child)

            population = new_population

        return max(population, key=lambda c: c.fitness)



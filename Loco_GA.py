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

@dataclass
class Train:
    id: int
    weight: float
    route: Tuple[str, str]       # (откуда, куда)
    departure_time: float
    duration: float

import random

def generate_synthetic_data(
        num_locomotives=10,
        num_trains=20,
        depots=("A", "B", "C")):
    """
    Генерация синтетических данных
    для экспериментальных расчетов (глава 4)
    """

    locomotives = {}
    for i in range(num_locomotives):
        locomotives[i] = Locomotive(
            id=i,
            loco_type="2ЭС6",
            power=random.uniform(4000, 7000),
            remaining_resource=random.uniform(20, 50),
            home_depot=random.choice(depots)
        )

    trains = {}
    for j in range(num_trains):
        dep = random.choice(depots)
        arr = random.choice([d for d in depots if d != dep])
        trains[j] = Train(
            id=j,
            weight=random.uniform(3000, 6000),
            route=(dep, arr),
            departure_time=random.uniform(0, 24),
            duration=random.uniform(2, 6)
        )
            
    return locomotives, trains

def preprocess_external_data(train_table, loco_table):
    """
    Предобработка данных,
    поступающих из внешних информационных систем
    """

    trains = {}
    for row in train_table:
        trains[row["train_id"]] = Train(
            id=row["train_id"],
            weight=row["weight"],
            route=(row["dep_station"], row["arr_station"]),
            departure_time=row["dep_time"],
            duration=row["duration"]
        )

    locomotives = {}
    for row in loco_table:
        locomotives[row["loco_id"]] = Locomotive(
            id=row["loco_id"],
            loco_type=row["type"],
            power=row["power"],
            remaining_resource=row["resource"],
            home_depot=row["depot"]
        )

    return locomotives, trains


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
                 generations=100,
                 tournament_selection=3,         # ← новый параметр
                 mutation_rate=0.1):        # ← новый параметр
        self.locomotives = locomotives
        self.trains = trains
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.tournament_selection = int(tournament_selection, self.population_size) # ← корректировка
        self.mutation_rate = float(mutation_rate)         # ← новый параметр

# защита
        if self.tournament_selection > self.population_size:
            self.tournament_selection = self.population_size
        if self.tournament_selection < 1:
            self.tournament_selection = 1

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

import time

def run_experiment(ga, baseline_solution, heuristic_solution):
    start = time.time()
    ga_solution = ga.run()
    ga_time = time.time() - start

    return {
        "baseline_fitness": baseline_solution.fitness,
        "heuristic_fitness": heuristic_solution.fitness,
        "ga_fitness": ga_solution.fitness,
        "ga_time_sec": ga_time
    }

import pandas as pd

# def build_assignment_table(solution, locomotives, trains):
    #rows = []
    #for loco_id, train_ids in solution.assignment.items():
     #   loco = locomotives[loco_id]
       # for t_id in train_ids:
         #   t = trains[t_id]
           # rows.append({
             #   "Поезд": t.id,
             #   "Откуда": t.route[0],
             #   "Куда": t.route[1],
             #   "Отправление": t.departure_time,
              #  "Прибытие": t.departure_time + t.duration,
              #  "Локомотив": loco.id,
              #  "Тип локомотива": loco.loco_type,
               # "Остаточный ресурс": loco.remaining_resource
          #  })
  #  return pd.DataFrame(rows)} 


def print_assignment_table(solution, locomotives, trains):
    print("\nРезультаты назначения локомотивов:\n")

    for loco_id, train_ids in solution.assignment.items():
        loco = locomotives[loco_id]
        print(f"Локомотив {loco_id} | Депо {loco.home_depot} "
              f"| Остаточный ресурс: {loco.remaining_resource:.1f}")

        for t_id in train_ids:
            t = trains[t_id]
            print(f"  Поезд {t.id}: {t.route[0]} → {t.route[1]}, "
                  f"отпр {t.departure_time:.1f}, длит {t.duration:.1f}")
        print()



import matplotlib.pyplot as plt

def plot_assignment(solution, trains):
    fig, ax = plt.subplots()

    y = 0
    for loco_id, train_ids in solution.assignment.items():
        for t_id in train_ids:
            t = trains[t_id]
            ax.barh(
                y,
                t.duration,
                left=t.departure_time,
                height=0.4
            )
        y += 1

    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Локомотивы")
    ax.set_title("Прогнозный график назначения локомотивов")
    plt.show()

if __name__ == "__main__":

    # 1. Ввод данных
    locomotives, trains = generate_synthetic_data()

    # 2. Запуск алгоритма
    ga = GeneticAlgorithm(
        locomotives,
        trains,
        population_size=50,
        generations=100,
        tournament_selection=5,
        mutation_rate=0.1
    )

    solution = ga.run()

    # 3. Вывод результатов
    print_assignment_table(solution, locomotives, trains)
    plot_assignment(solution, trains)

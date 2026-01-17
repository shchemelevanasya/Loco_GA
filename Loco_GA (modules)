# ===============================
# МОДУЛЬ ИМПОРТОВ И НАСТРОЕК
# ===============================

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import copy
import matplotlib.pyplot as plt

# ===============================
# МОДУЛЬ ОПИСАНИЯ ОБЪЕКТОВ
# ===============================

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

# ===============================
# МОДУЛЬ ВВОДА ДАННЫХ
# ===============================

def generate_synthetic_data(num_locomotives=10, num_trains=20):
    depots = ["A", "B", "C"]

    locomotives = {
        i: Locomotive(i, "2ЭС6",
                      random.uniform(4000, 7000),
                      random.uniform(20, 50),
                      random.choice(depots))
        for i in range(num_locomotives)
    }

    trains = {
        j: Train(j,
                 random.uniform(3000, 6000),
                 (random.choice(depots), random.choice(depots)),
                 random.uniform(0, 24),
                 random.uniform(2, 6))
        for j in range(num_trains)
    }

    return locomotives, trains

# ===============================
# МОДУЛЬ ГЕНЕТИЧЕСКОГО АЛГОРИТМА
# ===============================

class Chromosome:
    def __init__(self, assignment):
        self.assignment = assignment
        self.fitness = None

def is_feasible(chrom, locos, trains):
    for lid, tids in chrom.assignment.items():
        loco = locos[lid]
        total_time = sum(trains[t].duration for t in tids)
        if total_time > loco.remaining_resource:
            return False
        for t in tids:
            if loco.power < trains[t].weight:
                return False
    return True

def fitness(chrom, trains):
    idle = sum(len(v) == 0 for v in chrom.assignment.values())
    mass = sum(trains[t].weight for v in chrom.assignment.values() for t in v)
    chrom.fitness = mass - 1000 * idle
    return chrom.fitness

class GeneticAlgorithm:
    def __init__(self, locos, trains, pop_size=50, gens=100):
        self.locos = locos
        self.trains = trains
        self.pop_size = pop_size
        self.gens = gens

    def run(self):
        population = []

        while len(population) < self.pop_size:
            assgn = {l: [] for l in self.locos}
            for t in self.trains:
                assgn[random.choice(list(self.locos))].append(t)
            c = Chromosome(assgn)
            if is_feasible(c, self.locos, self.trains):
                population.append(c)

        for _ in range(self.gens):
            for c in population:
                fitness(c, self.trains)
            population.sort(key=lambda x: x.fitness, reverse=True)
            population = population[:self.pop_size // 2]
            while len(population) < self.pop_size:
                population.append(copy.deepcopy(random.choice(population)))

        return max(population, key=lambda x: x.fitness)

# ===============================
# МОДУЛЬ ВЫВОДА РЕЗУЛЬТАТОВ
# ===============================

def print_solution(sol, locos, trains):
    for l, ts in sol.assignment.items():
        print(f"Локомотив {l}:")
        for t in ts:
            print(f"  Поезд {t} {trains[t].route}")

# ===============================
# ТОЧКА ВХОДА
# ===============================

if __name__ == "__main__":
    locos, trains = generate_synthetic_data()
    ga = GeneticAlgorithm(locos, trains)
    sol = ga.run()
    print_solution(sol, locos, trains)

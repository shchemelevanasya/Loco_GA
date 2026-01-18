# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import copy
import time
import matplotlib.pyplot as plt

@dataclass
class Locomotive:
    id: int
    loco_type: str
    power: float                 # тяговая мощность
    remaining_resource: float    # ресурс до ТО (часы)
    home_depot: str

@dataclass
class Train:
    id: int
    weight: float
    route: Tuple[str, str]       # (откуда, куда)
    departure_time: float
    duration: float

def generate_synthetic_data(
        num_locomotives: int = 10,
        num_trains: int = 20,
        depots=("A", "B", "C")):
    """
    Генерация синтетических данных для экспериментальных расчетов
    """
    locomotives: Dict[int, Locomotive] = {}
    for i in range(num_locomotives):
        locomotives[i] = Locomotive(
            id=i,
            loco_type="2ЭС6",
            power=random.uniform(4000, 7000),
            remaining_resource=random.uniform(20, 50),
            home_depot=random.choice(depots)
        )

    trains: Dict[int, Train] = {}
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
    Предобработка данных, поступающих из внешних информационных систем
    """
    trains: Dict[int, Train] = {}
    for row in train_table:
        trains[row["train_id"]] = Train(
            id=row["train_id"],
            weight=row["weight"],
            route=(row["dep_station"], row["arr_station"]),
            departure_time=row["dep_time"],
            duration=row["duration"]
        )

    locomotives: Dict[int, Locomotive] = {}
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
        self._fitness: Optional[float] = None

    @property
    def fitness(self) -> float:
        if self._fitness is None:
            raise RuntimeError("Fitness ещё не вычислен – вызовите fitness_function()")
        return self._fitness

    @fitness.setter
    def fitness(self, val: float):
        self._fitness = float(val)

def check_traction(loco: Locomotive, train: Train) -> bool:
    return loco.power >= train.weight

def check_resource(loco: Locomotive, trains: List[Train]) -> bool:
    total_duration = sum(t.duration for t in trains)
    return total_duration <= loco.remaining_resource

def is_feasible(chromosome: Chromosome,
                locomotives: Dict[int, Locomotive],
                trains: Dict[int, Train]) -> bool:
    for loco_id, train_ids in chromosome.assignment.items():
        if loco_id not in locomotives:
            return False
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

def calculate_idle_time(chromosome: Chromosome) -> int:
    # количество локомотивов без назначений
    return sum(1 for trains in chromosome.assignment.values() if len(trains) == 0)

def calculate_empty_run(chromosome: Chromosome) -> int:
    # прокси-оценка: количество разрывов маршрута (число переходов > 0)
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
    Чем выше fitness — тем лучше (модель: минимизируем idle и empty, максимизируем mass)
    """
    idle = calculate_idle_time(chromosome)
    empty = calculate_empty_run(chromosome)
    mass = calculate_train_mass(chromosome, trains)

    fitness = (
        -weights[0] * idle
        -weights[1] * empty
        +weights[2] * mass
    )

    chromosome.fitness = fitness
    return fitness

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
        assignment = {loco_id: [] for loco_id in loco_ids}
        train_ids = train_ids_master[:]  # local copy
        random.shuffle(train_ids)
        for t_id in train_ids:
            loco_id = random.choice(loco_ids)
            assignment[loco_id].append(t_id)
        chrom = Chromosome(assignment)
        if is_feasible(chrom, locomotives, trains):
            population.append(chrom)

    # fallback: если не удалось получить допустимую популяцию, создаём детерминированную раскладку
    if not population:
        assignment = {loco_id: [] for loco_id in loco_ids}
        for i, t_id in enumerate(train_ids_master):
            assignment[loco_ids[i % len(loco_ids)]].append(t_id)
        chrom = Chromosome(assignment)
        # не гарантируем is_feasible — всё же возвращаем хотя бы одну хромосому
        population.append(chrom)

    return population

def tournament_selection(population: List[Chromosome],
                         k: int = 3) -> Chromosome:
    if not population:
        raise ValueError("Популяция пуста – нельзя провести отбор")
    k = min(k, len(population))
    if k <= 0:
        raise ValueError("Неверный размер турнира")
    candidates = random.sample(population, k)
    # Предполагается, что fitness уже вычислен для всех кандидатов
    for c in candidates:
        if getattr(c, "_fitness", None) is None:
            raise RuntimeError("Fitness для кандидата не вычислен")
    return max(candidates, key=lambda c: c.fitness)

def crossover(parent1: Chromosome,
              parent2: Chromosome) -> Chromosome:
    child_assignment: Dict[int, List[int]] = {}
    for loco_id in parent1.assignment.keys():
        if random.random() < 0.5:
            child_assignment[loco_id] = copy.deepcopy(parent1.assignment[loco_id])
        else:
            child_assignment[loco_id] = copy.deepcopy(parent2.assignment.get(loco_id, []))
    return Chromosome(child_assignment)

def mutation(chromosome: Chromosome,
             mutation_rate: float = 0.1):
    loco_ids = list(chromosome.assignment.keys())
    # небольшая мутация: перекидываем поезд с вероятностью mutation_rate
    for loco_id, trains in list(chromosome.assignment.items()):
        if trains and random.random() < mutation_rate:
            t_id = random.choice(trains)
            trains.remove(t_id)
            new_loco = random.choice(loco_ids)
            chromosome.assignment[new_loco].append(t_id)

class GeneticAlgorithm:
    def __init__(self, locomotives: Dict[int, Locomotive], trains: Dict[int, Train],
                 population_size: int = 50,
                 generations: int = 100,
                 tournament_k: int = 3,
                 mutation_rate: float = 0.1,
                 weights=(0.4, 0.3, 0.3)):
        self.locomotives = locomotives
        self.trains = trains
        self.population_size = max(1, int(population_size))
        self.generations = max(1, int(generations))
        self.tournament_k = max(1, min(int(tournament_k), self.population_size))
        self.mutation_rate = float(mutation_rate)
        self.weights = tuple(float(w) for w in weights)

    def run(self, reporter=None) -> Chromosome:
        population = generate_initial_population(
            self.population_size,
            self.locomotives,
            self.trains
        )
        if not population:
            raise RuntimeError("Начальная популяция пуста")

        # Эволюция
        for gen in range(self.generations):
            # Вычисляем fitness для всей популяции
            for chrom in population:
                fitness_function(chrom, self.locomotives, self.trains, weights=self.weights)

            best = max(population, key=lambda c: c.fitness)
            if reporter:
                try:
                    reporter.log_generation(gen, best.fitness)
                except Exception:
                    # защитная заглушка, если reporter неожиданно не реализует log_generation
                    pass

            new_population: List[Chromosome] = []

            # Генерируем детей до размера population_size
            while len(new_population) < self.population_size:
                # отбор родителей турниром
                p1 = tournament_selection(population, self.tournament_k)
                p2 = tournament_selection(population, self.tournament_k)
                child = crossover(p1, p2)
                mutation(child, self.mutation_rate)
                # проверка допустимости
                if is_feasible(child, self.locomotives, self.trains):
                    new_population.append(child)
                # защита от зацикливания: если не получилось создать детей, добавляем элиту
                if len(new_population) < self.population_size and len(new_population) + len(population) >= self.population_size:
                    # дополняем лучшими родителями
                    parents_sorted = sorted(population, key=lambda c: c.fitness, reverse=True)
                    needed = self.population_size - len(new_population)
                    for p in parents_sorted[:needed]:
                        # клонируем родителя
                        new_population.append(Chromosome(copy.deepcopy(p.assignment)))

                # Если несколько неудачных попыток — прерываем (защита)
                if len(new_population) == 0 and len(population) == 0:
                    break

            population = new_population

            if not population:
                raise RuntimeError(
                    "Популяция пуста: ни одна хромосома не прошла ограничения. "
                    "Проверь данные или смягчи ограничения в is_feasible."
                )

        # Финальная оценка
        for chrom in population:
            fitness_function(chrom, self.locomotives, self.trains, weights=self.weights)

        return max(population, key=lambda c: c.fitness)

class GAReporter:
    """Сбор статистики по ходу эволюции"""
    def __init__(self):
        self.generation_log: List[Tuple[int, float]] = []   # (номер_поколения, лучшая_пригодность)
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
        print(f"Итоговая целевая функция: {solution.fitness:.2f}")

    def print_generation_curve(self):
        print("\n=== 3. Поколение → лучшая пригодность ===")
        print(f"{'Поколение':>9} | {'fitness':>10}")
        print("-" * 23)
        if not self.generation_log:
            print("(нет данных)")
            return
        # Показываем не более 20 точек
        step = max(1, len(self.generation_log) // 20)
        for gen, fit in self.generation_log[::step]:
            print(f"{gen:9} | {fit:10.2f}")

def sensitivity_table(locomotives: Dict[int, Locomotive],
                      trains: Dict[int, Train],
                      base_weights=(0.4, 0.3, 0.3),
                      deltas=(-0.2, -0.1, 0, 0.1, 0.2),
                      population_size=50,
                      generations=50):
    """2. Чувствительность к весам критериев — запускает GA для разных сдвигов весов"""
    print("\n=== 2. Чувствительность целевой функции к весам ===")
    print(f"{'Δ':>6} | {'w_idle':>6} {'w_empty':>7} {'w_mass':>7} | {'fitness':>10}")
    print("-" * 45)
    for d in deltas:
        w_idle = base_weights[0] + d
        w_empty = base_weights[1] + d
        w_mass = base_weights[2] - 2 * d  # пробуем компенсировать
        # нормируем, чтобы сумма была 1 и неотрицательные
        ws = [max(0.0, x) for x in (w_idle, w_empty, w_mass)]
        s = sum(ws)
        if s == 0:
            ws = list(base_weights)
        else:
            ws = [x / s for x in ws]

        ga = GeneticAlgorithm(locomotives, trains,
                              population_size=population_size,
                              generations=generations,
                              tournament_k=5,
                              mutation_rate=0.1,
                              weights=tuple(ws))
        try:
            sol = ga.run()
            fitness_val = sol.fitness
        except Exception as e:
            fitness_val = float("nan")
            print(f"Ошибка при запуске GA для d={d}: {e}")
        print(f"{d:6.2f} | {ws[0]:6.2f} {ws[1]:7.2f} {ws[2]:7.2f} | {fitness_val:10.2f}")

def run_experiment(ga: GeneticAlgorithm, baseline_solution: Chromosome, heuristic_solution: Chromosome):
    start = time.time()
    ga_solution = ga.run()
    ga_time = time.time() - start

    return {
        "baseline_fitness": getattr(baseline_solution, "_fitness", None),
        "heuristic_fitness": getattr(heuristic_solution, "_fitness", None),
        "ga_fitness": ga_solution.fitness,
        "ga_time_sec": ga_time
    }

# Табличные и печатные функции
def print_assignment_table(solution: Chromosome, locomotives: Dict[int, Locomotive], trains: Dict[int, Train]):
    print("\nРезультаты назначения локомотивов:\n")
    for loco_id, train_ids in solution.assignment.items():
        loco = locomotives.get(loco_id)
        if loco is None:
            print(f"Локомотив {loco_id} (нет данных)")
            continue
        print(f"Локомотив {loco_id} | Депо {loco.home_depot} | Остаточный ресурс: {loco.remaining_resource:.1f}")
        for t_id in train_ids:
            t = trains[t_id]
            print(f"  Поезд {t.id}: {t.route[0]} → {t.route[1]}, отпр {t.departure_time:.1f}, длит {t.duration:.1f}")
        print()

def print_locomotive_summary(locomotives: Dict[int, Locomotive],
                             trains: Dict[int, Train],
                             solution: Chromosome) -> None:
    print("\n=== Сводная таблица локомотивов ===")
    print(f"{'ID':>3} | {'Депо приписки':<12} | {'Текущее место':<12} | {'Остаток пробега':<15}")
    print("-" * 65)
    for loco_id, train_ids in solution.assignment.items():
        loco = locomotives.get(loco_id)
        if loco is None:
            continue
        if train_ids:
            last_train = trains[train_ids[-1]]
            current_location = last_train.route[1]
            remaining_resource = loco.remaining_resource - sum(trains[t_id].duration for t_id in train_ids)
        else:
            current_location = loco.home_depot
            remaining_resource = loco.remaining_resource
        print(f"{loco.id:3} | {loco.home_depot:<12} | {current_location:<12} | {remaining_resource:15.1f}")

def print_fitness_components(locomotives: Dict[int, Locomotive],
                             trains: Dict[int, Train],
                             solution: Chromosome) -> None:
    idle_count = calculate_idle_time(solution)
    empty_run = calculate_empty_run(solution)
    total_locos = len(locomotives)
    used_locos = sum(1 for lst in solution.assignment.values() if lst)
    unused_locos = total_locos - used_locos
    print("\n=== Элементы функции пригодности ===")
    print(f"Простой локомотивов (без работы): {idle_count}")
    print(f"Порожние пробеги (оценка разрывов): {empty_run}")
    print(f"Локомотивов в работе: {used_locos}")
    print(f"Локомотивов неиспользовано: {unused_locos}")

def print_detailed_plan(locomotives: Dict[int, Locomotive],
                        trains: Dict[int, Train],
                        solution: Chromosome) -> None:
    print("\n=== План назначения локомотивов ===")
    for loco_id, train_ids in solution.assignment.items():
        loco = locomotives.get(loco_id)
        if loco is None:
            continue
        print(f"\nЛокомотив {loco_id}  (депо приписки: {loco.home_depot})")
        print(f"  Начало работы:")
        print(f"    Депо начала работы: {loco.home_depot}")
        print(f"    Остаточный ресурс (начало): {loco.remaining_resource:.1f} ч")
        if not train_ids:
            print("    Назначений нет")
        else:
            print("    Назначения:")
            for t_id in train_ids:
                t = trains[t_id]
                print(f"      Поезд {t_id}:  {t.route[0]} → {t.route[1]}  отпр {t.departure_time:.1f} ч  приб {t.departure_time + t.duration:.1f} ч")
        if train_ids:
            last_train = trains[train_ids[-1]]
            current_depot = last_train.route[1]
            remaining_res = loco.remaining_resource - sum(trains[t_id].duration for t_id in train_ids)
        else:
            current_depot = loco.home_depot
            remaining_res = loco.remaining_resource
        print(f"  Конец работы:")
        print(f"    Текущее депо: {current_depot}")
        print(f"    Остаточный ресурс (конец): {remaining_res:.1f} ч")

def plot_assignment(solution: Chromosome, trains: Dict[int, Train]):
    fig, ax = plt.subplots()
    y = 0
    for loco_id, train_ids in solution.assignment.items():
        for t_id in train_ids:
            t = trains[t_id]
            ax.barh(y, t.duration, left=t.departure_time, height=0.4)
        y += 1
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Локомотивы")
    ax.set_title("Прогнозный график назначения локомотивов")
    plt.show()

if __name__ == "__main__":
    # Блок запуска — пытаемся использовать streamlit, но корректно падаем на печать
    try:
        import streamlit as st
    except Exception:
        class _FakeST:
            @staticmethod
            def error(msg): print("ERROR:", msg)
            @staticmethod
            def success(msg): print("OK:", msg)
            @staticmethod
            def stop(): raise SystemExit()
        st = _FakeST()

    try:
        locomotives, trains = generate_synthetic_data()
        ga = GeneticAlgorithm(locomotives, trains,
                              population_size=60,
                              generations=120,
                              tournament_k=5,
                              mutation_rate=0.15,
                              weights=(0.4, 0.3, 0.3))
        reporter = GAReporter()
        reporter.start()
        solution = ga.run(reporter=reporter)
    except Exception as e:
        st.error(f"Ошибка в ga.run(): {e}")
        # если streamlit — остановка, если нет — печатаем и выходим
        try:
            st.stop()
        except SystemExit:
            pass
    else:
        st.success("Алгоритм завершён")
        print_assignment_table(solution, locomotives, trains)

    # Отчёты
    try:
        reporter.print_summary(solution)      # таблица 1
    except Exception:
        pass

    try:
        sensitivity_table(locomotives, trains, population_size=30, generations=30)  # таблица 2 (ускорённый вариант)
    except Exception as e:
        print("Ошибка при формировании sensitivity_table:", e)

    try:
        reporter.print_generation_curve()     # таблица 3
    except Exception:
        pass

    # Дополнительные отчёты и визуализация
    try:
        print_assignment_table(solution, locomotives, trains)
        plot_assignment(solution, trains)
        print_locomotive_summary(locomotives, trains, solution)
        print_fitness_components(locomotives, trains, solution)
        print_detailed_plan(locomotives, trains, solution)
    except Exception as e:
        print("Ошибка при выводе результатов:", e)
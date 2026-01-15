# Loco_GA
Проверка
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA для оперативного назначения локомотивов (один файл)
Реализация опирается на содержание главы 3 диссертации: кодирование, операторы,
инициализация, repair, многокритериальная целевая функция и экспериментальный модуль.
Запуск: python ga_locomotive_assignment.py
Требования: Python 3.8+, numpy, pandas, matplotlib, tqdm (опционально)
"""

import random
import math
import time
from collections import defaultdict, namedtuple
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import trange
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ---------------------------
# Структуры данных
# ---------------------------

Train = namedtuple('Train', [
    'id',             # уникальный идентификатор поезда
    'origin',         # станция отправления (индекс)
    'destination',    # станция прибытия (индекс)
    'dep_time',       # время отправления (минуты от начала горизонта)
    'arr_time',       # время прибытия (минуты)
    'mass',           # масса/нагрузка (тонны)
    'req_traction'    # требуемая тяга (условная единица)
])

Locomotive = namedtuple('Locomotive', [
    'id',             # id локомотива
    'home_depot',     # индекс депо приписки
    'traction',       # тяговая характеристика (макс. допустимая нагрузка)
    'location',       # текущее местоположение (станция индекс)
    'available_from', # время, с которого локомотив доступен (мин)
    'since_maintenance', # пробег/время с последнего ТО (условная мера)
    'max_before_maintenance' # допустимый пробег/время до ТО
])

# ---------------------------
# Вспомогательные функции
# ---------------------------

def minutes(h, m=0):
    """Удобство: часы->минуты"""
    return h * 60 + m

# ---------------------------
# Класс задачи: генерация синтетических данных
# ---------------------------

class ProblemData:
    """
    Хранит входные данные задачи: поезда, локомотивы, матрицу времени/расстояний.
    Включает генератор синтетических данных для экспериментов.
    """
    def __init__(self, n_trains=50, n_locs=10, n_stations=20, horizon_minutes=24*60, seed=42):
        self.n_trains = n_trains
        self.n_locs = n_locs
        self.n_stations = n_stations
        self.horizon = horizon_minutes
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.stations = list(range(n_stations))
        self.trains = []
        self.locomotives = []
        self.time_matrix = None  # matrix of travel times between stations (min)
        self._generate_time_matrix()
        self._generate_locomotives()
        self._generate_trains()

    def _generate_time_matrix(self):
        """Генерация симметричной матрицы времени между станциями"""
        n = self.n_stations
        base = np.random.randint(20, 180, size=(n, n))
        mat = (base + base.T) // 2
        np.fill_diagonal(mat, 0)
        self.time_matrix = mat.astype(int)

    def _generate_locomotives(self):
        """Генерация локомотивов с разными тяговыми характеристиками и депо"""
        for i in range(self.n_locs):
            home = random.randrange(self.n_stations)
            traction = random.choice([1000, 1500, 2000])  # условные единицы
            location = home if random.random() < 0.8 else random.randrange(self.n_stations)
            available_from = random.randint(0, 6*60)  # доступны в первые 6 часов
            since_maint = random.randint(0, 5000)
            max_before_maint = random.randint(8000, 15000)
            loco = Locomotive(
                id=i,
                home_depot=home,
                traction=traction,
                location=location,
                available_from=available_from,
                since_maintenance=since_maint,
                max_before_maintenance=max_before_maint
            )
            self.locomotives.append(loco)

    def _generate_trains(self):
        """Генерация поездов с временными окнами и требованиями тяги"""
        for i in range(self.n_trains):
            o = random.randrange(self.n_stations)
            d = random.randrange(self.n_stations)
            while d == o:
                d = random.randrange(self.n_stations)
            # длительность поездки зависит от матрицы времени + случайный фактор
            base_time = self.time_matrix[o, d]
            # если base_time == 0 (редко), задаём минимальное
            if base_time == 0:
                base_time = random.randint(30, 180)
            dep = random.randint(0, self.horizon - base_time - 60)
            arr = dep + base_time + random.randint(-10, 30)
            mass = random.choice([500, 1000, 1500, 2000])  # тонны
            # требуемая тяга пропорциональна массе с дискретизацией
            req_tr = 1000 if mass <= 800 else (1500 if mass <= 1500 else 2000)
            train = Train(
                id=i,
                origin=o,
                destination=d,
                dep_time=dep,
                arr_time=arr,
                mass=mass,
                req_traction=req_tr
            )
            self.trains.append(train)

    def to_dataframe(self):
        """Возвращает DataFrame поездов и локомотивов для вывода"""
        df_tr = pd.DataFrame([t._asdict() for t in self.trains])
        df_loc = pd.DataFrame([l._asdict() for l in self.locomotives])
        return df_tr, df_loc

# ---------------------------
# Кодирование хромосомы
# ---------------------------

"""
Выбор кодирования:
- Простое и воспроизводимое кодирование: массив длины n_trains, где значение = id локомотива,
  либо -1 если поезд не назначен.
- Альтернативное кодирование (для специализированного crossover): список списков (последовательности оборотов)
  для каждого локомотива. В реализации используем основное кодирование (train->loc) и вспомогательные
  функции для преобразования в обороты.
Преимущество: простота операторов и оценки; недостаток: сложнее сохранять последовательности оборотов при кроссовере.
"""

# ---------------------------
# Оценка решения и целевая функция
# ---------------------------

class Evaluator:
    """
    Оценивает решение (assignment: array length n_trains -> loco_id or -1).
    Вычисляет компоненты целевой функции:
      - deadhead: суммарные порожние пробеги (в минутах или условных единицах)
      - idle_time: суммарное время простоя локомотивов между назначениями
      - penalty_violations: штрафы за несоответствие тяги, временные конфликты, ТО и т.д.
      - imbalance: мера неравномерности загрузки парка (std или разница)
    """
    def __init__(self, data: ProblemData, weights=None):
        self.data = data
        # веса для компонентов: deadhead, idle_time, penalties, imbalance
        if weights is None:
            self.weights = {'deadhead': 1.0, 'idle_time': 0.5, 'penalty': 1000.0, 'imbalance': 0.1}
        else:
            self.weights = weights

    def evaluate(self, assignment):
        """
        assignment: list/np.array длины n_trains, значения в [0..n_locs-1] или -1 (unassigned)
        Возвращает: fitness_value (меньше лучше), dict с компонентами
        """
        n_trains = len(self.data.trains)
        n_locs = len(self.data.locomotives)
        # Инициализация статистик по локомотивам
        loco_schedules = {l.id: [] for l in self.data.locomotives}
        penalties = 0.0

        # Проверка соответствия тяги и простых штрафов
        for t_idx, loco_id in enumerate(assignment):
            train = self.data.trains[t_idx]
            if loco_id is None or loco_id == -1:
                # штраф за не назначенный поезд (высокий)
                penalties += 10000.0
                continue
            loco = self.data.locomotives[loco_id]
            # тяга
            if loco.traction < train.req_traction:
                penalties += 5000.0  # большой штраф
            # добавляем поезд в расписание локомотива
            loco_schedules[loco_id].append(train)

        # Для каждого локомотива упорядочим по времени отправления и проверим конфликты
        total_deadhead = 0.0
        total_idle = 0.0
        loads = []
        for loco_id, trains in loco_schedules.items():
            if not trains:
                loads.append(0)
                continue
            # сортировка по dep_time
            trains_sorted = sorted(trains, key=lambda x: x.dep_time)
            loads.append(len(trains_sorted))
            # начальная позиция и время локомотива
            loco = self.data.locomotives[loco_id]
            cur_loc = loco.location
            cur_time = loco.available_from
            # пробег/время с последнего ТО (условно учитываем как суммарное время работы)
            since_maint = loco.since_maintenance
            max_before_maint = loco.max_before_maintenance
            for tr in trains_sorted:
                # время, чтобы подвести локомотив к месту отправления
                travel_to_origin = self.data.time_matrix[cur_loc, tr.origin]
                # если локомотив не успевает прибыть до dep_time -> штраф
                arrival_time = cur_time + travel_to_origin
                if arrival_time > tr.dep_time:
                    # штраф за опоздание на подводку
                    penalties += 2000.0 + (arrival_time - tr.dep_time) * 5.0
                # порожний пробег (deadhead) в условных единицах: время * коэффициент
                dead = travel_to_origin
                total_deadhead += dead
                # время простоя между прибытия и отправлением (если прибыл раньше)
                idle = max(0, tr.dep_time - arrival_time)
                total_idle += idle
                # обновляем текущее состояние после выполнения поезда
                # время прибытия локомотива в пункт назначения
                cur_time = tr.arr_time
                cur_loc = tr.destination
                # увеличиваем since_maint условно на время работы
                since_maint += (tr.arr_time - tr.dep_time)
            # после всех поездов локомотив должен вернуться в депо (или быть в депо)
            travel_home = self.data.time_matrix[cur_loc, loco.home_depot]
            total_deadhead += travel_home
            # проверка ТО: если since_maint + суммар > max_before_maint -> штраф
            if since_maint > max_before_maint:
                penalties += 5000.0 + (since_maint - max_before_maint) * 0.1

        # imbalance: стандартное отклонение загрузки (чем меньше, тем лучше)
        if loads:
            imbalance = float(np.std(loads))
        else:
            imbalance = 0.0

        # Скаляризация многокритериальной функции
        w = self.weights
        fitness = (w['deadhead'] * total_deadhead +
                   w['idle_time'] * total_idle +
                   w['penalty'] * penalties +
                   w['imbalance'] * imbalance)

        components = {
            'deadhead': total_deadhead,
            'idle_time': total_idle,
            'penalties': penalties,
            'imbalance': imbalance
        }
        return fitness, components

# ---------------------------
# Генетический алгоритм
# ---------------------------

class GeneticAlgorithm:
    """
    Основной класс GA. Поддерживает:
      - инициализацию (random, greedy, hybrid)
      - операторы: crossover (one-point), sequence crossover (для оборотов), mutation
      - repair после операторов
      - селекцию (турнир), элитизм, критерии останова
    """
    def __init__(self, data: ProblemData, pop_size=100, generations=200,
                 p_crossover=0.8, p_mutation=0.2, elite_size=2,
                 weights=None, seed=42, verbose=True):
        self.data = data
        self.pop_size = pop_size
        self.generations = generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.elite_size = elite_size
        self.weights = weights
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        np.random.seed(seed)
        self.evaluator = Evaluator(data, weights=weights)
        # population: list of assignments (list of ints)
        self.population = []
        self.fitness_cache = []
        self.history = {'best': [], 'mean': [], 'components': []}

    # -----------------------
    # Инициализация популяции
    # -----------------------
    def init_population(self, method='hybrid'):
        """
        method: 'random', 'greedy', 'hybrid'
        """
        self.population = []
        n_tr = len(self.data.trains)
        n_loc = len(self.data.locomotives)
        if method == 'random':
            for _ in range(self.pop_size):
                # случайное назначение, но с учётом тяги (иначе будут штрафы)
                assign = []
                for t in self.data.trains:
                    # вероятность назначить поезд
                    if random.random() < 0.95:
                        # выбираем случайный локомотив, который удовлетворяет тяге с вероятностью 0.8
                        candidates = [l.id for l in self.data.locomotives if l.traction >= t.req_traction]
                        if candidates and random.random() < 0.9:
                            assign.append(random.choice(candidates))
                        else:
                            assign.append(random.randrange(n_loc))
                    else:
                        assign.append(-1)
                self.population.append(assign)
        elif method == 'greedy':
            # жадный: для каждого поезда назначаем ближайший по времени и месту локомотив, удовлетворяющий тяге
            for _ in range(self.pop_size):
                assign = [-1] * n_tr
                # копии состояния локомотивов
                loco_state = {l.id: {'loc': l.location, 'time': l.available_from} for l in self.data.locomotives}
                for t_idx, t in enumerate(sorted(self.data.trains, key=lambda x: x.dep_time)):
                    # кандидаты по тяге
                    candidates = []
                    for l in self.data.locomotives:
                        if l.traction >= t.req_traction:
                            travel = self.data.time_matrix[loco_state[l.id]['loc'], t.origin]
                            arrival = loco_state[l.id]['time'] + travel
                            wait = max(0, t.dep_time - arrival)
                            score = arrival + wait  # чем меньше, тем лучше
                            candidates.append((score, l.id))
                    if candidates:
                        candidates.sort()
                        chosen = candidates[0][1]
                        assign[t_idx] = chosen
                        # обновляем состояние выбранного локомотива
                        loco_state[chosen]['time'] = t.arr_time
                        loco_state[chosen]['loc'] = t.destination
                    else:
                        assign[t_idx] = -1
                self.population.append(assign)
        elif method == 'hybrid':
            # часть случайных, часть жадных
            n_rand = int(self.pop_size * 0.4)
            n_greedy = self.pop_size - n_rand
            # случайные
            for _ in range(n_rand):
                assign = []
                for t in self.data.trains:
                    if random.random() < 0.95:
                        candidates = [l.id for l in self.data.locomotives if l.traction >= t.req_traction]
                        if candidates and random.random() < 0.9:
                            assign.append(random.choice(candidates))
                        else:
                            assign.append(random.randrange(len(self.data.locomotives)))
                    else:
                        assign.append(-1)
                self.population.append(assign)
            # жадные
            for _ in range(n_greedy):
                assign = [-1] * n_tr
                loco_state = {l.id: {'loc': l.location, 'time': l.available_from} for l in self.data.locomotives}
                for t_idx, t in enumerate(sorted(self.data.trains, key=lambda x: x.dep_time)):
                    candidates = []
                    for l in self.data.locomotives:
                        if l.traction >= t.req_traction:
                            travel = self.data.time_matrix[loco_state[l.id]['loc'], t.origin]
                            arrival = loco_state[l.id]['time'] + travel
                            wait = max(0, t.dep_time - arrival)
                            score = arrival + wait
                            candidates.append((score, l.id))
                    if candidates:
                        candidates.sort()
                        chosen = candidates[0][1]
                        assign[t_idx] = chosen
                        loco_state[chosen]['time'] = t.arr_time
                        loco_state[chosen]['loc'] = t.destination
                    else:
                        assign[t_idx] = -1
                self.population.append(assign)
        else:
            raise ValueError("Unknown init method")
        # вычислим fitness для начальной популяции
        self._evaluate_population()

    # -----------------------
    # Операторы: crossover
    # -----------------------
    def one_point_crossover(self, parent1, parent2):
        """One-point crossover для массива назначений"""
        n = len(parent1)
        pt = random.randint(1, n - 1)
        child1 = parent1[:pt] + parent2[pt:]
        child2 = parent2[:pt] + parent1[pt:]
        return child1, child2

    def sequence_crossover(self, parent1, parent2):
        """
        Специализированный crossover, пытающийся сохранить последовательности оборотов.
        Подход: выбираем случайный локомотив L, берем все поезда, назначенные L в parent1,
        и пытаемся вставить их в child на те же позиции, остальные заполняем из parent2.
        """
        n = len(parent1)
        child = [-1] * n
        # выбираем локомотив
        loco_ids = [l.id for l in self.data.locomotives]
        chosen_loco = random.choice(loco_ids)
        # копируем все поезда, назначенные chosen_loco в parent1
        for i, a in enumerate(parent1):
            if a == chosen_loco:
                child[i] = chosen_loco
        # заполняем остальные позициями из parent2, если не конфликтуют по тяге
        for i, a in enumerate(parent2):
            if child[i] == -1:
                # проверка тяги
                train = self.data.trains[i]
                if a != -1 and self.data.locomotives[a].traction >= train.req_traction:
                    child[i] = a
        # оставшиеся незаполненные - назначаем случайно допустимым локомотивом
        for i in range(n):
            if child[i] == -1:
                train = self.data.trains[i]
                candidates = [l.id for l in self.data.locomotives if l.traction >= train.req_traction]
                if candidates:
                    child[i] = random.choice(candidates)
                else:
                    child[i] = -1
        return child

    # -----------------------
    # Мутации
    # -----------------------
    def mutation_random_replace(self, individual, p_replace=0.05):
        """Случайная замена назначений с вероятностью p_replace"""
        n = len(individual)
        child = individual.copy()
        for i in range(n):
            if random.random() < p_replace:
                train = self.data.trains[i]
                candidates = [l.id for l in self.data.locomotives if l.traction >= train.req_traction]
                if candidates:
                    child[i] = random.choice(candidates)
                else:
                    child[i] = -1
        return child

    def mutation_local_swap(self, individual, p_swap=0.02):
        """Локальные перестановки: меняем назначение двух случайных поездов"""
        child = individual.copy()
        n = len(child)
        for _ in range(int(n * p_swap) + 1):
            i, j = random.randrange(n), random.randrange(n)
            child[i], child[j] = child[j], child[i]
        return child

    def mutation_depot_aware(self, individual, p_move=0.03):
        """
        Перемещение оборота между локомотивами с учётом депо:
        пытаемся переназначить поезд на локомотив из того же депо, чтобы уменьшить deadhead.
        """
        child = individual.copy()
        for i, a in enumerate(child):
            if random.random() < p_move:
                train = self.data.trains[i]
                # локомотивы из депо, совпадающего с origin станции
                candidates = [l.id for l in self.data.locomotives if l.traction >= train.req_traction]
                if not candidates:
                    continue
                # выбираем локомотив с home_depot ближе к origin
                best = min(candidates, key=lambda lid: self.data.time_matrix[self.data.locomotives[lid].home_depot, train.origin])
                child[i] = best
        return child

    # -----------------------
    # Repair / корректировка
    # -----------------------
    def repair(self, individual):
        """
        Попытка исправить недопустимые назначения:
         - если локомотив не удовлетворяет тяге -> переназначить на допустимый
         - если конфликт по времени (локомотив назначен на два перекрывающихся поезда) -> попытаться переназначить один из них
        Примечание: repair не гарантирует устранение всех штрафов, но уменьшает грубые нарушения.
        """
        n = len(individual)
        child = individual.copy()
        # 1) тяга
        for i, a in enumerate(child):
            if a == -1:
                continue
            train = self.data.trains[i]
            loco = self.data.locomotives[a]
            if loco.traction < train.req_traction:
                # ищем кандидата с достаточной тягой
                candidates = [l.id for l in self.data.locomotives if l.traction >= train.req_traction]
                if candidates:
                    child[i] = random.choice(candidates)
                else:
                    child[i] = -1
        # 2) временные конфликты: для каждого локомотива проверяем перекрытия
        loco_assignments = defaultdict(list)
        for i, a in enumerate(child):
            if a == -1:
                continue
            loco_assignments[a].append(self.data.trains[i])
        for loco_id, trains in loco_assignments.items():
            # сортировка по dep_time
            trains_sorted = sorted(trains, key=lambda x: x.dep_time)
            cur_time = self.data.locomotives[loco_id].available_from
            cur_loc = self.data.locomotives[loco_id].location
            for tr in trains_sorted:
                travel = self.data.time_matrix[cur_loc, tr.origin]
                arrival = cur_time + travel
                if arrival > tr.dep_time:
                    # конфликт: попробуем переназначить этот поезд на другой локомотив
                    # находим кандидата, который может обслужить поезд и доступен
                    candidates = []
                    for l in self.data.locomotives:
                        if l.traction >= tr.req_traction:
                            # грубая проверка доступности: доступен ли локомотив до dep_time
                            # (используем его available_from и местоположение)
                            travel2 = self.data.time_matrix[l.location, tr.origin]
                            if l.available_from + travel2 <= tr.dep_time + 60:  # допускаем небольшой запас
                                candidates.append(l.id)
                    if candidates:
                        # выбираем случайного кандидата
                        new_l = random.choice(candidates)
                        # найти индекс поезда в child и переназначить
                        for idx, t in enumerate(self.data.trains):
                            if t.id == tr.id:
                                child[idx] = new_l
                                break
                        # обновляем локальные переменные: не продолжаем проверять этот tr для старого loco
                    else:
                        # не удалось переназначить -> оставляем и штрафуем позже
                        pass
                # обновляем cur_time/cur_loc для следующей итерации
                cur_time = tr.arr_time
                cur_loc = tr.destination
        return child

    # -----------------------
    # Селекция и замещение
    # -----------------------
    def tournament_selection(self, k=3):
        """Турнирная селекция: возвращает одного индивида (копию)"""
        best = None
        for _ in range(k):
            idx = random.randrange(len(self.population))
            if best is None or self.fitness_cache[idx] < self.fitness_cache[best]:
                best = idx
        return deepcopy(self.population[best])

    # -----------------------
    # Вспомогательные: оценка популяции
    # -----------------------
    def _evaluate_population(self):
        self.fitness_cache = []
        comps = []
        for ind in self.population:
            f, c = self.evaluator.evaluate(ind)
            self.fitness_cache.append(f)
            comps.append(c)
        return self.fitness_cache, comps

    # -----------------------
    # Основной цикл GA
    # -----------------------
    def run(self, init_method='hybrid', verbose=True, adapt_mutation=True, time_limit=None):
        """
        Запуск GA.
        Параметры:
          - init_method: 'random', 'greedy', 'hybrid'
          - adapt_mutation: уменьшать p_mutation по поколениям
          - time_limit: ограничение времени в секундах (опционально)
        Возвращает: best_solution, best_fitness, history
        """
        start_time = time.time()
        self.init_population(method=init_method)
        best_idx = int(np.argmin(self.fitness_cache))
        best_fit = self.fitness_cache[best_idx]
        best_sol = deepcopy(self.population[best_idx])

        gen_iter = range(self.generations)
        if _HAS_TQDM and verbose:
            gen_iter = trange(self.generations, desc='GA generations')

        no_improve = 0
        best_history = []

        for gen in gen_iter:
            if time_limit and (time.time() - start_time) > time_limit:
                if verbose:
                    print("Time limit reached, stopping.")
                break
            new_pop = []
            # элитизм: сохраняем лучших
            sorted_idx = np.argsort(self.fitness_cache)
            elites = [deepcopy(self.population[i]) for i in sorted_idx[:self.elite_size]]
            new_pop.extend(elites)

            # адаптация вероятности мутации
            if adapt_mutation:
                p_mut = self.p_mutation * (1.0 - gen / max(1, self.generations))
            else:
                p_mut = self.p_mutation

            # генерируем новых потомков
            while len(new_pop) < self.pop_size:
                # селекция родителей
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                # crossover
                if random.random() < self.p_crossover:
                    if random.random() < 0.5:
                        child1, child2 = self.one_point_crossover(parent1, parent2)
                    else:
                        child1 = self.sequence_crossover(parent1, parent2)
                        child2 = self.sequence_crossover(parent2, parent1)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                # мутации
                if random.random() < p_mut:
                    child1 = self.mutation_random_replace(child1, p_replace=0.05)
                    child1 = self.mutation_local_swap(child1, p_swap=0.02)
                    child1 = self.mutation_depot_aware(child1, p_move=0.02)
                if random.random() < p_mut:
                    child2 = self.mutation_random_replace(child2, p_replace=0.05)
                    child2 = self.mutation_local_swap(child2, p_swap=0.02)
                    child2 = self.mutation_depot_aware(child2, p_move=0.02)
                # repair
                child1 = self.repair(child1)
                child2 = self.repair(child2)
                new_pop.append(child1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(child2)

            self.population = new_pop
            # оценка
            fitness_vals, comps = self._evaluate_population()
            best_idx = int(np.argmin(fitness_vals))
            mean_fit = float(np.mean(fitness_vals))
            best_fit_gen = fitness_vals[best_idx]
            best_sol_gen = deepcopy(self.population[best_idx])
            # сохраняем историю
            self.history['best'].append(best_fit_gen)
            self.history['mean'].append(mean_fit)
            self.history['components'].append(comps[best_idx])
            best_history.append((best_fit_gen, deepcopy(self.population[best_idx])))

            # обновление глобального лучшего
            if best_fit_gen < best_fit:
                best_fit = best_fit_gen
                best_sol = deepcopy(best_sol_gen)
                no_improve = 0
            else:
                no_improve += 1

            # критерий останова: отсутствие улучшения за 50 поколений
            if no_improve >= 50:
                if verbose:
                    print(f"No improvement for {no_improve} generations, stopping at gen {gen}.")
                break

        total_time = time.time() - start_time
        # финальная оценка лучшего решения
        best_fitness, best_components = self.evaluator.evaluate(best_sol)
        if verbose:
            print(f"GA finished in {total_time:.2f}s. Best fitness: {best_fitness:.2f}")
            print("Best components:", best_components)
        return best_sol, best_fitness, best_components, self.history

# ---------------------------
# Вспомогательные функции вывода и визуализации
# ---------------------------

def plot_history(history, title='GA fitness dynamics'):
    plt.figure(figsize=(10, 5))
    plt.plot(history['best'], label='best')
    plt.plot(history['mean'], label='mean')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (lower better)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def schedule_to_dataframe(data: ProblemData, assignment):
    """Преобразует assignment в DataFrame с информацией по каждому поезду"""
    rows = []
    for i, a in enumerate(assignment):
        t = data.trains[i]
        rows.append({
            'train_id': t.id,
            'origin': t.origin,
            'destination': t.destination,
            'dep_time': t.dep_time,
            'arr_time': t.arr_time,
            'mass': t.mass,
            'req_traction': t.req_traction,
            'assigned_loco': a
        })
    df = pd.DataFrame(rows)
    return df

def plot_deadhead_distribution(data: ProblemData, assignment):
    """Простейшая оценка распределения порожних пробегов по локомотивам"""
    loco_dead = defaultdict(float)
    loco_loc = {l.id: l.location for l in data.locomotives}
    loco_time = {l.id: l.available_from for l in data.locomotives}
    for i, a in enumerate(assignment):
        if a == -1:
            continue
        t = data.trains[i]
        travel_to_origin = data.time_matrix[loco_loc[a], t.origin]
        loco_dead[a] += travel_to_origin
        loco_time[a] = t.arr_time
        loco_loc[a] = t.destination
    # добавим возврат в депо
    for l in data.locomotives:
        travel_home = data.time_matrix[loco_loc[l.id], l.home_depot]
        loco_dead[l.id] += travel_home
    ids = list(loco_dead.keys())
    vals = [loco_dead[i] for i in ids]
    plt.figure(figsize=(8,4))
    plt.bar([str(i) for i in ids], vals)
    plt.xlabel('Locomotive id')
    plt.ylabel('Deadhead (time units)')
    plt.title('Распределение порожних пробегов по локомотивам')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------
# Экспериментальный модуль
# ---------------------------

def run_experiment(seed=42, n_trains=50, n_locs=10, generations=200, pop_size=120):
    print("Генерация синтетических данных...")
    data = ProblemData(n_trains=n_trains, n_locs=n_locs, seed=seed)
    df_tr, df_loc = data.to_dataframe()
    print(f"Сгенерировано {len(df_tr)} поездов и {len(df_loc)} локомотивов.")
    # Параметры GA
    weights = {'deadhead': 1.0, 'idle_time': 0.5, 'penalty': 1000.0, 'imbalance': 0.2}
    ga = GeneticAlgorithm(data,
                          pop_size=pop_size,
                          generations=generations,
                          p_crossover=0.85,
                          p_mutation=0.25,
                          elite_size=4,
                          weights=weights,
                          seed=seed,
                          verbose=True)
    print("Инициализация и запуск GA...")
    best_sol, best_fit, best_components, history = ga.run(init_method='hybrid', verbose=True, adapt_mutation=True, time_limit=None)
    print("Лучшее значение целевой функции:", best_fit)
    print("Компоненты:", best_components)
    # Вывод расписания
    df_schedule = schedule_to_dataframe(data, best_sol)
    print("\nПример назначений (первые 20 строк):")
    print(df_schedule.head(20).to_string(index=False))
    # Визуализации
    plot_history(history, title='Динамика пригодности GA')
    plot_deadhead_distribution(data, best_sol)
    # Ключевые метрики
    evaluator = Evaluator(data, weights=weights)
    fitness, comps = evaluator.evaluate(best_sol)
    print("\nКлючевые метрики лучшего решения:")
    print(f"Fitness: {fitness:.2f}")
    print(f"Deadhead (суммарный): {comps['deadhead']:.2f}")
    print(f"Idle time (суммарный): {comps['idle_time']:.2f}")
    print(f"Penalties: {comps['penalties']:.2f}")
    print(f"Imbalance (std of loads): {comps['imbalance']:.2f}")
    return data, best_sol, df_schedule, history

# ---------------------------
# Рекомендации по использованию
# ---------------------------
USAGE_RECOMMENDATIONS = """
Рекомендации по использованию и тюнингу:
- При увеличении числа поездов (n_trains): увеличьте pop_size и generations, например pop_size ~ 2-4 * n_trains.
- При увеличении числа локомотивов: можно уменьшить pop_size, но усилить repair-операторы.
- Критичные ограничения: соответствие тяги (жёсткое), временные конфликты и ТО. Repair уменьшает грубые нарушения, но не заменяет строгую валидацию.
- Для практического внедрения: интегрировать реальные данные (местоположения, точные времена, бригады, ТО), добавить жёсткие ограничения (hard constraints) и гибридизировать GA с локальным поиском (memetic).
- Для многокритериальной оптимизации: рассмотреть Pareto-GA (NSGA-II) вместо скаляризации весами.
"""

# ---------------------------
# Точка входа
# ---------------------------

if __name__ == '__main__':
    # Параметры эксперимента можно менять здесь
    SEED = 2026
    N_TRAINS = 50
    N_LOCOMOTIVES = 10
    GENERATIONS = 200
    POP_SIZE = 120

    data, best_sol, df_schedule, history = run_experiment(seed=SEED,
                                                         n_trains=N_TRAINS,
                                                         n_locs=N_LOCOMOTIVES,
                                                         generations=GENERATIONS,
                                                         pop_size=POP_SIZE)
    print(USAGE_RECOMMENDATIONS)

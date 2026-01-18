```python
import math
import pytest
from Loco_GA import (
    generate_synthetic_data,
    Locomotive,
    Train,
    Chromosome,
    is_feasible_fast,
    fitness_function,
    GeneticAlgorithm,
    calculate_empty_run_time,
    distance_to_time,
)


def test_generate_synthetic_data_shapes():
    locos, trains, coords = generate_synthetic_data(num_locomotives=5, num_trains=8, depot_names=("D1","D2","D3"), seed=42)
    assert len(locos) == 5
    assert len(trains) == 8
    assert len(coords) == 3
    for k, v in coords.items():
        assert len(v) == 3  # x,y,elev


def test_is_feasible_fast_basic():
    locos = {0: Locomotive(0, "t", power=5000, remaining_resource=10.0, home_depot="A")}
    trains = {0: Train(0, weight=4000, route=("A","B"), departure_time=0.0, duration=5.0)}
    chrom = Chromosome({0: [0]})
    assert is_feasible_fast(chrom, locos, trains) is True
    # overweight train
    trains2 = {0: Train(0, weight=6000, route=("A","B"), departure_time=0.0, duration=5.0)}
    chrom2 = Chromosome({0: [0]})
    assert is_feasible_fast(chrom2, locos, trains2) is False


def test_distance_to_time_and_empty_run_time():
    # two stations 100 km apart, small elevation diff
    a = (0.0, 0.0, 0.0)
    b = (100.0, 0.0, 50.0)  # 100 units distance, 50 elevation
    dist = math.hypot(100.0, 50.0)  # 3D hypotenuse
    t = distance_to_time(dist, speed_kmh=50.0, slope_elevation_diff=50.0, slope_penalty_coefficient=0.05)
    assert t > 0.0
    # test empty run time uses distance->time
    locos = {0: Locomotive(0, "t", power=8000, remaining_resource=50.0, home_depot="S1")}
    trains = {0: Train(0, weight=3000, route=("S1","S2"), departure_time=0.0, duration=2.0)}
    station_coords = {"S1": (0.0, 0.0, 0.0), "S2": (100.0, 0.0, 50.0)}
    chrom = Chromosome({0: [0]})
    et = calculate_empty_run_time(chrom, locos, trains, station_coords, reposition_speed_kmh=50.0)
    assert et > 0.0


def test_ga_seq_vs_parallel_consistency():
    locos, trains, station_coords = generate_synthetic_data(num_locomotives=6, num_trains=10, depot_names=("A","B","C"), seed=1)
    # run GA sequentially
    ga_seq = GeneticAlgorithm(locos, trains, population_size=20, generations=10, use_multiprocessing=False, multiprocessing_threshold=1000)
    best_seq = ga_seq.run()
    # run GA with multiprocessing (force by lowering threshold)
    ga_mp = GeneticAlgorithm(locos, trains, population_size=20, generations=10, use_multiprocessing=True, multiprocessing_threshold=1)
    best_mp = ga_mp.run()
    # fitness should be comparable (not necessarily identical due to randomness); check they are finite numbers
    assert math.isfinite(best_seq.fitness)
    assert math.isfinite(best_mp.fitness)
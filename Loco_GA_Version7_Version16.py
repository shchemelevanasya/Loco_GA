# -*- coding: utf-8 -*-
"""
Loco_GA — Генетический алгоритм назначения локомотивов

Полный файл, объединяющий все предыдущие изменения и исправления:
 - Вставлена функция generate_initial_population (устранение NameError)
 - Устойчивый выбор стиля matplotlib при отсутствии seaborn
 - Мутации: swap_locos, replace_loco, range_shuffle
 - Кроссоверы: one_point, two_point, uniform, priority
 - Адаптивная вероятность мутации и адаптивные веса кроссоверов
 - Возможность выбрать подмножество операторов кроссовера и задать вероятность для каждого
 - Поддержка Minimize/Maximize (опция в UI)
 - Streamlit UI (на русском): загрузка данных, выбор операторов, запуск GA, экспорт графиков/CSV
 - Все визуализации и подписи на русском; функции построения графиков устойчивы к разным структурам логов
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
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Попытка импортировать seaborn; безопасный fallback к доступным стилям matplotlib
try:
    import seaborn as sns  # type: ignore
    sns.set_style("whitegrid")
    USE_SEABORN = True
except Exception:
    USE_SEABORN = False
    _preferred_styles = ["seaborn-v0_8", "seaborn", "ggplot", "tableau-colorblind10", "classic", "default"]
    for _style in _preferred_styles:
        try:
            plt.style.use(_style)
            break
        except Exception:
            continue

# Глобальные настройки Matplotlib для академического качества изображений
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.constrained_layout.use": True
})

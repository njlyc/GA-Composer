import numpy as np
from typing import Literal
from src.modules.obj.obj_wrapper import OBJ_SGL_FUNCS, OBJ_SGL_OPT_TARGET
from src.utils.utils import _get_available_pitches
from src.modules.Translator import Translator


def register(name: str, min_max: Literal[1, -1]):
    def wrapper(func):
        OBJ_SGL_OPT_TARGET[name] = min_max
        OBJ_SGL_FUNCS[name] = func
        return func
    return wrapper
    
@register("naive", 1)
def naive_fitness(x: np.ndarray, rest_code: int, fermata_code: int, translator: Translator):
    return np.sum(x)

@register("continuous", -1)
def continuous_fitness(x: np.ndarray, rest_code: int, fermata_code: int, translator: Translator):
    rest_cnt = 0
    farmata_cnt = 0
    note_cnt = 0
    for c in x:
        if c == rest_code:
            rest_cnt += 1
        elif c == fermata_code:
            farmata_cnt += 1
        else:
            note_cnt += 1
    return farmata_cnt / len(x)

@register("continuous_pitch_only", -1)
def continuous_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    return np.var(x)

@register("random", 1)
def random_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    return np.random.random()

@register("no_octave_jump", -1)
def cn_g1_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return np.mean(np.abs(np.diff(x)) <= 12)

@register("only_CDEGA", -1)
def cn_g4_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    pitches = ["G3", "A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5",]
    good_idx = [translator.pitch2idx[pitch] for pitch in pitches]
    return np.bitwise_or.reduce([ (x == idx) for idx in good_idx ]).mean()

@register("not_monotonic", -1)
def cn_g5_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return np.abs(x[-1] - x[0]) <= len(x)

@register("not_flat", -1)
def cn_g6_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return (np.diff(x) != 0).mean()

@register("not_intense", 1)
def cn_g7_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return np.var(np.diff(x))

@register("l1_loss", 1)
def l1_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return np.abs(np.diff(x)).mean()

@register("reduce_rest", -1)
def reduce_rest(x: np.ndarray, rest_code: int, translator: Translator):
    return np.mean(x != rest_code)

@register("returning", 1)
def returning_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    N = len(x)
    return np.abs(x[:N//2].mean() - x[N//2:].mean())

@register("weighted", -1)
def weighted_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    return \
          cn_g4_fitness(x, rest_code, translator) \
        - cn_g7_fitness(x, rest_code, translator) * 0.001 \
        + cn_g6_fitness(x, rest_code, translator) * 0.3 \
        - reduce_rest(x, rest_code, translator) * 0.95 \
        + l1_fitness(x, rest_code, translator) * 0.007 \
        - returning_fitness(x, rest_code, translator) * 0.03 \


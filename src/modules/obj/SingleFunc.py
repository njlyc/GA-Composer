import numpy as np
from typing import Literal
from src.modules.obj.obj_wrapper import OBJ_SGL_FUNCS, OBJ_SGL_OPT_TARGET

def register(name: str, min_max: Literal[1, -1]):
    def wrapper(func):
        OBJ_SGL_OPT_TARGET[name] = min_max
        OBJ_SGL_FUNCS[name] = func
        return func
    return wrapper
    
@register("naive", 1)
def naive_fitness(x: np.ndarray, rest_code: int, fermata_code: int):
    return np.sum(x)

@register("continuous", -1)
def continuous_fitness(x: np.ndarray, rest_code: int, fermata_code: int):
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
def continuous_fitness(x: np.ndarray, rest_code: int):
    return np.var(x)
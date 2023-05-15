import numpy as np
from typing import Any, Literal, Dict, Callable, Tuple
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
def no_octave(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return np.mean(np.abs(np.diff(x)) <= 12)

@register("only_CDEGA", -1)
def only_CDEGA(x: np.ndarray, rest_code: int, translator: Translator):
    pitches = ["G3", "A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5",]
    good_idx = [translator.pitch2idx[pitch] for pitch in pitches]
    return np.bitwise_or.reduce([ (x == idx) for idx in good_idx ]).mean()

@register("not_monotonic", -1)
def not_monotonic(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return np.abs(x[-1] - x[0]) <= len(x)

@register("not_flat", -1)
def not_flat(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return (np.diff(x) != 0).mean()

@register("not_intense", 1)
def not_intense(x: np.ndarray, rest_code: int, translator: Translator):
    x = x[x != rest_code]
    return np.std(np.diff(x))

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


class semitones_fitness:
    
    _functions: Dict[int, Callable] = {}
        
    def __init__(self, semitones: int):
        self.semitones = semitones
        if semitones in self._functions.keys():
            return
        
        @register(f"semitone_{self.semitones}", -1)
        def _semitone_fun(x: np.ndarray, rest_code: int, translator: Translator):
            x = x[x != rest_code]
            return np.mean( np.abs(np.diff(x)) % 12 == semitones  )
        
        self._functions[semitones] = _semitone_fun
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._functions[self.semitones](*args, **kwds)
        

class trend_fitness:
    
    _functions: Dict[Tuple[float, float], Callable] = {}

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end
        if (self.start, self.end) in self._functions.keys():
            return 
        
        @register(f"trend_{start}_{end}", -1)
        def _trend_fun(x: np.ndarray, rest_code: int, translator: Translator):
            assert start >= 0 and end <= 1
            y = x[int(start * translator.encode_dim ): int(end * translator.encode_dim)]
            interval = np.sign(np.diff( y[y != 0] ))
            return np.mean(interval) if len(interval) else 0
        
        self._functions[(start, end)] = _trend_fun

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._functions[(self.start, self.end)](*args, **kwds)


@register("weighted", -1)
def weighted_fitness(x: np.ndarray, rest_code: int, translator: Translator):
    return np.sum([
        fun(x, rest_code, translator) * weight
        for weight, fun in [
           (  1.0 , semitones_fitness(2) ),
           (  1.0 , semitones_fitness(3) ),
           (  1.0 , semitones_fitness(5) ),
           (  0.3 , semitones_fitness(7) ),
           ( 10.0 , no_octave ),
           (  0.5 , trend_fitness(.0 , .25) ),
           ( -0.5 , trend_fitness(.25, .5 ) ),
           (  0.0 , trend_fitness(.50, .75) ),
           ( -0.5 , trend_fitness(.75, 1. ) ),
           (  1.0 , only_CDEGA ),
           ( -0.2 , not_intense)
        ]
    ])
          
          

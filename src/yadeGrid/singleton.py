from typing import Type, Callable, TypeVar, Any, Dict
from abc import ABCMeta

S = TypeVar('S')  # Type variable for the Singleton decorator


def Singleton(cls: Type[S]) -> Callable[..., S]:
    '''
    Singleton decorator. Usage:
        @Singleton
        class MyClass:
            pass
    '''
    instances = {}

    def get_instance(*args: Any, **kwargs: Any) -> S:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


T = TypeVar('T', bound='SingletonMeta')  # Separate TypeVar for SingletonMeta


class SingletonMeta(type):
    _instances: Dict[type, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self not in self._instances:
            self._instances[self] = super().__call__(*args, **kwargs)
        return self._instances[self]
    

class CombinedMeta(SingletonMeta, ABCMeta):
    pass

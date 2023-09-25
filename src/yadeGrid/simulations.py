# import numba
import numpy as np
from attrs import define, field
from abc import ABC, abstractmethod
from yadeGrid.interaction import Interaction
from yadeGrid.singleton import Singleton, SingletonMeta, CombinedMeta
from yadeGrid.body import Body, Quaternion
from yadeGrid.vectorFunc import norm
from yadeGrid.yadeTypes import Vector3D, F64
from typing import Callable, Any, Dict, Tuple, Iterator


def raise_on_modify(instance: Any, attribute: Any, value: Any) -> None:
    raise AttributeError(f"Cannot modify the attribute {attribute.name}")


@define(slots=True)
class BodyContainer(metaclass=SingletonMeta):
    bodies: list[Body] = field(default=[])
    _next_id: int      = field(default=0, repr=False)  # repr=False ensures _next_id is not shown in the representation

    def add_body(self, *args: Any, **kwargs: Any) -> None:
        new_body = Body(id=self._next_id, *args, **kwargs)
        self.bodies.append(new_body)
        self._next_id += 1

    def __getitem__(self, body_id: int) -> Body:
        for body in self.bodies:
            if body.id == body_id:
                return body
        raise IndexError(f"No body found with id {body_id}")

    def __iter__(self) -> Iterator[Body]:
        for body in self.bodies:
            yield body


@define(slots=True)
class InteractionContainer(metaclass=SingletonMeta):
    _interactions: Dict[Tuple[int, int], Interaction] = field(factory=dict)

    def add_interaction(self, *args: Any, **kwargs: Any) -> None:
        inter = Interaction(*args, **kwargs)
        key = (inter.body1.id, inter.body2.id)
        self._interactions[key] = inter

    def __getitem__(self, ids: Tuple[int, int]) -> Interaction:
        id1, id2 = ids
        if id1 > id2:
            id1, id2 = id2, id1  # Swap so that id1 is always the smaller ID
        return self._interactions[(id1, id2)]

    def __iter__(self) -> Iterator[Tuple[Tuple[int, int], Interaction]]:
        return iter(self._interactions.items())


@define(slots=True)
class Scene(metaclass=SingletonMeta):
    """The scene of the simulation"""
    bodies: BodyContainer            = field(factory=lambda: BodyContainer())
    # This ensures the Singleton instance of BodyContainer is used

    intactions: InteractionContainer = field(factory=lambda: InteractionContainer())
    # This ensures the Singleton instance of InteractionContainer is used


@define(slots=True)
class SerialEngine(ABC):
    """Engines or processes that run in series (one after another) in the simulation"""

    iterPeriod: int = field(default=1)
    scene: Scene = field(factory=lambda: Scene())  # This ensures the Singleton instance of Scene is used

    @abstractmethod
    def run(self) -> None:
        """What happens during a simulation"""
        pass


@define(slots=True)
class ForceResetter(SerialEngine, metaclass=CombinedMeta):
    def run(self) -> None:
        for body in self.scene.bodies.bodies:
            body.reset_forceTorque()


@define(slots=True)
class InteractionsCalculator(SerialEngine, metaclass=CombinedMeta):
    '''Calculates the interactions between the nodes, using the interaction objects we define'''

    # This makes sure the iterPeriod is not modified
    iterPeriod: int = field(default=1, repr=True, on_setattr=raise_on_modify)
    lnteractionList: InteractionContainer = field(factory=lambda: InteractionContainer())

    def __attrs_post_init__(self) -> None:
        for _, interaction in self.lnteractionList:
            interaction.reset_ForceTorque()


    def run(self) -> None:
        for _, interaction in self.lnteractionList:
            interaction.calc_ForcesTorques()


@define(slots=True)
class LeapFrogIntegrator(SerialEngine, metaclass=CombinedMeta):
    '''Integrator that uses the LeapFrog method'''
    dt: float       = field(default=1e-6)

    iterPeriod: int = field(default=1, repr=True, on_setattr=raise_on_modify)
    # This makes sure the iterPeriod is not modified

    bodyList: list[Body] = field(default=[])

    def __attrs_post_init__(self) -> None:
        self.bodyList = self.scene.bodies.bodies

    def run(self) -> None:
        for body in self.bodyList:
            mass: F64 = body.mass
            pos0: Vector3D = body.pos
            vel0: Vector3D = body.vel
            ori0: Quaternion  = body.ori
            angVel0: Vector3D = body.angVel

            vel1: Vector3D = vel0 + (self.dt * body.force / mass) * int(body.DynamicQ)
            pos1: Vector3D = pos0 + self.dt * vel1

            angVel1: Vector3D   = angVel0 + (self.dt * body.torque / body.inertia) * int(body.DynamicQ)
            angVelMag: F64 = norm(angVel1)
            angle: F64     = self.dt * angVelMag
            delta_q: Quaternion = Quaternion()

            if angle != 0.0:
                axis: Vector3D = angVel1 / angVelMag
                cos_x: F64     = np.cos(angle / 2)
                sin_x: F64     = np.sin(angle / 2)
                delta_q        = Quaternion(np.array([cos_x, *axis * sin_x]))

            ori1 = delta_q * ori0

            body.pos   = pos1
            body.vel   = vel1
            body.ori   = ori1
            body.angVel = angVel1


class CustomPythonEngine(SerialEngine):
    pyFunction: Callable[..., Any] = field(default=lambda: None)
    args: tuple[Any, ...]   = field(default=())
    kwargs: dict[str, Any]  = field(default={})

    def __init__(self, *args, pyFunction=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__()  # Call the parent constructor
        if pyFunction:
            self.pyFunction = pyFunction

    def run(self) -> None:
        self.pyFunction(*self.args, **self.kwargs)


@define(slots=True)
class SimulationLoop:
    engines: list[SerialEngine]

    def simulate(self, no_iteratons) -> None:
        for i in range(no_iteratons):
            for engine in self.engines:
                if i % engine.iterPeriod == 0:
                    engine.run()

# -*- encoding=utf-8 -*-

from builtins import zip
from builtins import range
from yade import pack, geom, qt
from yade.gridpfacet import *
from pylab import *
import json

def normalise(a):
    """Normalise a vector"""
    len = norm(a)
    if len == 0.0:
        raise ZeroDivisionError("Cannot normalise a zero vector")
    res = a / len
    return res

O.engines = [
        ForceResetter(),
        PyRunner(command='move()', iterPeriod = 1),
        InsertionSortCollider([Bo1_GridConnection_Aabb()]),
        InteractionLoop(
                [Ig2_GridNode_GridNode_GridNodeGeom6D()], [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(setCohesionNow=True, setCohesionOnNewContacts=False)],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment()]
        ),
        PyRunner(command='record()', iterPeriod=1),
        PyRunner(command='display()', iterPeriod=26),
        NewtonIntegrator(gravity=(0, 0, 0), damping=0.1, label='newton')
]

O.materials.append(
        CohFrictMat(young=70e9, poisson=0.35, density=2700, frictionAngle=radians(10), normalCohesion=1e7, shearCohesion=1e7, momentRotationLaw=True, label='mat')
)

### Parameters ###
r = 0.1

### Create all nodes first
nodeIds = []
oris = [[0,0,0], [1,0,0]]
for i in oris:
	nodeIds.append(O.bodies.append(gridNode(i, r, wire=False, fixed=False, material='mat')))

### Create connections between the nodes
connectionIds = []
for i, j in zip(nodeIds[:-1], nodeIds[1:]):
	connectionIds.append(O.bodies.append(gridConnection(i, j, r)))

### Set a fixed node
O.bodies[0].dynamic = False
O.bodies[1].dynamic = False
# O.bodies[1].state.vel = [1,0,0]

oris = []
twists = []
bends = []

def move():
#  |      __init__( (object)arg1, (float)w, (float)x, (float)y, (float)z) -> None :
#  |          Initialize from coefficients.
#  |          
#  |          .. note:: The order of coefficients is *w*, *x*, *y*, *z*. The [] operator numbers them differently, 0...4 for *x* *y* *z* *w*!
#  |      

        vel = 5e4
        it = O.iter
        time = it*O.dt
        ang  = vel*time
        axis = normalise(np.array([1, 1, 0]))
        ori = Quaternion(np.cos(ang/2), *axis * np.sin(ang/2))
        O.bodies[1].state.ori = ori

def record():
        global oris, twists
        ori = O.bodies[1].state.ori
        momentTwist = O.interactions[0,1].phys.moment_twist
        bendTwist   = O.interactions[0,1].phys.moment_bending
        oris.append([ori[3], ori[0], ori[1], ori[2]])
        twists.append(list(momentTwist))
        bends.append(list(bendTwist))
        print(O.iter)


def display():
        ans = {}
        ans['ori'] = oris
        ans['torsion_moment'] = twists
        ans['bending_moment'] = bends
        print("ori: ", oris)
        print("torsion_moment: ", twists)
        print("bends:", bends)
        file = "torsionBendingTest.json"
        with open(file, "w") as file:
                json.dump(ans, file, indent=4)
        O.pause()



O.dt = 1e-06
O.saveTmp()
qt.View()

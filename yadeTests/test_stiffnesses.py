# -*- encoding=utf-8 -*-

from builtins import zip
from builtins import range
from yade import pack, geom, qt
from yade.gridpfacet import *
from pylab import *

O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_GridConnection_Aabb()]),
        InteractionLoop(
                [Ig2_GridNode_GridNode_GridNodeGeom6D()], [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(setCohesionNow=True, setCohesionOnNewContacts=False)],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment()]
        ),
        NewtonIntegrator(gravity=(0, 0, -10), damping=0.1, label='newton')
]

O.materials.append(
        CohFrictMat(young=70e9, poisson=0.35, density=2700, frictionAngle=radians(10), normalCohesion=1e7, shearCohesion=1e7, momentRotationLaw=True, label='mat')
)

### Parameters ###
r = 0.1

### Create all nodes first
nodeIds = []
poses = [[0,0,0], [1,0,0], [3,0,0]]
for i in poses:
	nodeIds.append(O.bodies.append(gridNode(i, r, wire=False, fixed=False, material='mat', color=color)))

### Create connections between the nodes
connectionIds = []
for i, j in zip(nodeIds[:-1], nodeIds[1:]):
	connectionIds.append(O.bodies.append(gridConnection(i, j, r, color=color)))

### Set a fixed node
O.bodies[0].dynamic = False

O.dt = 1e-06
O.saveTmp()
qt.View()

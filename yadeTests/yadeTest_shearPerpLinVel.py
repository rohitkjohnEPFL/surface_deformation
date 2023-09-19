# -*- encoding=utf-8 -*-

from builtins import zip
from builtins import range
from yade import pack, geom, qt
from yade.gridpfacet import *
from pylab import *
import json

## Shear force is tested by moving the test particle perpendicular to the edge
## No rotation

O.engines = [
        ForceResetter(),
        # PyRunner(command='move()', iterPeriod = 1),
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
poses = [[0,0,0], [1,0,0]]
for i in poses:
	nodeIds.append(O.bodies.append(gridNode(i, r, wire=False, fixed=False, material='mat')))

### Create connections between the nodes
connectionIds = []
for i, j in zip(nodeIds[:-1], nodeIds[1:]):
	connectionIds.append(O.bodies.append(gridConnection(i, j, r)))

### Set a fixed node
O.bodies[0].dynamic = False
O.bodies[1].dynamic = False
# O.bodies[1].state.vel = [1,0,0]

poses = []
fs = []
vels = []
shearIncs = []

vel = 1.0
O.bodies[1].state.vel = [0,vel,0]

def move():
        vel = 1.0
        it = O.iter
        time = it*O.dt
        pos = 1.0 + vel*time
        # O.bodies[1].state.pos = [0,pos,0]
        O.bodies[1].state.vel = [0,vel,0]


def record():
        global poses, fs, vels, shearIncs
        pos = O.bodies[1].state.pos
        v   = O.bodies[1].state.vel
        force = O.interactions[0,1].phys.shearForce
        dus = O.interactions[0,1].geom.shearInc
        poses.append(list(pos))
        fs.append(list(force))
        vels.append(list(v))
        shearIncs.append(list(dus))
        print(O.iter)


def display():
        ans = {}
        ans['pos'] = poses
        ans['force'] = fs
        ans['vels'] = vels
        ans['shearInc'] = shearIncs
        print("pos: ", poses)
        print("Force: ", fs)
        print("Vel: ", vels)

        file = "shearPerpLinearVel.json"
        with open(file, "w") as file:
                json.dump(ans, file, indent=4)
        O.pause()



O.dt = 1e-06
O.saveTmp()
qt.View()

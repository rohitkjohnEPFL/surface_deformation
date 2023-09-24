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
        PyRunner(command='record_pos()', iterPeriod=1),
        InsertionSortCollider([Bo1_GridConnection_Aabb()]),
        InteractionLoop(
                [Ig2_GridNode_GridNode_GridNodeGeom6D()], [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(setCohesionNow=True, setCohesionOnNewContacts=False)],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment()]
        ),
        PyRunner(command='record_force()', iterPeriod=1),
        PyRunner(command='display()', iterPeriod=26),
        ForceResetter(),
        NewtonIntegrator(gravity=(0, 0, 0), damping=0.1, label='newton')
]

O.materials.append(
        CohFrictMat(young=70e9, poisson=0.35, density=2700, frictionAngle=radians(10), normalCohesion=1e7, shearCohesion=1e7, momentRotationLaw=True, label='mat')
)

### Parameters ###
r = 0.1

### Create all nodes first
nodeIds = []
pos_1 = [[0,0,0], [1,0,0]]
for i in pos_1:
	nodeIds.append(O.bodies.append(gridNode(i, r, wire=False, fixed=False, material='mat')))

### Create connections between the nodes
connectionIds = []
for i, j in zip(nodeIds[:-1], nodeIds[1:]):
	connectionIds.append(O.bodies.append(gridConnection(i, j, r)))

### Set a fixed node
# O.bodies[0].dynamic = False
# O.bodies[1].dynamic = False
# O.bodies[1].state.vel = [1,0,0]

poses_1  = []
fs_1   = []
ms_1   = []
vels_1 = []
oris_1  = []
ang_vels_1 = []

poses_2  = []
fs_2   = []
ms_2   = []
vels_2 = []
oris_2  = []
ang_vels_2 = []

O.bodies[0].state.vel = np.array([-5,6,9]) * 1e-3
O.bodies[1].state.vel = np.array([1,2,3]) * 1e-3
O.bodies[0].state.angVel = np.array([2,9,3]) * 1e1
O.bodies[1].state.angVel = np.array([7,6.5,11]) * 1e1

normal_force = []
shear_force  = []
bending_moment = []
torsion_moment = []
twists = []
bendings = []
shearIncs = []

def record_pos():
        global poses_1, fs_1, vels_1, oris_1, ang_vels_1, ms_1
        global poses_2, fs_2, vels_2, oris_2, ang_vels_2, ms_2

        global normal_force, shear_force, bending_moment, torsion_moment, twists, bendings, shearIncs


        pos_1 = O.bodies[0].state.pos
        v_1   = O.bodies[0].state.vel
        ori_1 = O.bodies[0].state.ori
        ang_vel1 = O.bodies[0].state.angVel

        pos_2 = O.bodies[1].state.pos
        v_2   = O.bodies[1].state.vel
        ori_2 = O.bodies[1].state.ori
        ang_vel2 = O.bodies[1].state.angVel

        poses_1.append(list(pos_1))
        vels_1.append(list(v_1))
        oris_1.append([ori_1[3], ori_1[0], ori_1[1], ori_1[2]])
        ang_vels_1.append(list(ang_vel1))

        poses_2.append(list(pos_2))
        vels_2.append(list(v_2))
        oris_2.append([ori_2[3], ori_2[0], ori_2[1], ori_2[2]])
        ang_vels_2.append(list(ang_vel2))

        norm_f = O.interactions[0,1].phys.normalForce
        shea_f = O.interactions[0,1].phys.shearForce
        tors_m = O.interactions[0,1].phys.moment_twist
        bend_m = O.interactions[0,1].phys.moment_bending

        twis = O.interactions[0,1].geom.twist
        bend = O.interactions[0,1].geom.bending

        shearIncrement = O.interactions[0,1].geom.shearInc

        normal_force.append(list(norm_f))
        shear_force.append(list(shea_f))
        bending_moment.append(list(tors_m))
        torsion_moment.append(list(bend_m))
        twists.append(twis)
        bendings.append(list(bend))
        shearIncs.append(list(shearIncrement))

 
def record_force():
        global fs_1, fs_2, ms_1, ms_2

        force_1 = O.forces.f(0)
        force_2 = O.forces.f(1)
        moment_1 = O.forces.m(0)
        moment_2 = O.forces.m(1)

        fs_1.append(list(force_1))
        fs_2.append(list(force_2))
        ms_1.append(list(moment_1))
        ms_2.append(list(moment_2))

        contactPoint = O.interactions[0, 1].geom.contactPoint
        pos1 = O.bodies[0].state.pos
        pos2 = O.bodies[1].state.pos

        norm_f = O.interactions[0,1].phys.normalForce
        shea_f = O.interactions[0,1].phys.shearForce
        tors_m = O.interactions[0,1].phys.moment_twist
        bend_m = O.interactions[0,1].phys.moment_bending

        


def display():
        ans = {}
        ans['pos1']    = poses_1
        ans['force1']  = fs_1
        ans['moment_1'] = ms_1
        ans['vel1']    = vels_1
        ans['ori1']    = oris_1
        ans['angVel1'] = ang_vels_1

        ans['pos2']    = poses_2
        ans['force2']  = fs_2
        ans['moment_2'] = ms_1

        ans['vel2']    = vels_2
        ans['ori2']    = oris_2
        ans['angVel2'] = ang_vels_2

        ans['normal_f']  = normal_force
        ans['shear_f']   = shear_force
        ans['bend_m']    = bending_moment
        ans['torsion_m'] = torsion_moment

        ans['twist']   = twists
        ans['bending'] = bendings
        ans['shearInc']= shearIncs

        file = "TotalForceTorque.json"
        with open(file, "w") as file:
                json.dump(ans, file, indent=4)
        O.pause()



O.dt = 1e-06
O.saveTmp()
qt.View()

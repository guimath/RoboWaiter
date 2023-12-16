import nimblephysics as nimble
import torch
import numpy as np
import time
import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep
LOAD_PATH = ABS_PATH + 'states_ld' + os.sep
BOTTLE_PATH = ABS_PATH + 'new_bottle' + os.sep
SERVE_PORT = 8000



def show_prep_cap_twist():
    urdf_file_path = ABS_PATH + 'ur5_gripper_ee_box.urdf'
    urdf_file_path = ABS_PATH + 'ur5_gripper_ee.urdf'
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])
    world.setTimeStep(0.01)

    arm = world.loadSkeleton(urdf_file_path)
    ground = world.loadSkeleton(ABS_PATH + 'ground_obj.urdf')
    urdfParser = nimble.utils.DartLoader()
    bottle = world.loadSkeleton(BOTTLE_PATH + 'bottle_fixed_org.urdf', [-0.5, 0.1, 0.5])

    capJoint, cap = bottle.createScrewJointAndBodyNodePair(bottle.getBodyNode("bottle_link"))
    cap.setName("cap")
    mesh = nimble.dynamics.MeshShape(np.array([0.015, 0.02, 0.015]), ABS_PATH + 'box.obj')
    cap_shape = cap.createShapeNode(mesh)
    cap_shape.createCollisionAspect()
    SHOW_TRUE_CAP = True
    if SHOW_TRUE_CAP :
        actual_cap_mesh = nimble.dynamics.MeshShape(np.array([0.001, -0.0023, 0.001]), BOTTLE_PATH + 'cap.obj')
        true_cap_shape = cap.createShapeNode(actual_cap_mesh)
        true_cap_shape.setRelativeTranslation([0,0.03,0])
        true_cap_shape.createVisualAspect()
    else :
        cap_shape.createVisualAspect()
    childOffset = nimble.math.Isometry3()
    childOffset.set_translation([0, 0.12, 0])
    capJoint.setTransformFromParentBodyNode(childOffset)
    capJoint.setAxis([0, 1, 0])
    cap.setCollidable(True)
    cap.setFrictionCoeff(10000.)
    cap.setMass(0.001)
    capJoint.setPitch(0.03)

    arm.getJoint(10).setDampingCoefficient(0, 1e-4)
    arm.getJoint(12).setDampingCoefficient(0, 1e-4)

    for link in  ["right_inner_finger", "left_inner_finger", "right_inner_knuckle", "left_inner_knuckle"]:
        arm.getBodyNode(link).setCollidable(True)
        arm.getBodyNode(link).setFrictionCoeff(10000.)

    world.removeDofFromActionSpace(8)
    print(world.checkCollision())

    gui = nimble.NimbleGUI(world)
    gui.serve(SERVE_PORT)
    gui.nativeAPI().renderWorld(world, "world")
    states = torch.tensor(np.load(LOAD_PATH + 'prep_twist_cap.npy'))
    gui.loopStates(states)
    gui.blockWhileServing()



def show_cap_twist():
    urdf_file_path = ABS_PATH + 'ur5_gripper_ee.urdf'
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])
    world.setTimeStep(0.005)
    ground = world.loadSkeleton(ABS_PATH + 'ground_obj.urdf')

    arm = world.loadSkeleton(urdf_file_path)
    arm.clampPositionsToLimits()
    for i in range(arm.getNumJoints()):
        _joint = arm.getJoint(i)
        _joint.setPositionLimitEnforced(True)

    urdfParser = nimble.utils.DartLoader()
    bottle = urdfParser.parseSkeleton(BOTTLE_PATH + 'bottle_fixed_high.urdf')
    capJoint, cap = bottle.createScrewJointAndBodyNodePair(bottle.getBodyNode('bottle_link'))
    cap.setName("cap")
    mesh = nimble.dynamics.MeshShape(np.array([0.015, 0.015, 0.015]), ABS_PATH + 'box.obj')
    cap_shape = cap.createShapeNode(mesh)
    cap_shape.createCollisionAspect()
    SHOW_TRUE_CAP = False
    if SHOW_TRUE_CAP :
        actual_cap_mesh = nimble.dynamics.MeshShape(np.array([0.001, -0.0023, 0.001]), BOTTLE_PATH + 'cap.obj')
        true_cap_shape = cap.createShapeNode(actual_cap_mesh)
        true_cap_shape.setRelativeTranslation([0,0.03,0])
        true_cap_shape.createVisualAspect()
    else :
        cap_shape.createVisualAspect()
    childOffset = nimble.math.Isometry3()
    childOffset.set_translation([0, 0.12, 0])
    capJoint.setTransformFromParentBodyNode(childOffset)
    capJoint.setAxis([0, 1, 0])
    cap.setCollidable(True)
    cap.setFrictionCoeff(10000.)
    cap.setMass(0.001)
    capJoint.setPitch(0.03)

    arm.getJoint(10).setDampingCoefficient(0, 1e-4)
    arm.getJoint(12).setDampingCoefficient(0, 1e-4)

    for link in  ["right_inner_finger", "left_inner_finger", "right_inner_knuckle", "left_inner_knuckle"]:
        arm.getBodyNode(link).setCollidable(True)
        arm.getBodyNode(link).setFrictionCoeff(10000.)

    world.addSkeleton(bottle)
    world.removeDofFromActionSpace(8)
    print(world.checkCollision())
    gui = nimble.NimbleGUI(world)
    gui.serve(SERVE_PORT)
    gui.nativeAPI().renderWorld(world, "world")
        
    state =  [-0.9403792479054327, -1.0017459761382705, 0.7038864097987512, -1.2729367565816876, -1.5707963250225934, 0.6304170788894639, 0.2, 0.2, 0]
    state = np.concatenate([state, np.zeros(9)])
    state[6]=state[7]=0.2
    world.setState(state)
    print(arm.getBodyNode('gripper_center_link').getWorldTransform().translation())
    state = torch.tensor(world.getState())
    states = []
    torque = -1e-3
    new_action = torch.concat([torch.zeros(6), torch.tensor([torque, torque])])
    for i in range(100):
        if world.checkCollision():
            break
        state = nimble.timestep(world, state, new_action)
        states.append(state)

    torque = -1e-2
    state[-4] = -1
    state[-1] = 1
    new_action = torch.concat([torch.zeros(5), torch.tensor([0, torque, torque])])
    for i in range(500):
        state = nimble.timestep(world, state, new_action)
        states.append(state)

    print(state[8])
    gui.loopStates(states[::2])
    np.save(LOAD_PATH + 'cap_twist.npy',state.detach().numpy())
    gui.blockWhileServing()


def show_bottle_grasp():
    urdf_file_path = ABS_PATH + 'ur5_gripper_ee.urdf'
    world = nimble.simulation.World()
    world.setGravity([0, -5, 0])
    world.setTimeStep(0.01)

    arm = world.loadSkeleton(urdf_file_path)
    ground = world.loadSkeleton(ABS_PATH + 'ground_obj.urdf')
    bottle = world.loadSkeleton(BOTTLE_PATH + 'bottle_fixed_org.urdf', [-0.5, 0.1, 0.51])
   
    world.removeDofFromActionSpace(8)

    gui = nimble.NimbleGUI(world)
    gui.serve(SERVE_PORT)
    states = torch.tensor(np.load(LOAD_PATH + "bottle_grasp.npy"))
    gui.loopStates(states)
    gui.blockWhileServing()

def show_move_and_pour_bottle():
    urdf_file_path = ABS_PATH + 'ur5_gripper_ee_bottle.urdf'
    world = nimble.simulation.World()
    world.setGravity([0, -5, 0])
    world.setTimeStep(0.01)

    arm = world.loadSkeleton(urdf_file_path)
    ground = world.loadSkeleton(ABS_PATH + 'ground_obj.urdf')
    bowl = world.loadSkeleton(BOTTLE_PATH + 'bowl.urdf', basePosition=[0.5, 0.04, 0.65])
    
    gui = nimble.NimbleGUI(world)
    gui.serve(SERVE_PORT)
    states = torch.tensor(np.load(LOAD_PATH + "move_and_pour_bottle.npy"))
    gui.loopStates(states)
    gui.blockWhileServing()

def show_many_moves(num):
    POS = [[0.4, -0.6], [-0.4, 0.5], [-0.6, -0.5], [0.5, 0.5], [0.4, 0.7]]
    urdf_file_path = ABS_PATH + 'ur5_gripper_ee.urdf'
    world = nimble.simulation.World()
    world.setGravity([0, -5, 0])
    world.setTimeStep(0.01)

    arm = world.loadSkeleton(urdf_file_path)
    ground = world.loadSkeleton(ABS_PATH + 'ground_obj.urdf')
    bottle = world.loadSkeleton(BOTTLE_PATH + "bottle_fixed_org.urdf", [POS[num][0], 0.1, POS[num][1]])

    world.removeDofFromActionSpace(8)
    gui = nimble.NimbleGUI(world)
    gui.serve(SERVE_PORT)
    states = torch.tensor(np.load(LOAD_PATH + f'move_to_bottle_{num}.npy'))
    gui.loopStates(states)
    gui.blockWhileServing()


if __name__ ==  '__main__':
    # functions will create the appropriate world and play the action in the sim
    
    # show_prep_cap_twist()
    # show_cap_twist()
    show_bottle_grasp()
    # show_move_and_pour_bottle()
    # show_many_moves(2)

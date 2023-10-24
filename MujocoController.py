import mujoco_py as mp
#from pathlib import Path
import numpy as np
import ikpy.chain
from simple_pid import PID
import time

import os
import logging
logger = logging.getLogger('ROBW')
logging.basicConfig(format= '%(levelname)10s: %(message)s')#, level=logging.DEBUG)
logger.setLevel(logging.DEBUG)
URDF_FILE = 'assets/ur5_gripper.urdf'
XML_FILE = 'assets/UR5gripper_2_finger_cylinder.xml'

class MujocoController(object):
    """
    Class for control of an robotic arm in MuJoCo.
    It can be used on its own, in which case a new model, simulation and viewer will be created.
    It can also be passed these objects when creating an instance, in which case the class can be used
    to perform tasks on an already instantiated simulation.
    """

    def __init__(self, path_xml=XML_FILE, path_urdf=URDF_FILE, model=None, simulation=None, viewer=None):
        if model is None:
            self.model = mp.load_model_from_path(path_xml)
        else:
            self.model = model
        self.sim = mp.MjSim(self.model) if simulation is None else simulation
        self.viewer = mp.MjViewer(self.sim) if viewer is None else viewer

        self.ee_chain = ikpy.chain.Chain.from_urdf_file(path_urdf, active_links_mask=[False]+ [True]*6+[False])
        self.create_lists()
        self.groups = {
            'all': list(range(7)),
            'arm': list(range(6)),
            'gripper' : [6]
        }
        self.joint_nb = len(self.controller_list)
        self.current_target_joint_values = [
            self.controller_list[i].setpoint for i in range(self.joint_nb)
        ]
        
    def create_lists(self):
        ''' Creates PID controller list (one per joint)'''
        sample_time = 0.0001
        # p_scale = 1
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1
        self.controller_list = [
            PID( # 0 = Shoulder Pan Joint
                7 * p_scale,
                0.0 * i_scale,
                1.1 * d_scale,
                setpoint=0,
                output_limits=(-2, 2),
                sample_time=sample_time,
            ),
            PID( # 1 = Shoulder Lift Joint
                10 * p_scale,
                0.0 * i_scale,
                1.0 * d_scale,
                setpoint=-1.57,
                output_limits=(-2, 2),
                sample_time=sample_time,
            ),
            PID(  # 2 = Elbow Joint
                5 * p_scale,
                0.0 * i_scale,
                0.5 * d_scale,
                setpoint=1.57,
                output_limits=(-2, 2),
                sample_time=sample_time,
            ),
            PID( # 3 = Wrist 1 Joint
                7 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(-1, 1),
                sample_time=sample_time,
            ),
            PID( # 4 = Wrist 2 Joint
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(-1, 1),
                sample_time=sample_time,
            ),
            PID( # 5 = Wrist 3 Joint
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            ),
            PID( # 6 = Gripper Joint
                0.1 * p_scale,
                i_gripper,
                0.00 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        ]

    def move_group_to_joint_target(
        self,
        group="all",
        target=None,
        tolerance=0.05,
        max_steps=1000):
        ''' Updates joints targets and checks targets are reached'''
        ids = self.groups[group]
        steps = 1
        frames = []

        if target is not None:
          for i, id in enumerate(ids):
            # if id == 5 : 
            #     self.current_target_joint_values[id] = 0
            #     continue
            self.current_target_joint_values[id] = target[i]

        for j in range(self.joint_nb):
            self.controller_list[j].setpoint = self.current_target_joint_values[j]

        while steps<max_steps:
            for i in range(self.joint_nb):
                self.sim.data.ctrl[i] = self.controller_list[i](self.sim.data.qpos[i])

            self.sim.step()
            self.viewer.render()

            deltas = [self.current_target_joint_values[j] - self.sim.data.qpos[j] for j in range(self.joint_nb)]
            deltas = [deltas[id] for id in ids] # only checking moving group
            #if (steps%100) == 0 : logger.debug(f'{steps} deltas={[f"{delta:.3f}" for delta in deltas]}')
            if np.max(np.abs(deltas))<tolerance:
                break
            steps += 1
        return frames

    def open_gripper(self, half=False):
        ''' Opens the gripper while keeping the arm in a steady position.'''
        # print('Open: ', self.sim.data.qpos[self.actuated_joint_ids][self.groups['Gripper']])
        if half : return self.move_group_to_joint_target(
            group="gripper", target=[0.0], max_steps=1000, tolerance=0.02)

        return self.move_group_to_joint_target(
            group="gripper", target=[0.4], max_steps=1000, tolerance=0.02)

    def close_gripper(self):
        ''' Closes the gripper while keeping the arm in a steady position.'''
        return self.move_group_to_joint_target(group="gripper", target=[-0.4], tolerance=0.03, max_steps=300)

    def move_ee_to_xyz(self, pos, orientation=[0, 0, -1], axis='X'):
        ''' Moves end effector to wanted pose'''
        joint_angles = self.ik(pos, orientation, axis)

        if joint_angles is None: return None
        return self.move_group_to_joint_target(group="arm", target=joint_angles)

    def ik(self, pos, orientation=[0, 0, -1], axis='X'):
        '''inverse kinematics to get get joint value for a given pose'''
        
        # Orientation note : 
        # - axis is the axis we want to align
        # - orientation is the vec we want to align it to

        # inverse kinematics solver chain starts at the base link so needs to offset by it
        # also specifies position of the ee joint, might want to add an offset so that the 
        # target position is the grasp center.
        correct = self.sim.data.body_xpos[self.model.body_name2id("base_link")] #- (rot@[0, -0.005, 0.16])
        gripper_center_position = pos - correct
        try :
            # trying to solve from actual position
            initial_position= [0] + [self.sim.data.qpos[i] for i in range(self.joint_nb)] 
            joint_angles = self.ee_chain.inverse_kinematics(
                target_position=gripper_center_position,
                target_orientation=orientation,
                orientation_mode=axis,
                initial_position=initial_position
            )
        except :
            # solving without current position
            joint_angles = self.ee_chain.inverse_kinematics(
                target_position=gripper_center_position,
                target_orientation=orientation,
                orientation_mode=axis,
                initial_position=[-1.0] + [-1.0] * 7
            )
        
        # Verifying accuracy of ik 
        joint_angles[6] = 0
        trans_mat = self.ee_chain.forward_kinematics(joint_angles)

        prediction = (
            trans_mat[:3, 3] + correct# self.sim.data.body_xpos[self.model.body_name2id("base_link")] - [0, -0.005, 0.16]
        )
        diff = abs(prediction - pos)
        error = np.sqrt(diff.dot(diff))
        if orientation== [0,0,-1]:
            joint_angles[6] = (np.arctan2(trans_mat[1,1], trans_mat[0,1])+ np.pi)%(2*np.pi)-np.pi#rot[3]
        else : 
            joint_angles[6] = (np.arctan2(trans_mat[0,1], trans_mat[1,1])+ np.pi)%(2*np.pi)-np.pi#rot[3]

        #print(self.ee_chain.forward_kinematics(joint_angles))
        joint_angles = joint_angles[1:-1]
        # x = qx / sqrt(1-qw*qw)x = qx / sqrt(1-qw*qw)
        #change = 

        if error > 0.02:
            logger.debug(f'x={prediction[0]:+.3f} y={prediction[1]:+.3f} z={prediction[2]:+.3f}')
        return joint_angles

    def stay(self, duration):
        ''' Waiting equivalent'''
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.move_group_to_joint_target(
                max_steps=10, tolerance=0.0000001
            )
            elapsed = (time.time() - starting_time) * 1000

    def xpos_print(self):
        for i in range(controller.model.nbody):
            print(f'{controller.model.body_id2name(i):30s} {controller.sim.data.body_xpos[i]}')
    def diff_xpos_print(self):
        try : 
            for i in range(controller.model.nbody):
                print(f'{controller.model.body_id2name(i):30s} {self.prev_xpos[i]-controller.sim.data.body_xpos[i]}')
        except : pass
        self.prev_xpos = controller.sim.data.body_xpos.copy()
    
    def get_xpos_name(self, name:str):
        return self.sim.data.body_xpos[self.model.body_name2id(name)]

    def pour_all(self): 
        target = self.current_target_joint_values.copy()
        target[0] = (target[0]+np.pi/2 + np.pi)%(2*np.pi)- np.pi
        self.move_group_to_joint_target('arm', target)
        target[5] = (target[5]-np.pi/2 + np.pi)%(2*np.pi)- np.pi
        self.move_group_to_joint_target('arm', target)


    def prep_twist(self):
        target = self.current_target_joint_values.copy()
        target[5] = - np.pi
        self.move_group_to_joint_target('arm', target)

    def untwist(self):
        target = self.current_target_joint_values.copy()
        target[5] = -np.pi
        self.move_group_to_joint_target('arm', target, max_steps=2000)
        self.close_gripper()
        for i in range(5):
            target[5] = np.pi
            self.move_group_to_joint_target('arm', target, max_steps=2000)
            self.open_gripper(half=True)
            target[5] = -np.pi
            self.move_group_to_joint_target('arm', target, max_steps=2000)
            self.close_gripper()
        target[0] = (target[0]+np.pi/2 + np.pi)%(2*np.pi)- np.pi
        self.move_group_to_joint_target('arm', target)
        self.open_gripper(half=True)

def test_all_joints(controller):
    target = controller.current_target_joint_values.copy()
    print(target)
    controller.close_gripper()
    controller.stay(1000)
    for i in range(7):
        logger.debug(i)
        controller.current_target_joint_values[i] = target[i]+ 0.3
        controller.controller_list[i].setpoint = 0
        controller.stay(1000)
        controller.controller_list[i].setpoint = target[i]
        controller.current_target_joint_values[i] = target[i]
        controller.stay(1000)
    logger.info('Test all joints complete')

def test_orientation(controller):
    target_xyz = np.array([0.0, 0.4, 1.5])
    diff = np.zeros((10,3))
    ee_pos = np.zeros((10,3))
    ee_pos[0] = target_xyz
    for i, orientation in enumerate([[1,0,0], [1,1,0], [0,1,0], [-1, 1, 0], [-1,0,0], [-1, -1, 0], [0, -1, 0], [1, -1, 0], [1,0,0]]):
        logger.debug(i+1)
        #orientation = [orientation[0], orientation[2], orientation[1]]
        controller.move_ee_to_xyz(target_xyz, orientation=orientation)
        ee_pos[i+1] = controller.get_xpos_name('ee_link')#(controller.get_xpos_name("right_inner_knuckle") + controller.get_xpos_name("left_inner_knuckle"))/2
        diff[i+1] = ee_pos[i]-ee_pos[i+1]
        controller.stay(100)   
    logger.debug(ee_pos)
    import matplotlib.pyplot as plt   
    plt.scatter(ee_pos[:,0], ee_pos[:,1])
    for i in range(10): 
        plt.annotate(str(i), (ee_pos[i][0], ee_pos[i][1])) 
    plt.show()
    logger.debug(diff)
    logger.info('test_orientation completed')

def grab_bottle(controller:MujocoController):
    controller.open_gripper()
    HOR_ORIENT = [-1.0, 0.0, 0.0]
    xyz = np.array([0.2, -0.5, 1.2])
    controller.move_ee_to_xyz(xyz, HOR_ORIENT) # just above
    controller.stay(1000)
    xyz[2] = 1.0
    controller.move_ee_to_xyz(xyz, HOR_ORIENT) # next to bottle
    controller.stay(1000)
    controller.close_gripper() # grabbing
    controller.stay(1000)
    xyz[2] = 1.2
    controller.move_ee_to_xyz(xyz, HOR_ORIENT) # lifting
    controller.stay(1000)
    controller.pour_all()
    
def twist_cap(controller:MujocoController):
    controller.open_gripper()
    VER_ORIENT = [0, 0.0, -1]
    xyz = np.array([0.05, -0.5, 1.4])
    controller.move_ee_to_xyz(xyz, VER_ORIENT) # just above
    controller.stay(1000)
    xyz[2] = 1.35
    controller.move_ee_to_xyz(xyz, VER_ORIENT) # next to bottle
    controller.stay(1000)
    controller.untwist()

if __name__ == '__main__':
    controller = MujocoController()
    twist_cap(controller)
    while True:
        controller.stay(1000)


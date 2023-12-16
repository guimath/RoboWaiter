import nimblephysics as nimble
import torch
import os
import numpy as np

ABS_PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep
ASSET_PATH = ABS_PATH + 'nimble' + os.sep

UR5_URDF    = ASSET_PATH + 'ur5_gripper_no_col.urdf'
# UR5_URDF    = ASSET_PATH + 'ur5_gripper_no_col copy.urdf'
GROUND_URDF = ASSET_PATH + 'ground_obj.urdf'
BOTTLE_URDF = ASSET_PATH + 'bottle.urdf'
BOTTLE1 = ASSET_PATH + 'new_bottle/bottle_and_cap.urdf'
BOX1 = ASSET_PATH + 'box1.urdf'
TIME_STEP = 0.01
SERVE_PORT = 8000
EPOCH_MAX = 600
LOSS_MIN = 1e-2
class NimbleController():
    states : []
    '''All the states to be displayed'''
    action : []
    '''Current torques applied to each joint'''

    def __init__(self, arm_urdf =UR5_URDF, env_urdf= [GROUND_URDF, BOTTLE_URDF]):

        self.world = nimble.simulation.World()
        self.world.setGravity([0, 0, 0]) # TODO change to real value 
        self.world.setTimeStep(TIME_STEP)
        self.arm = self.world.loadSkeleton(arm_urdf)
        # self.arm.setSelfCollisionCheck(True) 
        # lik = arm.getJoint(9)
        # rik = arm.getJoint(11)
        # print(lik.getName())
        # print(rik.getPositionUpperLimit(0))
        # print(rik.getName())
        # lik.setPositionUpperLimit(0,0.8) # only change idx zero because 1 DOF
        # rik.setPositionUpperLimit(0,0.8)
        # print(rik.getPositionUpperLimit(0))
        self.dofs = self.world.getNumDofs()
        for env in env_urdf:
            if env != GROUND_URDF :
                self.bottle_skel = self.world.loadSkeleton(env)
                for i in range(self.bottle_skel.getNumBodyNodes()):
                    node = self.bottle_skel.getBodyNode(i)
                    node.setFrictionCoeff(1000.)
                # print(test.getPositions())
                self.bottle = env
                # test.setPositions([0,0, 0, -0.5, 0.1, 0.035]) # 0.1, 0.035
            else : _ = self.world.loadSkeleton(env)
        self.remove_dof_from_actions()
        

        print(f"Number of DOFs: {self.world.getNumDofs()}")
        for i in [9, 11]:
            _joint = self.arm.getJoint(i)
            print(_joint.getName())
            _joint.setPositionLimitEnforced(True)
        
        self.gui = nimble.NimbleGUI(self.world)
        self.gui.serve(SERVE_PORT)
        self.gui.nativeAPI().renderWorld(self.world, "world")

        self.ikMap = nimble.neural.IKMapping(self.world)
        self.ikMap.addSpatialBodyNode(self.arm.getBodyNode("robotiq_base_link"))
        self.states = []
        self.action = torch.tensor(self.world.getAction())


    def remove_dof_from_actions(self):
        end_dofs = self.world.getNumDofs()
        # removing dofs from objects not arm
        for i in range(self.dofs,end_dofs):
            self.world.removeDofFromActionSpace(i) 

    def move_to_position(self, goal_pos, move_num_steps = 100, loss_threshold= LOSS_MIN):
        action_size = 8#self.world.getNumDofs()
        state_size  = self.world.getState().shape[-1]

        actions_robot = torch.zeros((move_num_steps, action_size), requires_grad=True)
        optimizer = torch.optim.Adam([{'params': actions_robot}], lr=0.3)
        #actions_other = torch.zeros((num_timesteps, self.world.getNumDofs() - 7), requires_grad=False)

        init_state = torch.tensor(self.world.getState()) # torch.zeros_like(
        print("Initial state:", init_state)
        init_hand_pos = nimble.map_to_pos(self.world, self.ikMap, init_state)
        target_pos = torch.Tensor(goal_pos)
        print("Target Position:", target_pos)

        for k in range(EPOCH_MAX):
            state = init_state
            states = [state.detach()]
            hand_poses = [nimble.map_to_pos(self.world, self.ikMap, state).detach()]
            for i in range(move_num_steps):
                action = actions_robot[i]
                state = nimble.timestep(self.world, state, action)
                state[6:8] = 0.009
                state[state_size//2+6:state_size//2+8] = 0
                #print(state)
                states.append(state.detach())
                hand_poses.append(nimble.map_to_pos(self.world, self.ikMap, state).detach())

            hand_pos = nimble.map_to_pos(self.world, self.ikMap, state)
            loss_grasp = ((hand_pos[3:] - target_pos)**2 * torch.tensor([10.0, 10.0, 1.0])).mean() + 10 * ((hand_pos[:3] - torch.Tensor([np.pi/2,0.0,0.0]))**2).mean()
            loss_stop = (state[state_size//2:state_size//2+action_size]**2).mean()
            loss = loss_grasp + 0.1*loss_stop

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if k%10==0 or loss < loss_threshold:
                pos = [f'{num:.2f}' for num in hand_pos.detach().numpy()[3:]]
                ori = [f'{num:.2f}' for num in hand_pos.detach().numpy()[:3]]
                print(f'\n ------------------------------------')
                print(f'Loss for epoch {k}: {loss.detach().numpy()}')
                print(f'Position: {pos}, Hand orientation: {ori}')
                print(f"Final joint speeds: {[f'{num**2:.2f}' for num in state[8:]]}")
            if loss < loss_threshold:
                return actions_robot, states

        print(f'time out ({loss=})')
        return actions_robot, states

    def move_ee_to_xyz(self, pos, orientation=None, loss_threshold= LOSS_MIN, move_num_steps=100):
        self.gui.nativeAPI().createSphere(f'goal_{len(self.states)}', pos=np.array(pos), radii=np.array([0.01, 0.01, 0.01]))
        self.gui.nativeAPI().renderWorld(self.world, "world")
        _actions, states = self.move_to_position(pos, move_num_steps, loss_threshold)
        self.states += states
        self.gui.loopStates(self.states)

    def block_while_serving(self):
        self.gui.blockWhileServing()

    def check_config(self, state):
        num = len(state)
        tar = self.world.getState()
        for i in range(num):
            tar[i] = state[i]
        self.world.setState(tar)
        self.gui.nativeAPI().renderWorld(self.world, "world")

    def show_states(self, show=False):
        if show : 
            print(f'showing {len(self.states)} frames')
            self.gui.loopStates(self.states)

    def close_gripper(self, torque=-1e-4, show=False):
        # self.action[6] = torque/10
        # self.run_for(20, reset=False)
        self.action[6:8] = torque
        prev = 0
        i = 0
        old_state = self.world.getState()
        while np.abs(old_state[8+6]) > 2e-6 or old_state[6]>0.1: #  or np.abs(old_state[6]-old_state[7])>1e-4
            prev = old_state[6]
            self.time_step(torch.tensor(old_state))
            old_state= self.world.getState()
            print(f'{i: <3} p0={old_state[6]: <23} p1={old_state[7]: <23} delta={old_state[8+6]}', end='\n')
            i+= 1
            if i == 300 : break
        # if np.abs(old_state[6]-prev) < 1e-6 and np.abs(old_state[6]-old_state[7])>1e-4 : 
        #     self.close_gripper(torque=torque*2, show=False)
        self.action[6] = torque
        
        self.show_states(show)

    def open_gripper(self, torque=3e-4, show=False):
        self.action[6:8] = torque
        old_state = self.world.getState()
        while old_state[6]<0.799 :
            self.time_step(torch.tensor(old_state))
            old_state= self.world.getState()
            print(f'p0={old_state[6]: <23} p1={old_state[7]: <23}', end='\n')
        self.states = self.states[-100:]
        self.show_states(show)
     

    def run_for(self, nstep, reset=True, show=False):
        if reset : self.states = []
        print(f'Running for {nstep} steps')
        print(self.action)
        for i in range(nstep):
            self.time_step()
            print(f'{i}', end= '\r')
        print(f'{nstep} steps done')
        # self.states = self.states[-500:]
        
        self.show_states(show)

    def time_step(self, state = None):
        state = torch.tensor(self.world.getState()) if state is None else state
        out = nimble.timestep(self.world, state, self.action)
        out[self.dofs+1] = out[self.dofs]
        self.arm.clampPositionsToLimits()
        self.states.append(out)
        return out
    

def main():
    controller = NimbleController(env_urdf=[GROUND_URDF])
    gripper_val = 0.7
    state = [-1.2480710619904059, -0.9789437212492733, 1.5658405516053635, -0.5868968316491749, -2.8188673909386233, -3.9482173086469174e-09, gripper_val, gripper_val]
    controller.check_config(state)
    controller.close_gripper(torque=-1e-4, show=False)
    controller.action[1] = -1
    controller.run_for(100, reset=False, show=True)
    # controller.open_gripper(show=True)
    controller.block_while_serving()

if __name__ == '__main__':
    main()
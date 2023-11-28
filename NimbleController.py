import nimblephysics as nimble
import torch
import os
import numpy as np
import time

ABS_PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep

UR5_URDF    = ABS_PATH + 'nimble' + os.sep + 'ur5_gripper_reduced.urdf'
GROUND_URDF = ABS_PATH + 'nimble' + os.sep + 'ground.urdf'
BOTTLE_URDF = ABS_PATH + 'nimble' + os.sep + 'bottle.urdf'
TIMESTEP = 0.01
SERVE_PORT = 8060
EPOCH_MAX = 600
LOSS_MIN = 1e-2
class NimbleController():
    def __init__(self, arm_urdf =UR5_URDF, env_urdf= [GROUND_URDF, BOTTLE_URDF]):

        self.world = nimble.simulation.World()
        self.world.setGravity([0, 0, 0]) # TODO change to real value 
        self.world.setTimeStep(TIMESTEP)

        arm = self.world.loadSkeleton(arm_urdf)
        arm.setSelfCollisionCheck(True) 
        dofs = self.world.getNumDofs()
        for env in env_urdf:
            _ = self.world.loadSkeleton(env)
        end_dofs = self.world.getNumDofs()

        # removing dofs from objects not arm
        for i in range(dofs,end_dofs):
            self.world.removeDofFromActionSpace(i) 

        print(f"Number of DOFs: {self.world.getNumDofs()}")

        self.gui = nimble.NimbleGUI(self.world)
        self.gui.serve(SERVE_PORT)
        self.gui.nativeAPI().renderWorld(self.world, "world")

        self.ikMap = nimble.neural.IKMapping(self.world)
        self.ikMap.addSpatialBodyNode(arm.getBodyNode("robotiq_base_link"))
        self.states = []

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

    def close_gripper(self, torque=-3e-2, show=False):
        action = self.world.getAction()
        action[6:8]= torque
        action = torch.tensor(action) # torch.zeros_like(
        self.run_for(
            nstep  = 40, 
            action = action, 
            reset  = False, 
            show   = show
        )

    def open_gripper(self, show=False):
        action = self.world.getAction()
        action = torch.tensor(action) # torch.zeros_like(
        action[6:8] = 2e-2
        old_state = self.world.getState()
        print('opening gripper')
        while old_state[6]<0.8 :
            state = nimble.timestep(self.world, torch.tensor(old_state), action)
            self.states.append(state.detach())
            old_state= self.world.getState()
            print(f'p0={old_state[6]: <23} p1={old_state[7]: <23}', end='\r')

        print(f'Done opening {"": <80}')
        action[6:8]= 0
        state = nimble.timestep(self.world, torch.tensor(old_state), action)
        self.states.append(state.detach())
        if show : self.gui.loopStates(self.states)

    # def open_gripper_pid(self):
    #     action = self.world.getAction()
    #     action = torch.tensor(action) # torch.zeros_like(
    #     TARGET = 0.9
    #     k_pos = 5e-2
    #     k_vel = 0e-1
    #     print(f'opening gripper')
    #     # old_state= self.world.getState()
    #     # action[6] = 5e-4
    #     # action[7] = 5e-4
    #     # while old_state[6] < 0 :
    #     #     state = nimble.timestep(self.world, torch.tensor(old_state), action)
    #     #     self.states.append(state.detach())
    #     #     old_state= self.world.getState()
    #     #     print(f'{old_state[6]} {old_state[7]}')
    #     # action[6] = 0
    #     # action[7] = 5e-4
    #     # while self.world.getState()[7] < 0 :
    #     #     action[7] = 3e-5
    #     #     state = nimble.timestep(self.world, torch.tensor(self.world.getState()), action)
    #     #     self.states.append(state.detach())
    #     #     print(self.world.getState()[7])
    #     torque = [0,0]
    #     while True :
    #         old_state= self.world.getState()
    #         pos = old_state[6:8]
    #         vel = old_state[8+6:8+8]
    #         for i in range(2):
    #             torque[i] = np.min([1e20 ,(TARGET-pos[i])*k_pos + vel[i]*k_vel]) # max torque 

    #         action[6]=torque[0]
    #         action[7]=torque[1]
    #         print(f't={torque[0]: <23} pos0={pos[0]: <20} pos1={pos[1]: <20}', end='\n')
    #         if np.abs(TARGET-pos[0]) < 1e-4 and np.abs(TARGET-pos[1]) < 1e-4 and \
    #             np.abs(vel[0]) < 6e-5  and np.abs(vel[1]) < 6e-5: break
    #         # if pos1 > TARGET : break
    #         state = nimble.timestep(self.world, torch.tensor(old_state), action)
    #         self.states.append(state.detach())

    #     action[6:8]= 0
    #     init_state = torch.tensor(self.world.getState())
    #     state = nimble.timestep(self.world, init_state, action)
    #     self.states.append(state.detach())
    #     print(f'opening done')
    #     self.gui.loopStates(self.states)

    def run_for(self, nstep, action=None, reset=True, show=False):
        if reset : self.states = []
        if action is None : action = torch.tensor(self.world.getAction())
        print(f'Running for {nstep} steps')
        for i in range(nstep):
            init_state = torch.tensor(self.world.getState())
            state = nimble.timestep(self.world, init_state, action)
            self.states.append(state.detach())
            print(f'{i}', end= '\r')
        print(f'{nstep} steps done')
        if show : self.gui.loopStates(self.states)

def main():
    controller = NimbleController()
    state = [-1.2480710611174737, -1.1466361548852488, 1.279203826335021, -0.13256767288089324, -2.8188673866633884, -5.459144247765835e-10, 0.4]
    state = [-1.2480710619904059, -0.9789437212492733, 1.5658405516053635, -0.5868968316491749, -2.8188673909386233, -3.9482173086469174e-09, 0.4]
    gripper_val = 0.4 #0.9
    state[6] = gripper_val
    state += [gripper_val]
    controller.check_config(state)
    controller.close_gripper()
    controller.open_gripper()
    controller.run_for(50, reset=False, show=True)
    # goal_pos = [0.6, 0.2, -0.4]
    # controller.move_ee_to_xyz(goal_pos, loss_threshold=1e-2)
    # goal_pos = [0.4, 0.6, -0.3]
    # controller.move_ee_to_xyz(goal_pos, loss_threshold=1e-3)
    controller.block_while_serving()

if __name__ == '__main__':
    main()
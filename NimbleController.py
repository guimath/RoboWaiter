import nimblephysics as nimble
import torch
import os
import numpy as np
import time

ABS_PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep

UR5_URDF    = ABS_PATH + 'nimble' + os.sep + 'ur5_gripper_reduced.urdf'
GROUND_URDF = ABS_PATH + 'nimble' + os.sep + 'ground.urdf'
BOTTLE_URDF = ABS_PATH + 'nimble' + os.sep + 'bottle.urdf'
SERVE_PORT = 8060
EPOCH_MAX = 600
LOSS_MIN = 1e-2
class NimbleController():
    def __init__(self, arm_urdf =UR5_URDF, env_urdf= []):

        self.world = nimble.simulation.World()
        self.world.setGravity([0, -1, 0]) # TODO change to real value ? 
        self.world.setTimeStep(0.01)

        arm = self.world.loadSkeleton(arm_urdf)
        # arm.setSelfCollisionCheck(True) 
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

def main():
    controller = NimbleController()
    goal_pos = [0.6, 0.2, -0.4]
    controller.move_ee_to_xyz(goal_pos, loss_threshold=1e-3)
    # goal_pos = [0.4, 0.6, -0.3]
    # controller.move_ee_to_xyz(goal_pos, loss_threshold=1e-3)
    controller.block_while_serving()

if __name__ == '__main__':
    main()
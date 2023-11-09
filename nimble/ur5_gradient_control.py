import nimblephysics as nimble
import torch
import os
import numpy as np
import time
# from simple_pid import PID

print(os.getcwd())

# Load the URDF file
urdf_file_path = './ur5_gripper_reduced.urdf'
world = nimble.simulation.World()
world.setGravity([0, -9.81, 0])
world.setTimeStep(0.01)
arm = world.loadSkeleton(urdf_file_path)
ground = world.loadSkeleton("./ground.urdf")
bottle = world.loadSkeleton("./bottle.urdf")

for i in range(8,14):
    world.removeDofFromActionSpace(i) #End gripper should have only one degree of freedom

print(f"Number of DOFs: {world.getNumDofs()}")

gui = nimble.NimbleGUI(world)
gui.serve(8060)
gui.nativeAPI().renderWorld(world, "world")


ikMap = nimble.neural.IKMapping(world)
ikMap.addSpatialBodyNode(arm.getBodyNode("robotiq_base_link"))



def move_to_position(goal_pos, move_num_steps = 100):
    global full_states

    action_size = 8#world.getNumDofs()
    state_size  = world.getState().shape[-1]
    print(action_size)
    actions_robot = torch.zeros((move_num_steps, action_size), requires_grad=True)
    optimizer = torch.optim.Adam([{'params': actions_robot}], lr=0.3)
	#actions_other = torch.zeros((num_timesteps, world.getNumDofs() - 7), requires_grad=False)

    init_state = torch.tensor(world.getState()) # torch.zeros_like(
    print("Initial state:", init_state)
    init_hand_pos = nimble.map_to_pos(world, ikMap, init_state)
    target_pos = torch.Tensor(goal_pos)
    print("Target Position:", target_pos)

    for k in range(3000):
        state = init_state
        states = [state.detach()]
        hand_poses = [nimble.map_to_pos(world, ikMap, state).detach()]
        for i in range(move_num_steps):
            action = actions_robot[i]
            state = nimble.timestep(world, state, action)
            state[6:8] = 0.009
            state[state_size//2+6:state_size//2+8] = 0
            #print(state)
            states.append(state.detach())
            hand_poses.append(nimble.map_to_pos(world, ikMap, state).detach())

        full_states = states
        hand_pos = nimble.map_to_pos(world, ikMap, state)


        loss_grasp = ((hand_pos[3:] - target_pos)**2 * torch.tensor([10.0, 10.0, 1.0])).mean() + 10 * ((hand_pos[:3] - torch.Tensor([np.pi/2,0.0,0.0]))**2).mean()
        loss_stop = (state[state_size//2:state_size//2+action_size]**2).mean()
        loss = loss_grasp + 0.1*loss_stop




        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if k%10==0:
            print(f"Loss for epoch {k}: {loss.detach().numpy()} \n")
            print(f"Position: {[f'{num:.2f}' for num in hand_pos.detach().numpy()[3:]]}, \
            Hand orientation: {[f'{num:.2f}' for num in hand_pos.detach().numpy()[:3]]},) \
            Final joint speeds: {[f'{num**2:.2f}' for num in state[8:]]}")
        # input()
        if loss < 1e-2:
            flag = True
            break

    return actions_robot, states


goal_pos = [-0.6, 0.2, 0.4]
gui.nativeAPI().createSphere("goal_pos", pos=np.array(goal_pos), radii=np.array([0.01, 0.01, 0.01]))
gui.nativeAPI().renderWorld(world, "world")
actions, states = move_to_position(goal_pos)
gui.loopStates(states)
world.setState(states[-1])
print(states)

# goal_pos = [0.6, 0.2, -0.4]
# gui.nativeAPI().createSphere("goal_pos2", pos=np.array(goal_pos), radii=np.array([0.01, 0.01, 0.01]))
# gui.nativeAPI().renderWorld(world, "world")
# # time.sleep(10)
# actions2, states2 = move_to_position(goal_pos)
# gui.loopStates(states2)

# print(states[-1])
# world.setState(states[-1])
# gui.loopStates(states)
# end_state = [-0.1577, -1.0746,  2.9038, -0.2648, -4.7167, -3.3100,  0.0090,  0.0090, -0.0629,  0.4103,  0.3762,  0.3480,  0.0478,  0.0295,  0.0000,  0.0000]
# world.setState(end_state)
# gui.nativeAPI().renderWorld(world, "world")
gui.blockWhileServing()
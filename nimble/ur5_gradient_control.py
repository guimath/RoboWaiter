import nimblephysics as nimble
import torch
import numpy as np


def point_to_line_distance(p0, p1, p2):
    """
    Calculate the distance from point p0 to the line defined by points p1 and p2.
    p0 is the point for which the gradient is required.
    p1 and p2 define the line and are assumed to be constant tensors.
    """

    # Vector from p1 to p0 and p2 to p0
    v1 = p0 - p1
    v2 = p0 - p2

    # Cross product of v1 and v2
    cross_product = torch.cross(v1, v2)

    # Euclidean norm of the cross product
    numerator = torch.norm(cross_product)

    # Vector from p1 to p2
    line_vector = p2 - p1

    # Euclidean norm of the line vector
    denominator = torch.norm(line_vector)

    # Distance calculation
    distance = numerator / denominator

    return distance


class Controller:
    def __init__(self, world, arm, numDofs=8):
        self.world = world
        self.arm=arm


        self.numDofs = numDofs-2 # Ignore the 2 Dofs for the end effector

        # If the world has extra joints that should not be controllable, then
        # the number of DOFs will be more than necessary. We will remove those DOFs
        # from the action space, i.e., the remaining DOFs will be zero by default.
        # We assume that the first numDofs are the joints that should be controlled.
        # All the remaining DOFs will be removed
        worldDofs = self.world.getNumDofs()
        if numDofs is not None:
            for i in range(numDofs, worldDofs):
                world.removeDofFromActionSpace(i)
        else:
            self.numDofs = worldDofs

        self.state_size = self.world.getState().shape[-1]

        self.collision_group = collision_group = world.getConstraintSolver().getCollisionGroup()
        self.ignore_fingers=False


        self.node_to_ikmap_index = {}

        self.ikMap = nimble.neural.IKMapping(world)
        for i, body in enumerate(self.arm.getBodyNodes()):
            self.ikMap.addSpatialBodyNode(body)
            self.node_to_ikmap_index[body.getName()] = [*range(i*6, i*6+6)]


        self.gripper_close=False


        self.arm.clampPositionsToLimits()
        for i in range(self.arm.getNumJoints()):
            _joint = self.arm.getJoint(i)
            print(_joint.getName(), _joint.getPositionLowerLimits(), _joint.getPositionUpperLimits())
            _joint.setPositionLimitEnforced(True)

    def timestep(self, world, state, action):

        if self.gripper_close:
            action = torch.concat([action, -torch.tensor([1e-4, 1e-4])])
        else:
            action = torch.concat([action, torch.tensor([1e-4, 1e-4])])


        out = nimble.timestep(world, state, action)
        out[self.numDofs+1] = out[self.numDofs]

        self.arm.clampPositionsToLimits()
    
        return out

    def check_arm_in_collision(self):
        collision_result = nimble.collision.CollisionResult()
        self.collision_group.collide(nimble.collision.CollisionOption(), collision_result)

        out = []
        for i, node in enumerate(self.arm.getBodyNodes()):
            if collision_result.inCollision(node): 
                print(node.getName())               
                if self.ignore_fingers and node.getName() in ["right_inner_finger", "left_inner_finger", "right_inner_knuckle", "left_inner_knuckle"]:
                    continue
                out.append(node)
        return out

    def optimize_loss(self, loss_fn, num_actions=50, init_state=None, epochs=5000, lr=1.0, actions=None, vis=True, lower_bound=0.001):
        if init_state is None:
            init_state = torch.tensor(self.world.getState())

        if actions is None:
            actions, _ = self.keep_still(num_actions=num_actions, init_state=init_state)

        avoid_states = []

        optimizer = torch.optim.Adam([{'params': actions}], lr=lr)

        for k in range(epochs):
            state = init_state.clone().detach()
            states = [state]
            ikMap_values = []
            for i in range(num_actions):
                action = actions[i]
                state = self.timestep(world, state, action)
                
                states.append(state)
                
                ikMap_value = nimble.map_to_pos(world, self.ikMap, state)
                ikMap_values.append(ikMap_value)

                self.world.setState(state.detach())
                
                objects_in_collision = self.check_arm_in_collision()
                
                if objects_in_collision!=[]:
                    avoid_state = [state.clone().detach()]
                    for body in objects_in_collision:
                        avoid_state.append((body, ikMap_value[self.node_to_ikmap_index[body.getName()]].clone().detach()))
                    avoid_states.append(avoid_state)
                    break
                
            

            self.world.reset()  

            loss = loss_fn(states, ikMap_values, avoid_states)


            if k%10==0:
                orient = (ikMap_values[-1][self.node_to_ikmap_index["gripper_center_link"]][:3])
                position = ikMap_values[-1][self.node_to_ikmap_index["gripper_center_link"]][3:]
                print(f"Loss for epoch {k}: {loss.detach().numpy()} Final Orientation: {orient.detach().numpy()} Final Position: {position.detach().numpy()}")
                
                if vis:
                    gui.loopStates(states)

            if loss<lower_bound and objects_in_collision==[]:
                return actions, [state.detach() for state in states]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

        return actions, [state.detach() for state in states]

    # The below function is used to find the initial actions that keep the arm at the 
    # current position without moving
    def keep_still(self, num_actions=60, init_state=None, initial_actions=None, epochs=150):
        print(f"Trying to keep still with {num_actions} actions")
        if initial_actions is None:
            actions = torch.zeros((num_actions, self.numDofs), requires_grad=True)
        else:
            actions = initial_actions.clone()
            actions.requires_grad = True

        if init_state is None:
            init_state = torch.tensor(self.world.getState())
        assert not self.check_arm_in_collision() # Need to change this to only consider contact between arm and the floor

        ikMap_value = nimble.map_to_pos(world, self.ikMap, init_state)
        position = ikMap_value[self.node_to_ikmap_index["gripper_center_link"]][3:]
        print(position) 
        # Check if there is collision if we were to simulate the current number of steps
        state = init_state.clone()
        collision = False
        for i in range(num_actions):
            action = actions[i]
            state = self.timestep(world, state, action)
            self.world.setState(state.detach())
            if self.check_arm_in_collision():
                collision = True
                break

        # If there is collision, then call with a smaller number of actions
        self.world.setState(init_state)
        if collision:
            new_num_actions = num_actions//2 + 1
            new_initial_actions, _ = self.keep_still(new_num_actions, initial_actions=actions[:new_num_actions].detach(), epochs=epochs*2)
            np.save(f'save_still_numact_{new_num_actions}.npy', actions.detach().numpy())
            # actions = torch.concat([new_initial_actions, torch.tensor(np.array([new_initial_actions[-1].detach().numpy()]*new_num_actions))], axis=0)
            actions = torch.concat([new_initial_actions, new_initial_actions], axis=0)
            actions = actions[:num_actions].detach().clone()
            actions.requires_grad = True
            print(f"Completed with {new_num_actions} actions, now trying to keep still with {num_actions} actions")
        
        def loss_fn(states, ikMap_values, avoid_states):
            init_state = states[0]
            # loss1 = [torch.square(state-init_state)[:self.numDofs].sum() for state in states])/num_actions
            loss1 = [(torch.square(state[self.state_size//2:self.state_size//2+self.numDofs])).sum() for state in states]
            loss1 = sum(loss1)/num_actions


            indices = self.node_to_ikmap_index["robotiq_base_link"]
            init_pose = ikMap_values[0][indices]
            loss2 = sum([torch.norm(ikMap_value[indices]-init_pose) for ikMap_value in ikMap_values])/num_actions


            loss_avoid_floor = torch.tensor([0], dtype=torch.float64)
            for ikMap_value in ikMap_values:
                base_link_height = ikMap_value[self.node_to_ikmap_index["robotiq_base_link"]][4]
                if base_link_height<0.05: # This is the forearm link y-value
                    loss_avoid_floor += 1/(base_link_height+1e-1)


            return loss1 + loss2 + loss_avoid_floor
            

        
        actions, states = self.optimize_loss(loss_fn, num_actions=num_actions, init_state = init_state, epochs=epochs, lr=0.5*num_actions/30, actions=actions, vis=True)

        return actions, states


    def move_gripper (self, position, orientation, num_actions=60, init_state=None, keep_orientation=False, alpha_move=1, alpha_orientation=0.1, alpha_stop=0.2, alpha_straight_line=0, alpha_collision_avoidance=1, epochs=5000, lr=1.0, lower_bound=0.001):
        if init_state is None:
            init_state = torch.tensor(self.world.getState())
        else:
            init_state = torch.tensor(init_state)
        print(f"Start State: {init_state}")

        init_pose = nimble.map_to_pos(self.world, self.ikMap, init_state).detach()
        init_orientation = init_pose[:3]

        target_position = torch.tensor(position)

        # Orientation is assumed to be composed two vectors.
        # One represents the direction of the robotiq_link and the other
        # is the orientaiton of gripper_center to the left_finger
        target_orientation_robotiq = torch.tensor(orientation[0])
        target_orientation_ee = torch.tensor(orientation[1])
        assert torch.nn.functional.cosine_similarity(target_orientation_robotiq, target_orientation_ee, dim=0)<1e-8
        print(f"Target Position: {target_position}, Target Orientation Robotiq: {target_orientation_robotiq}, Target Orientation EE: {target_orientation_ee}")

        def loss_fn(states, ikMap_values, avoid_states):
            num_actions = len(states)

            final_gripper_pose = ikMap_values[-1][self.node_to_ikmap_index["gripper_center_link"]]

            if keep_orientation:
                loss_orientation = torch.tensor([0], dtype=torch.float64)
                for ikMap_value in ikMap_values:
                    direction_robotiq = ikMap_value[self.node_to_ikmap_index["gripper_center_link"]][3:] - ikMap_value[self.node_to_ikmap_index["robotiq_base_link"]][3:]
                    loss_orientation += 1-torch.nn.functional.cosine_similarity(direction_robotiq, target_orientation_robotiq, dim=0)

                    direction_ee = ikMap_value[self.node_to_ikmap_index["left_inner_finger"]][3:]-ikMap_value[self.node_to_ikmap_index["gripper_center_link"]][3:]
                    loss_orientation += 1-torch.abs(torch.nn.functional.cosine_similarity(direction_ee, target_orientation_ee, dim=0))

                loss_orientation /= num_actions
            else:
                direction_robotiq = final_gripper_pose[3:] - ikMap_values[-1][self.node_to_ikmap_index["robotiq_base_link"]][3:]
                loss_orientation = 1-torch.nn.functional.cosine_similarity(direction_robotiq, target_orientation_robotiq, dim=0)

                direction_ee = ikMap_values[-1][self.node_to_ikmap_index["left_inner_finger"]][3:]-ikMap_values[-1][self.node_to_ikmap_index["gripper_center_link"]][3:]
                loss_orientation += 1-torch.abs(torch.nn.functional.cosine_similarity(direction_ee, target_orientation_ee, dim=0))

            loss_orientation = alpha_orientation*loss_orientation

            loss_move =  alpha_move*(torch.norm(final_gripper_pose[3:] - target_position)) #+ (sum([(ikMap_value[:3] - init_orientation)**2 for ikMap_value in ikMap_values]).mean()/num_actions) 
            

            loss_stop = alpha_stop*(states[-1][self.state_size//2:self.state_size//2+self.numDofs]**2).mean()

            loss_avoid_floor = torch.tensor([0], dtype=torch.float64)
            for ikMap_value in ikMap_values:
                forearm_link_height = ikMap_value[self.node_to_ikmap_index["forearm_link"]][4]
                if forearm_link_height<0.2: # This is the forearm link y-value
                    loss_avoid_floor += 1/(forearm_link_height+1e-2)

            for ikMap_value in ikMap_values:
                base_link_height = ikMap_value[self.node_to_ikmap_index["robotiq_base_link"]][4]
                if base_link_height<0.1: # This is the robotiq link y-value
                    loss_avoid_floor += 1/(base_link_height+1e-2)

            # Heuristic function
            loss_distance_to_center = torch.tensor([0], dtype=torch.float64)
            for ikMap_value in ikMap_values:
                distance_to_center = torch.norm(ikMap_value[self.node_to_ikmap_index["robotiq_base_link"]][[3, 5]])
                if distance_to_center<0.2:
                    loss_distance_to_center += 1/(distance_to_center+1e-4)


            loss_avoid_self_collision = torch.tensor([0], dtype=torch.float64)
            for avoid_state in avoid_states:
                for i, state in enumerate(states):
                    difference = torch.max((state-avoid_state[0])**2)
                    if difference<0.4:
                        for (body_node, reference_pose) in avoid_state[1:]:
                            body_pose = ikMap_values[i-1][self.node_to_ikmap_index[body_node.getName()]]
                            loss_avoid_self_collision += 1/(torch.norm(body_pose-reference_pose)+1)
            loss_avoid_self_collision = alpha_collision_avoidance * loss_avoid_self_collision


            # For the final motion, it is assume that the gripper will travel in a straight line to the destination
            loss_straight_line = torch.tensor([0], dtype=torch.float64)
            init_position = init_pose[self.node_to_ikmap_index["robotiq_base_link"]][3:]
            for ikMap_value in ikMap_values:
                position = ikMap_value[self.node_to_ikmap_index["robotiq_base_link"]][3:]
                loss_straight_line += point_to_line_distance(position, init_position, target_position)
            loss_straight_line = alpha_straight_line * loss_straight_line/num_actions

            loss = loss_move + loss_stop + loss_avoid_floor + loss_avoid_self_collision + loss_distance_to_center + loss_orientation + loss_straight_line

            print(f"Loss Move: {loss_move.detach().numpy().item():.3f}, Loss Stop: {loss_stop.detach().numpy().item():.3f}, ",
            f"Loss Avoid Floor: {loss_avoid_floor.detach().numpy().item():.3f}, Loss Avoid Self Collision: {loss_avoid_self_collision.detach().numpy().item():.3f}, ",
            f"Loss Distance to Center: {loss_distance_to_center.detach().numpy().item():.3f}, Loss Orientation: {loss_orientation.detach().numpy().item():.3f}, ",
            f"Loss Straight Line:{loss_straight_line.detach().numpy().item():.3f}")


            return loss

        return self.optimize_loss(loss_fn, num_actions=num_actions, init_state=init_state, epochs=epochs, lr=lr, lower_bound=lower_bound)

    def move_gripper_to_grab_object_horizontal(self, position, theta=None):
        # Note if orientation is provided only the y-value will be heeded. It is assumed that the x and z value are 0
        if theta==None:
            theta = torch.atan2(torch.tensor(position[0]), torch.tensor(position[2]))

        # Move to the a location which is 0.1m away from the desired location
        coarse_position_x = position[0] - 0.2*torch.sin(theta)
        coarse_position_z = position[2] - 0.2*torch.cos(theta)
        coarse_position = [coarse_position_x, position[1], coarse_position_z]

        orientation_robotiq = [np.sin(theta), 0, np.cos(theta)]
        orientation_ee = [-np.cos(theta),0,np.sin(theta)]


        initial_actions, initial_states = self.move_gripper(coarse_position, [orientation_robotiq, orientation_ee], epochs=2000, lr=1.0, lower_bound=0.01, alpha_orientation=5, alpha_stop=0.1, alpha_move=3.0)
        return initial_actions, initial_states
        initial_actions = np.load("complete_sequence_actions.npy")[:60]
        initial_states = np.load("complete_sequence_states.npy")[:60]


        self.ignore_fingers = True
        for finger_part in ["right_inner_finger", "left_inner_finger", "right_inner_knuckle", "left_inner_knuckle"]:
            arm.getBodyNode(finger_part).setCollidable(False)

        final_actions, final_states = self.move_gripper(position, [orientation_robotiq, orientation_ee], init_state = initial_states[-1], keep_orientation=True, alpha_straight_line=50, alpha_move=10, alpha_orientation=10, alpha_stop=10, alpha_collision_avoidance=0, epochs=600, lr=1.0)

        self.ignore_fingers = False
        for finger_part in ["right_inner_finger", "left_inner_finger", "right_inner_knuckle", "left_inner_knuckle"]:
            arm.getBodyNode(finger_part).setCollidable(True)

        return torch.concat([torch.tensor(initial_actions), final_actions]), list(initial_states)+final_states

    def move_gripper_to_grab_object_vertical(self, position, theta=0):
        coarse_position_y = position[1] + 0.2
        coarse_position = [position[0], coarse_position_y, position[2]]

        gui.nativeAPI().createSphere("goal_pos", pos=np.array(coarse_position), radii=np.array([0.01, 0.01, 0.01]))

        orientation_robotiq = [0, -1, 0]
        orientation_ee = [np.sin(theta), 0, np.cos(theta)]

        initial_actions, initial_states = self.move_gripper(coarse_position, [orientation_robotiq, orientation_ee], epochs=2000, lr=1.0, lower_bound=0.01, alpha_orientation=5, alpha_stop=0.1, alpha_move=10.0)

        self.ignore_fingers = True
        for finger_part in ["right_inner_finger", "left_inner_finger", "right_inner_knuckle", "left_inner_knuckle"]:
            arm.getBodyNode(finger_part).setCollidable(False)

        final_actions, final_states = self.move_gripper(position, [orientation_robotiq, orientation_ee], init_state = initial_states[-1], keep_orientation=True, alpha_straight_line=5, alpha_move=10, alpha_orientation=10, alpha_stop=10, alpha_collision_avoidance=0, epochs=600, lr=1.0)

        self.ignore_fingers = False
        for finger_part in ["right_inner_finger", "left_inner_finger", "right_inner_knuckle", "left_inner_knuckle"]:
            arm.getBodyNode(finger_part).setCollidable(True)

        return torch.concat([torch.tensor(initial_actions), final_actions]), list(initial_states)+final_states


    def close_gripper(self, init_state=None):

        assert not self.gripper_close, "Gripper is already closed"

        if init_state is None:
            init_state = torch.tensor(self.world.getState())
        else:
            init_state = torch.tensor(init_state)
        print(f"Current State: {init_state}")


        self.gripper_close=True
        actions, states = self.keep_still(init_state=init_state, num_actions=50, epochs=1000)

        if torch.all(states[-1][self.numDofs:self.numDofs+2]<1e-5):
            print("Successfully Closed gripper")
        else:
            print("Unable to close the gripper successfully")
            return None

        print(states[-1])

        return actions, states

    def open_gripper(self, init_state=None):

        assert self.gripper_close, "Gripper is already open"

        if init_state is None:
            init_state = torch.tensor(self.world.getState())
        else:
            init_state = torch.tensor(init_state)
        print(f"Current State: {init_state}")


        self.gripper_close=False
        # actions, states = self.keep_still(init_state=init_state, num_actions=20, epochs=1000)

        print(states[-1])

        if torch.all(states[-1][self.state_size//2+self.numDofs:self.state_size//2+self.numDofs+2]<1e-3):
            print("Successfully opened gripper")
        else:
            print("Unable to open the gripper successfully")
            return None

        return actions, states
    
    def run_actions(self, world, actions):
        self.states = []
        nstep = len(actions)
        print(f'Running for {nstep} steps')
        # print(self.action)
        state = torch.tensor(world.getState())
        for i in range(nstep):
            state = self.timestep(world, state, actions[i])
            self.states += [state]
            print(f'{i}', end= '\r')
        print(f'{nstep} steps done')
        return self.states
        # self.states = self.states[-500:]
        

import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep
UR5_URDF    = ABS_PATH + 'nimble' + os.sep + 'ur5_gripper_ee.urdf'
GROUND_URDF = ABS_PATH + 'nimble' + os.sep + 'ground_obj.urdf'
# BOTTLE_URDF = ABS_PATH + 'nimble' + os.sep + 'bottle.urdf'
if __name__=="__main__":
    urdf_file_path = './ur5_gripper_reduced.urdf'
    world = nimble.simulation.World()
    world.setGravity([0, -5.0, 0])
    world.setTimeStep(0.01)


    arm = world.loadSkeleton(UR5_URDF)
    ground = world.loadSkeleton(GROUND_URDF)
    # bottle = world.loadSkeleton(BOTTLE_URDF)

    arm.setSelfCollisionCheck(True)
    shape_node = ground.getBodyNode(0).getShapeNode(1)
    shape_node.createCollisionAspect()


    state = torch.zeros(world.getState().shape[-1])
    state[1] = -0.5
    # state[6]=-0.2
    # state[7]=-0.2
    # controller = Controller(world, arm)
    world.setState(state)

    gui = nimble.NimbleGUI(world)
    gui.serve(8000)
    gui.nativeAPI().renderWorld(world, "world")

    # print(world.checkCollision())
    # gui.blockWhileServing()
    #original_pos = [ 0.3471, 0.3974, 0.7626]
    goal_pos = [-0.4, 0.2, 0.4]
    controller = Controller(world, arm)
    LOAD = True
    if LOAD :
        ld= np.load('init_actions.npy')
        ld = np.concatenate([ld, [ld[-1]]*200])
        print(ld)
        initial_actions=torch.tensor(ld)
        states1 = controller.run_actions(world, initial_actions)
    else :
        actions1, states1 = controller.keep_still(num_actions=30, epochs=2000)
        np.save('init_actions.npy',actions1.detach().numpy())
    # actions, states = controller.keep_still(num_actions=30)
    # actions, states = controller.move_gripper_to_grab_object_horizontal(goal_pos)
    # Loss for epoch 390: [0.22393036] Final Orientation: [2.10454623 0.22788057 2.10784857] Final Position: [0.34661228 0.26591437 0.79478263]
    gui.loopStates(states1)
    gui.blockWhileServing()

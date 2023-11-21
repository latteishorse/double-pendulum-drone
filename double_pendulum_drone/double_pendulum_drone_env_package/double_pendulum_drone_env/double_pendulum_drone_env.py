from double_pendulum_drone_env.event_handler import *
from double_pendulum_drone_env.DPDrone import *

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import os

class DPDEnv(gym.Env):
    def __init__(self, render_sim=False, n_steps=500, render_path=True, render_shade=True, shade_distance=70,   n_fall_steps=10, change_target=False, initial_throw=True):
        self.render_sim = render_sim
        self.render_path = render_path
        self.render_shade = render_shade
        
        if self.render_sim is True:
            self.init_pygame()
            self.flight_path = []
            self.drop_path = []
            self.path_drone_shade = []
            
        self.init_pymunk()

        self.max_time_steps = n_steps
        self.stabilisation_delay = n_fall_steps
        self.drone_shade_distance = shade_distance
        self.force_scale = 3000
        self.initial_throw = initial_throw
        self.change_target = change_target
        self.force = 0
        self.done = False
        self.first_step = True
        self.current_time_step = 0
        self.info = {}
        self.left_force = -1
        self.right_force = -1
        
        # Target position
        self.x_target = random.uniform(350, 550)
        self.y_target = random.uniform(450, 650)
        # self.x_target = 400
        # self.y_target = 650

        min_action = np.array([-1, -1], dtype=np.float32)
        max_action = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        min_observation = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        max_observation = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800,800))
        pygame.display.set_caption("Double pendulum drone")
        self.clock = pygame.time.Clock()

        script_dir = os.path.dirname(__file__)
        icon_path = os.path.join("img", "icon.png")
        icon_path = os.path.join(script_dir, icon_path)
        pygame.display.set_icon(pygame.image.load(icon_path))

        img_path = os.path.join("img", "shade.png")
        img_path = os.path.join(script_dir, img_path)
        self.shade_image = pygame.image.load(img_path)

    def init_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -1000)

        if self.render_sim is True:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            pymunk.pygame_util.positive_y_is_up = True

        # Starting position
        initial_x = random.uniform(350, 550)
        # initial_y = 250
        initial_y = random.uniform(350, 550)
        
        angle_rand = random.uniform(-np.pi/2, np.pi/2)
        self.cart_mass = 1
        self.cart = Drone(initial_x, initial_y, angle_rand, 20, 100, 0.2, 0.4, 0.4, self.space)
        self.drone_radius = self.cart.drone_radius
        
        # pendulum parameters
        self.pole_1_mass = 0.2
        self.pole_1_length = 15
        pole_1_thickness = 15
    
        self.pole_2_mass = 0.1
        self.pole_2_length = 10
        pole_2_thickness = 10
        
        alpha = random.uniform(14*np.pi/36, 22*np.pi/36)
        pole_1_x = initial_x + self.pole_1_length * np.cos(alpha)
        pole_1_y = initial_y + self.pole_1_length * np.sin(alpha)
        
        self.pole_1 = Pole(pole_1_x, pole_1_y, initial_x, initial_y, pole_1_thickness, self.pole_1_mass, (255, 51, 51), self.space)

        beta = random.uniform(16*np.pi/36, 20*np.pi/36)
        pole_2_x = pole_1_x + self.pole_2_length * np.cos(beta)
        pole_2_y = pole_1_y + self.pole_2_length * np.sin(beta)
        self.pole_2 = Pole(pole_2_x, pole_2_y, pole_1_x, pole_1_y, pole_2_thickness, self.pole_2_mass, (255, 153, 51), self.space)

        self.pivot_1 = pymunk.PivotJoint(self.cart.body, self.pole_1.body, (0, 0), (-self.pole_1_length/2, 0))
        self.pivot_1.error_bias = 0
        self.pivot_1.collide_bodies = False
        self.space.add(self.pivot_1)
        
        self.pivot_2 = pymunk.PivotJoint(self.pole_1.body, self.pole_2.body, (self.pole_1_length/2, 0), (-self.pole_2_length/2, 0))
        self.pivot_2.error_bias = 0
        self.pivot_2.collide_bodies = False
        self.space.add(self.pivot_2)


    def step(self, action):
        self.force = action[0] * self.force_scale
        self.cart.body.apply_force_at_local_point((self.force, 0), (0, 0))
        
        pymunk.Body.update_velocity(self.pole_1.body, Vec2d(0, 0), 0.999, 1/60.0)
        pymunk.Body.update_velocity(self.pole_2.body, Vec2d(0, 0), 0.999, 1/60.0)
        pymunk.Body.update_velocity(self.cart.body, Vec2d(0, 0), 0.9999, 1/60.0)
        
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True and self.render_shade is True: self.add_drone_shade()
            self.info = self.initial_movement()

        self.left_force = (action[0]/2 + 0.5) * self.force_scale
        self.right_force = (action[1]/2 + 0.5) * self.force_scale
        self.cart.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.left_force), (-self.drone_radius, 0))
        self.cart.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.right_force), (self.drone_radius, 0))
        self.space.step(1.0/60)
        self.current_time_step += 1

        if self.first_step is True:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()
            self.first_step = False

        else:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()

        if self.render_sim is True and self.render_shade is True:
            x, y = self.cart.frame_shape.body.position
            if np.abs(self.shade_x-x) > self.drone_shade_distance or np.abs(self.shade_y-y) > self.drone_shade_distance:
                self.add_drone_shade()

        #reward function
        obs = self.get_observation()
        reward = 0

        # drone location (x,y) reward
        reward = ((1.0/(np.abs(obs[4])+0.1)) + (1.0/(np.abs(obs[5])+0.1))) 
        # reward = (1 + 0.5*(-np.abs(obs[4])+1) + 0.5*(-np.abs(obs[5])+1)) * 100
        # distance_x = obs[4]  
        # distance_y = obs[5]  

        ## pendulum reward 
        # pole_1_angle = obs[8] 
        # pole_1_angular_velocity = obs[10]  
        pole_2_angle = obs[9] 
        pole_2_angular_velocity = obs[11]  

        # pendulum - angle reward
        # angle_reward1 = 1.0 - abs(pole_1_angle)
        angle_reward2 = 1.0 - abs(pole_2_angle) 

        # pendulum - velocity reward

        # angular_velocity_reward1 = 1.0 - abs(pole_1_angular_velocity) 
        angular_velocity_reward2 = 1.0 - abs(pole_2_angular_velocity)  

        ## abnormal reward
        if np.abs(obs[8]) == 1 or np.abs(obs[9]) == 1:
            self.done = True
            abnormal_reward = -30
        if np.abs(obs[4]) == 1 or np.abs(obs[5]) == 1:
            self.done = True
            abnormal_reward = -30
            
        # total_reward = angle_reward2 + angular_velocity_reward2 + reward + abnormal_reward        
        total_reward = reward + abnormal_reward        

        if self.current_time_step == self.max_time_steps:
            self.done = True

        return obs, total_reward, self.done, self.info

    def get_observation(self):
        # pole
        pole_1_angle = -9*self.pole_1.body.angle/np.pi + 4.5
        pole_1_angle = np.clip(pole_1_angle, -1, 1)
        pole_2_angle = -9*self.pole_2.body.angle/np.pi + 4.5
        pole_2_angle = np.clip(pole_2_angle, -1, 1)
        pole_1_angular_velocity = np.clip(self.pole_1.body.angular_velocity/15, -1, 1)
        pole_2_angular_velocity = np.clip(self.pole_2.body.angular_velocity/15, -1, 1)

        # drone
        velocity_x, velocity_y = self.cart.frame_shape.body.velocity_at_local_point((0, 0))
        velocity_x = np.clip(velocity_x/1330, -1, 1)
        velocity_y = np.clip(velocity_y/1330, -1, 1)

        omega = self.cart.frame_shape.body.angular_velocity
        omega = np.clip(omega/12, -1, 1)

        alpha = self.cart.frame_shape.body.angle
        alpha = np.clip(alpha/(np.pi/2), -1, 1)

        x, y = self.cart.frame_shape.body.position

        if x < self.x_target:
            distance_x = np.clip((x/self.x_target) - 1, -1, 0)

        else:
            distance_x = np.clip((-x/(self.x_target-800) + self.x_target/(self.x_target-800)) , 0, 1)

        if y < self.y_target:
            distance_y = np.clip((y/self.y_target) - 1, -1, 0)

        else:
            distance_y = np.clip((-y/(self.y_target-800) + self.y_target/(self.y_target-800)) , 0, 1)

        pos_x = np.clip(x/400.0 - 1, -1, 1)
        pos_y = np.clip(y/400.0 - 1, -1, 1)

        return np.array([velocity_x, velocity_y, omega, alpha, distance_x, distance_y, pos_x, pos_y, pole_1_angle, pole_2_angle, pole_1_angular_velocity, pole_2_angular_velocity])

    def render(self, mode='human', close=False):      
        x, y = self.cart.body.position
        scale = 1.0/10
        
        pygame_events(self.space, self, self.change_target)
        self.screen.fill((248, 248, 248))

        pygame.draw.rect(self.screen, (160, 160, 160), pygame.Rect(0, 0, 800, 800), 8)

        if len(self.path_drone_shade):
            for shade in self.path_drone_shade:
                image_rect_rotated = pygame.transform.rotate(self.shade_image, shade[2]*180.0/np.pi)
                shade_image_rect = image_rect_rotated.get_rect(center=(shade[0], 800-shade[1]))
                self.screen.blit(image_rect_rotated, shade_image_rect)

        self.space.debug_draw(self.draw_options)

        vector_scale = 0.005
        l_x_1, l_y_1 = self.cart.frame_shape.body.local_to_world((-self.drone_radius, 0))
        l_x_2, l_y_2 = self.cart.frame_shape.body.local_to_world((-self.drone_radius, self.force_scale*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (l_x_1, 800-l_y_1), (l_x_2, 800-l_y_2), 4)

        l_x_2, l_y_2 = self.cart.frame_shape.body.local_to_world((-self.drone_radius, self.left_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (l_x_1, 800-l_y_1), (l_x_2, 800-l_y_2), 4)

        r_x_1, r_y_1 = self.cart.frame_shape.body.local_to_world((self.drone_radius, 0))
        r_x_2, r_y_2 = self.cart.frame_shape.body.local_to_world((self.drone_radius, self.force_scale*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (r_x_1, 800-r_y_1), (r_x_2, 800-r_y_2), 4)

        r_x_2, r_y_2 = self.cart.frame_shape.body.local_to_world((self.drone_radius, self.right_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (r_x_1, 800-r_y_1), (r_x_2, 800-r_y_2), 4)

        pygame.draw.circle(self.screen, (255, 0, 0), (self.x_target, 800-self.y_target), 5)

        if len(self.flight_path) > 2:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

        if len(self.drop_path) > 2:
            pygame.draw.aalines(self.screen, (255, 0, 0), False, self.drop_path)

        pygame.display.flip()
        self.clock.tick(60)



    def reset(self):
        self.__init__(render_sim=False, n_steps=500, render_path=True, render_shade=True, shade_distance=70,   n_fall_steps=100, change_target=False, initial_throw=True)
        return self.get_observation()

    def close(self):
        pygame.quit()

    def initial_movement(self):
        if self.initial_throw is True:
            throw_angle = random.random() * 2*np.pi
            throw_force = random.uniform(0, 25000)
            throw = Vec2d(np.cos(throw_angle)*throw_force, np.sin(throw_angle)*throw_force)

            self.cart.frame_shape.body.apply_force_at_world_point(throw, self.cart.frame_shape.body.position)

            throw_rotation = random.uniform(-3000, 3000)
            self.cart.frame_shape.body.apply_force_at_local_point(Vec2d(0, throw_rotation), (-self.drone_radius, 0))
            self.cart.frame_shape.body.apply_force_at_local_point(Vec2d(0, -throw_rotation), (self.drone_radius, 0))

            self.space.step(1.0/60)
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()

        else:
            throw_angle = None
            throw_force = None
            throw_rotation = None

        initial_stabilisation_delay = self.stabilisation_delay
        while self.stabilisation_delay != 0:
            self.space.step(1.0/60)
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True: self.render()
            self.stabilisation_delay -= 1

        self.stabilisation_delay = initial_stabilisation_delay

        return {'throw_angle': throw_angle, 'throw_force': throw_force, 'throw_rotation': throw_rotation}

    def add_postion_to_drop_path(self):
        x, y = self.cart.frame_shape.body.position
        self.drop_path.append((x, 800-y))

    def add_postion_to_flight_path(self):
        x, y = self.cart.frame_shape.body.position
        self.flight_path.append((x, 800-y))

    def add_drone_shade(self):
        x, y = self.cart.frame_shape.body.position
        self.path_drone_shade.append([x, y, self.cart.frame_shape.body.angle])
        self.shade_x = x
        self.shade_y = y

    def change_target_point(self, x, y):
        self.x_target = x
        self.y_target = y
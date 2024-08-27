import numpy as np
import gym
from gym.utils import EzPickle
from gym import spaces
import Box2D
from Box2D.b2 import (fixtureDef, polygonShape, circleShape)
from typing import Optional
from PIL import Image
import pygame
from pygame import gfxdraw
import imageio

class SlidePickupBox2DEnv(gym.Env, EzPickle):
    def __init__(self, device, cfg):
        EzPickle.__init__(self)
        
        self.device = device
        self.env_cfg = cfg
        self.collision_thresh = self.env_cfg.collision_thresh

        self.viewer = None

        # For pygame rendering
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

        self.world = Box2D.b2World()
        self.world.gravity = tuple(self.env_cfg['gravity'])

        self.N_O = 2
        self.n = 4*self.N_O
        self.m = 2
        self.dt = self.env_cfg.dt

        self.ppm = self.env_cfg['scene']['pixels_per_meter']
        self.generate_env()
        self.state = np.zeros(self.n)

    def generate_env(self):
        self._current_timestep = 0
        # Make table
        self.table = self.world.CreateStaticBody(
            position = tuple(self.env_cfg.table.initial_pos),
            fixtures = fixtureDef(
                shape=polygonShape(box=(self.env_cfg.table.size[0]/2, self.env_cfg.table.size[1]/2)),
                friction=self.env_cfg.table.friction,
                restitution=self.env_cfg.table.restitution,
            )
        )
        self.table.color1 = tuple(self.env_cfg.table.color1)
        self.table.color2 = tuple(self.env_cfg.table.color2)

        # Make block to be picked
        density_block1 = self.env_cfg.block1.mass/(self.env_cfg.block1.size[0]*self.env_cfg.block1.size[1])
        x_block1 = self.env_cfg.table.initial_pos[0] - self.env_cfg.table.size[0]/2 + self.env_cfg.block1.size[0]/2 # aligned with the left of the table
        y_block1 = self.env_cfg.table.initial_pos[1] + self.env_cfg.table.size[1]/2 + self.env_cfg.block1.size[1]/2 + self.collision_thresh
        self.block1 = self.world.CreateDynamicBody(
            position=(x_block1, y_block1),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(box=(self.env_cfg.block1.size[0]/2, self.env_cfg.block1.size[1]/2)),
                density=density_block1,
                friction=self.env_cfg.block1.friction,
                restitution=self.env_cfg.block1.restitution
            ),
            fixedRotation = False,
        )
        self.block1.color1 = tuple(self.env_cfg.block1.color1)
        self.block1.color2 = tuple(self.env_cfg.block1.color2)
        self.block1.bullet = True

        # Make second block
        density_block2 = self.env_cfg.block2.mass/(self.env_cfg.block2.size[0]*self.env_cfg.block2.size[1])
        x_block2 = self.env_cfg.table.initial_pos[0] - self.env_cfg.table.size[0]/2 + self.env_cfg.block2.size[0]/2 # aligned with the left of the table
        y_block2 = y_block1 + self.env_cfg.block1.size[1]/2 + self.env_cfg.block2.size[1]/2 + self.collision_thresh
        self.block2 = self.world.CreateDynamicBody(
            position=(x_block2, y_block2),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(box=(self.env_cfg.block2.size[0]/2, self.env_cfg.block2.size[1]/2)),
                density=density_block2,
                friction=self.env_cfg.block2.friction,
                restitution=self.env_cfg.block2.restitution
            ),
            fixedRotation = False,
        )
        self.block2.color1 = tuple(self.env_cfg.block2.color1)
        self.block2.color2 = tuple(self.env_cfg.block2.color2)
        self.block2.bullet = True

        self._drawlist = [self.table, self.block1, self.block2]
        self.state = self._get_state()

    def reset(self, start=None):
        self._current_timestep = 0

        if start is None:
            self.state = self.sample_random_state()
        else:
            self.state = start

        self.reset_to_state(self.state)
        self.state = self._get_state()
        
    def reset_to_state(self, state):
        pass

    def sample_random_state(self):
        pass

    def _destroy(self):
        if not self.table: return
        self.world.DestroyBody(self.table)
        self.world.DestroyBody(self.block1)
        self.world.DestroyBody(self.block2)
        self.table = None
        self.block1 = None
        self.block2 = None
        self._blocks = None
    
    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return Image.fromarray(np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        ))
    
    def map_pos_to_pixel(self, pos):
        pix_x = int((pos[0] - self.env_cfg.scene.low[0])*self.ppm) # (x-x_min)*ppm
        pix_y = int((pos[1] - self.env_cfg.scene.low[1])*self.ppm) # (y-y_min)*ppm
        return pix_x, pix_y

    def polygon_vertices_to_left_top_width_height(self, trans, vertices):
        min_x = min(vertex[0] for vertex in vertices)
        max_x = max(vertex[0] for vertex in vertices)
        min_y = min(vertex[1] for vertex in vertices)
        max_y = max(vertex[1] for vertex in vertices)

        left = min_x + trans.position[0]
        bottom = min_y + trans.position[1]
        pix_left, pix_bottom = self.map_pos_to_pixel((left, bottom))

        pix_width = int(self.ppm*(max_x - min_x))
        pix_height = int(self.ppm*(max_y - min_y))
        return pix_left, pix_bottom, pix_width, pix_height

    def step(self, action):
        done = False
        self._current_timestep += 1

        self.block1.ApplyForceToCenter(tuple(action), wake=True)
        self.block2.awake = True
        self.world.Step(self.dt, 6*30, 2*30)

        self.state = self._get_state()

    def _get_state(self):
        state = [
            self.block1.position[0],
            self.block1.position[1],
            self.block1.linearVelocity[0],
            self.block1.linearVelocity[1],
            self.block2.position[0],
            self.block2.position[1],
            self.block2.linearVelocity[0],
            self.block2.linearVelocity[1],
            ]
        
        return np.array(state)

    def render(self, mode='human'):
        # - human: render to the current display or terminal and
        #   return nothing. Usually for human consumption.
        # - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
        #   representing RGB values for an x-by-y pixel image, suitable
        #   for turning into a video.
        # - ansi: Return a string (str) or StringIO.StringIO containing a
        #   terminal-style text representation. The text can include newlines
        #   and ANSI escape sequences (e.g. for colors).
        
        sw = int(self.ppm*(self.env_cfg.scene.high[0] - self.env_cfg.scene.low[0]))
        sh = int(self.ppm*(self.env_cfg.scene.high[1] - self.env_cfg.scene.low[1]))

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((sw, sh))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((sw, sh))
        self.surf.fill((255, 255, 255))  # background color white

        for obj in self._drawlist:
            color = tuple(int(x * 255) for x in obj.color1)
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pix_x, pix_y = self.map_pos_to_pixel(trans*f.shape.pos)
                    gfxdraw.filled_circle(self.surf, pix_x, pix_y, int(f.shape.radius*self.ppm), color)
                
                if type(f.shape) is polygonShape:
                    pix_x, pix_y, pix_width, pix_height = self.polygon_vertices_to_left_top_width_height(trans,f.shape.vertices)
                    rect = pygame.Rect(pix_x, pix_y, pix_width, pix_height)  # left, bottom, width, height
                    gfxdraw.box(self.surf, rect, color)

        gpix_x, gpix_y = self.map_pos_to_pixel(self.env_cfg.goal)
        gfxdraw.aacircle(self.surf, gpix_x, gpix_y, 20, (255, 0, 0)) # goal in red

        self.surf = pygame.transform.flip(self.surf, False, True)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        
        if mode == "rgb_array":
            return self._create_image_array(self.surf, (sw, sh))

        # from gym.envs.classic_control import rendering
        # if self.viewer is None:
        #     self.viewer = rendering.Viewer(sw, sh)
        # self.viewer.set_bounds(-sw/ppm/2, sw/ppm/2, -sh/ppm/2, sh/ppm/2)
        
        # # Plotting dynamic objects
        # for obj in self._drawlist:
        #     for f in obj.fixtures:
        #         trans = f.body.transform
        #         if type(f.shape) is circleShape:
        #             t = rendering.Transform(translation=trans*f.shape.pos)
        #             self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
        #             self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
        #         else:
        #             path = [trans*v for v in f.shape.vertices]
        #             self.viewer.draw_polygon(path, color=obj.color1)
        #             path.append(path[0])
        #             self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        # return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        """Closes the viewer.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import os
    # test env
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'
    cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/envs/gym_envs.yaml')
    env = SlidePickupBox2DEnv(0, cfg['slide_pickup_box2d_env-v1'])
    
    steps = 5000
    goal = np.array(env.env_cfg.goal)
    path = np.linspace(env.state[:2], goal, steps)
    K = np.array((1.,20.))
    images = []
    for i in range(steps):
        action = K*(path[i] - env.state[:2])
        env.step(action)
        rgb = env.render(mode='rgb_array')
        images.append(rgb)
    # rgb.save("/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/env_test.png")
    imageio.mimsave(f'/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/env_test_small.gif', 
        [np.array(img) for i, img in enumerate(images) if i%10 == 0], duration=10)
    # import ipdb; ipdb.set_trace()
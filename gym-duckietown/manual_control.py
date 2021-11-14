#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import pandas as pd

# from experiments.utils import save_img


# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map2')
parser.add_argument('--seed', type=int, default=11, help='random seed')
args = parser.parse_args()

env = DuckietownEnv(
    map_name = args.map_name,
    domain_rand = False,
    draw_bbox = False,
    max_steps = args.max_steps,
    seed = args.seed
)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    if key_handler[key.W]:
        action = np.array([0.44, 0.0])
        printPic(action, intention='front')
    if key_handler[key.S]:
        action = np.array([-0.44, 0])
        printPic(action, intention='back')
    if key_handler[key.A]:
        action = np.array([0.35, +1])
        printPic(action, intention='left')
    if key_handler[key.D]:
        action = np.array([0.35, -1])
        printPic(action, intention='right')

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)
        im.save('./pics/screen.png')

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()

def printPic(action, intention):
    from PIL import Image
    import os
    
    obs, reward, done, info = env.step(action)
    dir='./pics'
    list = os.listdir(dir) 
    number_files = len(list)
    im = Image.fromarray(obs)
    im.save('./pics/%s.png' % (number_files))
    data = pd.read_csv(os.path.join('./pics/label.txt'), sep=',')
    new_row = pd.DataFrame([[number_files, reward, action[0], action[1], intention]], columns=data.columns)
    data = data.append(new_row)
    data.to_csv(os.path.join('./pics/label.txt'), sep=',', index=False)

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()

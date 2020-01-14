"""
 DDPG for Carla 0.9.6
"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import tensorflow as tf
import numpy as np
import time
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import DDPG

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
    import plot_record
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- find carla module ---------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------
# ==============================================================================
try:
    sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))) +
        '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from carla import Transform, Location, Rotation


# from agents.navigation.roaming_agent import RoamingAgent
# from agents.navigation.basic_agent import BasicAgent


# ==============================================================================
# -- Global functions ----------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [
        x for x in dir(
            carla.WeatherParameters) if re.match(
            '[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, actor_filter):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player1 = None
        self.vehicle2 = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        # ******I have made this random.choice just have one choice --> vehicle.bmw.grandtourer
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(
                self._actor_filter))
        # print("blue print")
        # print(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            # print("color")
            # print(blueprint.get_attribute('color').recommended_values)
            # ****** I have set the blueprint being white as '255,255,255'
            blueprint.set_attribute('color', '255,255,255')
        batch = []
        blueprint2 = random.choice(
            self.world.get_blueprint_library().filter(
                self._actor_filter))
        blueprint2.set_attribute('role_name', 'hero2')
        if blueprint2.has_attribute('color'):
            blueprint2.set_attribute('color', '0,0,0')
        # batch.append(SpawnActor(blueprint, transform))

        # Spawn the player.

        #     ******************* change positions ****************************
        # while self.player1 is None:
            # spawn_points = self.map.get_spawn_points()
            # print("length of sapwn_point is %d" % len(spawn_points) + ",just
            # choose one")  # 257 just choose one
        x_rand = random.randint(18000, 23000)
        x_rand_v2 = random.randint(x_rand + 1000, 25000)
        x_rand = x_rand / 100.0
        x_rand_v2 = x_rand_v2 / 100.0
        y_rand = random.randint(12855, 13455)
        y_rand = y_rand / 100.0
        x_speed_randon_v2 = random.randint(100, 3000)
        if x_speed_randon_v2 - 1000 > 0:
            x_speed_player = x_speed_randon_v2 - 100

        else:
            x_speed_player = 0
        x_speed_player = x_speed_player / 100

        x_speed_randon_v2 = x_speed_randon_v2 / 100
        spawn_point1 = Transform(
            Location(
                x=x_rand, y=y_rand, z=1.320625), Rotation(
                pitch=0.000000, yaw=179.999756, roll=0.000000))
        # ********************************* end ***************************
        if self.player1 is None:
            print("player1")
            self.player1 = self.world.try_spawn_actor(blueprint, spawn_point1)
            # print("player",self.player1)
        # if self.player1 is not None:
        spawn_point2 = Transform(
            Location(
                x=x_rand_v2, y=y_rand, z=1.320625), Rotation(
                pitch=0.000000, yaw=179.999756, roll=0.000000))
        if self.vehicle2 is None:
            print("vehicle2")
            self.vehicle2 = self.world.try_spawn_actor(
                blueprint2, spawn_point2)

            # print("vehicle2",self.vehicle2)
        if self.vehicle2 is not None:
            self.vehicle2.set_velocity(
                carla.Vector3D(
                    x=-x_speed_randon_v2,
                    y=0.00000,
                    z=0.000000))
        if self.player1 is not None:
            self.player1.set_velocity(
                carla.Vector3D(
                    x=-x_speed_player,
                    y=0.00000,
                    z=0.00000))
        if self.player1 is not None and self.vehicle2 is not None:
            # Set up the sensors.
            self.collision_sensor = CollisionSensor(self.player1, self.hud)
            self.lane_invasion_sensor = LaneInvasionSensor(self.player1, self.hud)
            self.gnss_sensor = GnssSensor(self.player1)
            self.camera_manager = CameraManager(self.player1, self.hud)
            self.camera_manager.transform_index = cam_pos_index
            self.camera_manager.set_sensor(cam_index, notify=False)
            actor_type = get_actor_display_name(self.player1)
            self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += 0 if reverse else 0
        self._weather_index %= len(self._weather_presets)
        print("weather_index")
        print(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player1.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        if self.camera_manager.sensor is not None:
            self.camera_manager.sensor.destroy()
        if self.camera_manager.sensor is not None:
            self.camera_manager.sensor = None
        if self.camera_manager.index is not None:
            self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player1,
            self.vehicle2]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def get_state(self):
        s = np.array([self.player1.get_location().x,
                      self.player1.get_location().y,
                      self.player1.get_location().z,
                      self.player1.get_velocity().x,
                      self.player1.get_velocity().y,
                      self.player1.get_velocity().z,
                      self.vehicle2.get_location().x,
                      self.vehicle2.get_location().y,
                      self.vehicle2.get_location().z,
                      self.vehicle2.get_velocity().x,
                      self.vehicle2.get_velocity().y,
                      self.vehicle2.get_velocity().z,
                      ])
        player1 = np.array([self.player1.get_location().x,
                      self.player1.get_location().y,
                      self.player1.get_location().z,
                      ])
        player2 = np.array([self.vehicle2.get_location().x,
                      self.vehicle2.get_location().y,
                      self.vehicle2.get_location().z,
                            ])
        dist = distance(player2, player1)
        return s, dist


# ==============================================================================
# -- RLControl -----------------------------------------------------------
# ==============================================================================
class RLControl(object):
    def __init__(self, world):
        # self._autopilot_enabled = start_in_autopilot
        # self.world = world
        if isinstance(world.player1, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._control.steer = 0
            self._control.throttle = 0

    def run_step(self, world, action):
        # not enough waypoints in the horizon? => add more!
        # print("player1")
        col_hist = world.collision_sensor.get_collision_history()
        # print("collision_history:")
        # print(col_hist)
        # print("frame_number_new:")
        col_fram = world.hud.frame
        # print(col_fram)

        # if col_hist.get(col_fram) is not None:
            # print("new_collision_value and newly start:")
            # print(col_hist.get(col_fram))

        control = carla.VehicleControl()
        action = action.tolist()
        control.steer = action[0]
        control.throttle = action[1]
        control.brake = 0
        # control.brake = action[2]
        control.hand_brake = False
        control.manual_gear_shift = False
        control.collisionFlag = col_hist.get(col_fram)
        if control.collisionFlag:
            print('collisionFlag')
        return control, control.collisionFlag

    def step(self, world, action):
        control, collision_flag = self.run_step(world, action)
        # print(world.player1.get_velocity())
        # print("vehicle2")
        vehicle1_x = world.player1.get_location().x
        vehicle1_y = world.player1.get_location().y
        vehicle1_z = world.player1.get_location().z
        self.vehicle1_location = np.array([vehicle1_x, vehicle1_y, vehicle1_z])
        vehicle1_vx = world.player1.get_velocity().x
        vehicle1_vy = world.player1.get_velocity().y
        vehicle1_vz = world.player1.get_velocity().z
        self.vehicle1_velocity = np.array(
            [vehicle1_vx, vehicle1_vy, vehicle1_vz])
        vehicle2_x = world.vehicle2.get_location().x
        vehicle2_y = world.vehicle2.get_location().y
        vehicle2_z = world.vehicle2.get_location().z
        self.vehicle2_location = np.array([vehicle2_x, vehicle2_y, vehicle2_z])
        vehicle2_vx = world.vehicle2.get_velocity().x
        vehicle2_vy = world.vehicle2.get_velocity().y
        vehicle2_vz = world.vehicle2.get_velocity().z
        self.vehicle2_velocity = np.array(
            [vehicle2_vx, vehicle2_vy, vehicle2_vz])

        state = np.concatenate(
            (self.vehicle1_location,
             self.vehicle1_velocity,
             self.vehicle2_location,
             self.vehicle2_velocity),
            axis=0)
        dist = distance(self.vehicle1_location, self.vehicle2_location)
        dest = np.array([90, 133, 80])
        r = reward_function(
            self.vehicle1_location,
            self.vehicle2_location,
            dest,
            collision_flag)
        done = False
        if collision_flag:
            done = True
        if distance(self.vehicle1_location, dest) < 5:
            done = True
        return control, state, r, done, dist


# ==============================================================================
# -- HUD -----------------------------------------------------------------
# ==============================================================================

# Info on the creen of pygame. The HUD means head up display in automobiles,
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player1.get_transform()  # get_transform
        v = world.player1.get_velocity()
        c = world.player1.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.bmw.*')
        self._info_text = [
            'Server:  % 16.0f FPS' %
            self.server_fps,
            'Client:  % 16.0f FPS' %
            clock.get_fps(),
            '',
            'Vehicle: % 20s' %
            get_actor_display_name(
                world.player1,
                truncate=20),
            'Map:     % 20s' %
            world.map.name,
            'Simulation time: % 12s' %
            datetime.timedelta(
                seconds=int(
                    self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' %
            (3.6 *
             math.sqrt(
                 v.x ** 2 +
                 v.y ** 2 +
                 v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' %
            (t.rotation.yaw,
             heading),
            'Location:% 20s' %
            ('(% 5.1f, % 5.1f)' %
             (t.location.x,
              t.location.y)),
            'GNSS:% 24s' %
            ('(% 2.6f, % 3.6f)' %
             (world.gnss_sensor.lat,
              world.gnss_sensor.lon)),
            'Height:  % 18.0f m' %
            t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

            def distance(l):
                return math.sqrt((l.x - t.location.x) ** 2 +
                                 (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)

            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player1.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (
            0.5 *
            width -
            0.5 *
            self.dim[0],
            0.5 *
            height -
            0.5 *
            self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(
                weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(
                weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(
                carla.Location(
                    x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(
                weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------
# ============set_velocity==================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(
                carla.Location(
                    x=-5.5,
                    z=2.8),
                carla.Rotation(
                    pitch=-15)),
            carla.Transform(
                carla.Location(
                    x=1.6,
                    z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.sensor.set_transform(
            self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(
                    weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification(
            'Recording %s' %
            ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()  # intial pygame
    pygame.font.init()  # initialize the font module
    # DDPG_PARAMETERS
    train = True
    s_dim = 12
    a_dim = 3
    var = 1  # control exploration
    memory_capcity = 10000  # memory size
    max_episodes = 5000
    render = True  # display
    # render = False  # display
    GLOBAL_RUNNING_R = []
    ddpg = DDPG.DDPG(a_dim, s_dim, train=train)
    for i in range(max_episodes):  # 一共运行 maxepisode 次
        client = carla.Client('127.0.0.1', 2000)  # host address and port
        world = client.load_world('Town01')  # load world of "Town01"
        client.set_timeout(4.0)  # 超时时间
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        if i % 30 == 1:
            ddpg.save_net()
            plot_record.plt_reward(
                GLOBAL_RUNNING_R=GLOBAL_RUNNING_R,
                title='carla')
        world = World(client.get_world(), hud, 'vehicle.bmw.grandtourer')
        if world.player1 is None:
            print("enter this continue player1 >>>>>>>>>>>>>>>>>>>>>>")
            continue
        if world.vehicle2 is None:
            print("enter this continue vehicle2 >>>>>>>>>>>>>>>>>>>>>>")
            if world.player1 is not None:
                world.player1.destroy()
            continue
        controller = RLControl(world)
        clock = pygame.time.Clock()
        s, dist = world.get_state()
        flag = False
        ep_reward = 0
        print("veh start")
        while True:  # 每次的内部循环
            # as soon as the server is ready continue!
            world.world.wait_for_tick(10.0)
            world.tick(clock)
            if render:
                world.render(display)  # 渲染
            pygame.display.flip()
            t1 = time.time()
            print(dist)
            if dist < 15:
                flag = True
            if flag is not True:
                a = np.array([0, 0.5, 0])
                control, s_, r, done, dist = controller.step(world, action=a)
            else:
                a = ddpg.choose_action(s)
                a = np.clip(np.random.normal(a, var), -1, 1)  # Add exploration noise
                # add randomness to action selection for exploration
                control, s_, r, done, dist = controller.step(world, action=a)
                ddpg.store_transition(s, a, r / 10, s_)
                if ddpg.memory_counter > memory_capcity:
                    var *= .999999  # decay the action randomness
                    ddpg.learn()
                ep_reward += r
            s = s_
            world.player1.apply_control(control)
            # print(7)
            if control.collisionFlag:
                # print("collision!")
                GLOBAL_RUNNING_R.append(ep_reward)
                print(i)

                break
            if world.player1.get_location().x < 95:

                # print("sucess!")
                print(i)
                GLOBAL_RUNNING_R.append(ep_reward)

                print(
                    'Episode:',
                    i,
                    ' Reward: %i' %
                    int(ep_reward),
                    'Explore: %.2f' %
                    var,
                )

                break
                # if world.player1.get_velocity() == 0:
                #     break
        if world is not None:
            world.destroy()
    ddpg.save_net()
    print('Running time: ', time.time() - t1)
    pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
# ==============================================================================
def distance(x, y):
    return np.sqrt(
        np.square(
            x[0] -
            y[0]) +
        np.square(
            x[1] -
            y[1]) +
        np.square(
            x[2] -
            y[2]))


def reward_function(state1, state2, dest, collision_flag):

    if collision_flag:
        r3 = -100
    else:
        r3 = 1
    # r4 = # time
    r = r3
    return r


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='500x500',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.bmw.grandtourer',  # here to set vehicle type
        help='actor filter (default: "vehicle.bmw.grandtourer")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    #  ./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15 -windowed -ResX=300 -ResY=300
    main()

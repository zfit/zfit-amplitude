#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   dalitz.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.03.2019
# =============================================================================
"""Utilities to handle Dalitz decays."""

from collections import OrderedDict

from particle import Particle

import tensorflow as tf

import zfit

from zfit_amplitude.amplitude import Decay, Amplitude
import zfit_amplitude.kinematics as kinematics


class DalitzParticle(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dalitz_name = str(self)


def get_mass_var_name(part1, part2):
    """Get Dalitz mass var name."""
    return f"m2{part1.dalitz_name}{part2.name}"


class ThreeBodyDalitz(Decay):
    """Helper class to configure 3-body Dalitz decays."""
    def __init__(self, top_particle, final_state_particles):
        if len(final_state_particles) != 3:
            raise ValueError("You need three final state particles!")
        self._top = DalitzParticle.from_string(top_particle)
        parts = p1, p2, p3 = [DalitzParticle.from_string(part)
                              for part in final_state_particles]
        # Check names and add suffixes if necessary
        for part_num, part_name in enumerate(final_state_particles):
            part_obj = parts[part_num]
            name_counts = final_state_particles[:part_num].count(part_name)
            if name_counts > 0:
                part_obj.dalitz_name = f"{part_obj.dalitz_name}{name_counts-1}"
        self._parts = OrderedDict((part.dalitz_name, part) for part in parts)
        # Build observables
        obs1 = zfit.Space(obs=get_mass_var_name(p1, p2),
                          limits=((p1.mass + p2.mass)**2,
                                  (self._top.mass - p3.mass)**2))
        obs2 = zfit.Space(obs=get_mass_var_name(p1, p3),
                          limits=((p1.mass + p3.mass)**2,
                                  (self._top.mass - p2.mass)**2))
        obs3 = zfit.Space(obs=get_mass_var_name(p2, p3),
                          limits=((p2.mass + p3.mass)**2,
                                  (self._top.mass - p1.mass)**2))
        obs = obs1 * obs2 * obs3
        obs_config = {obs1: (p1.name, p2.name),
                      obs2: (p1.name, p3.name),
                      obs3: (p2.name, p3.name)}

        def three_body_var_transformation(particles):
            return {obs.obs: kinematics.mass_squared(particles[part1]+particles[part2])
                    for obs, (part1, part2) in obs_config.items()}

        super().__init__(obs=obs, variable_transformations=three_body_var_transformation)


    def add_amplitude(self, resonance, resonance_children, amplitude_func, coeff, resonance_mass=None):
        # Sort resonance children according to the self._parts order
        sorted_parts = list(self._parts.keys())
        resonance_children = sorted(resonance_children,
                                    key=lambda x: sorted_parts.index(x))

        def get_res_mass(mass, width, name):
            """Calculate resonance mass."""
            def get_resonance_mass(mass_min, mass_max, n_events):
                bw = dynamics.RelativisticBreitWigner(obs=zfit.Space(f'M({name})',
                                                                     mass_min, mass_max),
                                                      name=f'BW({name})',
                                                      mres=mass, wres=width).sample(n_events)
                return tf.reshape(bw, (1, n_events))
            return get_resonance_mass

        if not resonance_mass:
            resonance_mass = get_res_mass

        non_resonant_part = [part for part_name, part in self._parts.items()
                             if part_name not in resonance_children][0]
        resonance = DalitzParticle.from_string(resonance)
        decay_tree = (self._top, [
            (resonance, [(self._parts[part_name], [])
                         for part_name in resonance_children]),
            (non_resonant_part, [])])

        return super().add_amplitude(ThreeBodyAmplitude(decay_tree,
                                                        amplitude_func,
                                                        resonance_mass),
                                     coeff)


class ThreeBodyAmplitude(Amplitude):
    def __init__(self, decay_tree, amplitude_func, resonance_mass):

        top = decay_tree[0]
        children = decay_tree[1]
        resonance, resonance_children = [child for child in children if child[1]][0]
        resonance_children = [child[0] for child in resonance_children]
        non_resonant_part = [child[0] for child in children if not child[1]][0]

        res_mass = resonance_mass(resonance.mass, resonance.width, resonance.dalitz_name)
        decay_tree = (top.name, top.mass, [
            (resonance.dalitz_name, res_mass, [(part.dalitz_name, part.mass, [])
                                               for part in resonance_children]),
            (non_resonant_part.dalitz_name, non_resonant_part.mass, [])])

        self.resonance = resonance
        self.particles = {part.dalitz_name: part
                          for part in resonance_children + [non_resonant_part]}
        self.mass_var = get_mass_var_name(*resonance_children)
        self.amplitude_func = amplitude_func
        super().__init__(decay_tree)

    def _amplitude(self, obs):
        mass_obs = obs.get_subspace(self.mass_var)
        return self.amplitude_func(mass_obs, self.resonance, self.particles)


# EOF


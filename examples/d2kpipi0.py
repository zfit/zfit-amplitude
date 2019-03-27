#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   d2kpipi0.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   26.03.2019
# =============================================================================
"""Implementation of the D0 -> K+ pi- pi0 Dalitz from PRL 103 (2009) 211801."""

from math import radians

import tensorflow as tf

import zfit

from particle.particle import literals as lp
from particle import Particle

from zfit_amplitude.amplitude import Decay, Amplitude
import zfit_amplitude.dynamics as dynamics


# pylint: disable=E1101
PI_MINUS = lp.pi_minus
PI_ZERO = lp.pi_0
K_PLUS = lp.K_plus
D_ZERO = lp.D_0


RESONANCES = {'rho(770)': ('m2pipi', lp.rho_770_plus),
              'K2*(1430)0': ('m2kpim', Particle.from_string('K(2)*(1430)')),
              'K0*(1430)+': ('m2kpiz', lp.K_0st_1430_plus),
              'K*(892)+': ('m2kpiz', lp.Kst_892_plus),
              'K0*(1430)0': ('m2kpim', lp.K_0st_1430_0),
              'K*(892)0': ('m2kpim', lp.Kst_892_0)}

COEFFS = {'rho(770)': zfit.ComplexParameter.from_polar('f(rho(770))', 1.0, 0.0, floating=False),
          'K2*(1430)0': zfit.ComplexParameter.from_polar('f(K2star(1430)0)', 0.088, radians(-17.2)),
          'K0*(1430)+': zfit.ComplexParameter.from_polar('f(K0star(1430)plus)', 6.78, radians(69.1)),
          'K*(892)+': zfit.ComplexParameter.from_polar('f(Kstar(892)plus)', 0.899, radians(-171)),
          'K0*(1430)0': zfit.ComplexParameter.from_polar('f(K0star(1430)0)', 1.65 , radians(-44.4)),
          'K*(892)0': zfit.ComplexParameter.from_polar('f(Kstar(892)0)', 0.398, radians(24.1))}


def resonance_mass(mass, width, name):
    def get_resonance_mass(mass_min, mass_max, n_events):
        bw = dynamics.RelativisticBreitWigner(obs=zfit.Space(f'M({name})',
                                                             mass_min, mass_max),
                                              name=f'BW({name})',
                                              mres=mass, wres=width).sample(n_events)
        return tf.reshape(bw, (1, n_events))
    return get_resonance_mass


class D2Kpipi0Amplitude(Amplitude):
    def __init__(self, resonance):
        self.mass_var, self.resonance = RESONANCES[resonance]
        res_mass = resonance_mass(self.resonance.mass, self.resonance.width, self.resonance.name)
        if self.mass_var == 'm2pipi':
            decay_tree = ('D0', D_ZERO.mass, [
                (self.resonance.name, res_mass, [
                    ('pi-', PI_MINUS.mass, []), ('pi0', PI_ZERO.mass, [])]),
                ('K+', K_PLUS.mass, [])])
        elif self.mass_var == 'm2kpi-':
            decay_tree = ('D0', D_ZERO.mass, [
                (self.resonance.name, res_mass, [
                    ('K-', K_PLUS.mass, []),
                    ('pi0', PI_ZERO.mass, [])]),
                ('pi-', PI_MINUS.mass, [])])
        else:
            decay_tree = ('D0', D_ZERO.mass, [
                (self.resonance.name, res_mass, [
                    ('K-', K_PLUS.mass, []),
                    ('pi-', PI_MINUS.mass, [])]),
                ('pi0', PI_ZERO.mass, [])])
        super().__init__(decay_tree)

    def _amplitude(self, obs):
        mass_obs = obs.get_subspace(self.mass_var)
        return dynamics.RelativisticBreitWigner(obs=mass_obs,
                                                name=self.resonance.name,
                                                mres=self.resonance.mass,
                                                wres=self.resonance.width,
                                                using_m_squared=True)


if __name__ == "__main__":
    print("This does currently not run")

    m2kpim = zfit.Space(obs='m2kpi-',
                        limits=((K_PLUS.mass + PI_MINUS.mass)**2, (D_ZERO.mass - PI_ZERO.mass)**2))
    m2kpiz = zfit.Space(obs='m2kpi0',
                        limits=((K_PLUS.mass + PI_ZERO.mass)**2, (D_ZERO.mass - PI_MINUS.mass)**2))
    m2kpipi = zfit.Space(obs='m2pipi',
                         limits=((PI_ZERO.mass + PI_MINUS.mass)**2, (D_ZERO.mass - K_PLUS.mass)**2))

    obs = m2kpim * m2kpiz * m2kpipi

    D2Kpipi0 = Decay(obs)
    for res in RESONANCES:
        D2Kpipi0.add_amplitude(D2Kpipi0Amplitude(res),
                               COEFFS[res])
    pdf = D2Kpipi0.pdf("D2Kpipi0")
    for dep in pdf.get_dependents(only_floating=False):
        print("{} {} Floating: {}".format(dep.name, zfit.run(dep), dep.floating))

# EOF

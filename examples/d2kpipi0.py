#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   d2kpipi0.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   26.03.2019
# =============================================================================
"""Implementation of the D0 -> K+ pi- pi0 Dalitz from PRL 103 (2009) 211801."""

from math import radians

from particle.particle import literals as lp
from particle import Particle

import zfit

from zfit_amplitude.amplitude import Decay, Amplitude, Resonance
import zfit_amplitude.dynamics as dynamics
import zfit_amplitude.kinematics as kinematics


polar_param = zfit.ComplexParameter.from_polar

# pylint: disable=E1101
PI_MINUS = lp.pi_minus
PI_ZERO = lp.pi_0
K_PLUS = lp.K_plus
D_ZERO = lp.D_0


RESONANCES = {'rho(770)': ('m2pipi', Resonance(lp.rho_770_plus,
                                               dynamics.RelativisticBreitWigner)),
              'K2*(1430)0': ('m2kpim', Resonance(Particle.from_string('K(2)*(1430)'),
                                                 dynamics.RelativisticBreitWigner)),
              'K0*(1430)+': ('m2kpi0', Resonance(lp.K_0st_1430_plus,
                                                 dynamics.RelativisticBreitWigner)),
              'K*(892)+': ('m2kpi0', Resonance(lp.Kst_892_plus,
                                               dynamics.RelativisticBreitWigner)),
              'K0*(1430)0': ('m2kpim', Resonance(lp.K_0st_1430_0,
                                                 dynamics.RelativisticBreitWigner)),
              'K*(892)0': ('m2kpim', Resonance(lp.Kst_892_0,
                                               dynamics.RelativisticBreitWigner))}

COEFFS = {'rho(770)': polar_param('f_rho770', 1.0, 0.0, floating=False),
          'K2*(1430)0': polar_param('f_K2star1430_0', 0.088, radians(-17.2)),
          'K0*(1430)+': polar_param('f_K0star1430_plus', 6.78, radians(69.1)),
          'K*(892)+': polar_param('f_Kstar892_plus', 0.899, radians(-171)),
          'K0*(1430)0': polar_param('f_K0star1430_0', 1.65, radians(-44.4)),
          'K*(892)0': polar_param('f_Kstar892_0', 0.398, radians(24.1))}


class D2Kpipi0Amplitude(Amplitude):
    """D -> K+pi-pi0 amplitude.

    Chooses the correct observable to use and implements a BW to represent the given
    resonance.

    Arguments:
        resonance (str): Name of the intermediate resonance. It is used to determine
        the structure of the decay.

    """

    def __init__(self, resonance, resonance_model):  # noqa
        self.mass_var, self.resonance = RESONANCES[resonance]
        res_mass = self.resonance.mass_sampler(self.resonance.name,
                                               mres=self.resonance.mass, wres=self.resonance.width,
                                               using_m_squared=False)
        # res_mass = self.resonance(self.resonance.mass, self.resonance.width, self.resonance.name)
        if self.mass_var == 'm2pipi':
            decay_tree = ('D0', D_ZERO.mass, [
                (self.resonance.name, res_mass, [
                    ('pi-', PI_MINUS.mass, []), ('pi0', PI_ZERO.mass, [])]),
                ('K+', K_PLUS.mass, [])])
        elif self.mass_var == 'm2kpim':
            decay_tree = ('D0', D_ZERO.mass, [
                (self.resonance.name, res_mass, [
                    ('K+', K_PLUS.mass, []),
                    ('pi0', PI_ZERO.mass, [])]),
                ('pi-', PI_MINUS.mass, [])])
        else:
            decay_tree = ('D0', D_ZERO.mass, [
                (self.resonance.name, res_mass, [
                    ('K+', K_PLUS.mass, []),
                    ('pi-', PI_MINUS.mass, [])]),
                ('pi0', PI_ZERO.mass, [])])
        super().__init__(decay_tree)

    def _amplitude(self, obs):
        mass_obs = obs.get_subspace(self.mass_var)
        return self.resonance.amplitude(self.resonance.name, mass_obs,
                                        mres=self.resonance.mass, wres=self.resonance.width, using_m_squared=True)


def var_transformation(particles):
    """Transform particles to our observables of interest."""
    return {'m2kpim': kinematics.mass_squared(particles['K+']+particles['pi-']),
            'm2kpi0': kinematics.mass_squared(particles['K+']+particles['pi0']),
            'm2pipi': kinematics.mass_squared(particles['pi-']+particles['pi-'])}


if __name__ == "__main__":
    m2kpim = zfit.Space(obs='m2kpim',
                        limits=((K_PLUS.mass + PI_MINUS.mass)**2, (D_ZERO.mass - PI_ZERO.mass)**2))
    m2kpiz = zfit.Space(obs='m2kpi0',
                        limits=((K_PLUS.mass + PI_ZERO.mass)**2, (D_ZERO.mass - PI_MINUS.mass)**2))
    m2kpipi = zfit.Space(obs='m2pipi',
                         limits=((PI_ZERO.mass + PI_MINUS.mass)**2, (D_ZERO.mass - K_PLUS.mass)**2))

    masses = m2kpim * m2kpiz * m2kpipi

    D2Kpipi0 = Decay(masses, variable_transformations=var_transformation)
    for res, (_, res_model) in RESONANCES.items():
        D2Kpipi0.add_amplitude(D2Kpipi0Amplitude(res, res_model),
                               COEFFS[res])
    pdf = D2Kpipi0.pdf("D2Kpipi0")
    for dep in pdf.get_dependents(only_floating=False):
        print("{} {} Floating: {}".format(dep.name, zfit.run(dep), dep.floating))

    pdf.sample(10)

# EOF

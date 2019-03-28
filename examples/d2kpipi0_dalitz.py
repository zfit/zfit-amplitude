#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   d2kpipi0_dalitz.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   28.03.2019
# =============================================================================
"""Implementation of the D0 -> K+ pi- pi0 Dalitz from PRL 103 (2009) 211801.

This implementation uses zfit_amplitude.dalitz.

"""

from math import radians

import zfit

from zfit_amplitude.dalitz import ThreeBodyDalitz
import zfit_amplitude.dynamics as dynamics


polar_param = zfit.ComplexParameter.from_polar


def bw_amplitude(mass_obs, resonance, _):
    """Calculate the resonance mass as a BW."""
    return dynamics.RelativisticBreitWigner(obs=mass_obs,
                                            name=resonance.name,
                                            mres=resonance.mass,
                                            wres=resonance.width,
                                            using_m_squared=True)


RESONANCES = [('rho(770)', ('pi-', 'pi0'), bw_amplitude),
              ('K(2)*(1430)0', ('K+', 'pi-'), bw_amplitude),
              ('K(0)*(1430)+', ('K+', 'pi0'), bw_amplitude),
              ('K*(892)+', ('K+', 'pi0'), bw_amplitude),
              ('K(0)*(1430)0', ('K+', 'pi-'), bw_amplitude),
              ('K*(892)0', ('K+', 'pi-'), bw_amplitude)]

COEFFS = {'rho(770)': polar_param('f(rho(770))', 1.0, 0.0, floating=False),
          'K(2)*(1430)0': polar_param('f(K2star(1430)0)', 0.088, radians(-17.2)),
          'K(0)*(1430)+': polar_param('f(K0star(1430)plus)', 6.78, radians(69.1)),
          'K*(892)+': polar_param('f(Kstar(892)plus)', 0.899, radians(-171)),
          'K(0)*(1430)0': polar_param('f(K0star(1430)0)', 1.65, radians(-44.4)),
          'K*(892)0': polar_param('f(Kstar(892)0)', 0.398, radians(24.1))}


if __name__ == "__main__":
    D2Kpipi0 = ThreeBodyDalitz('D0', ['K+', 'pi-', 'pi0'])
    for res, children, amp in RESONANCES:
        D2Kpipi0.add_amplitude(res, children, amp,
                               COEFFS[res])
    pdf = D2Kpipi0.pdf("D2Kpipi0")
    for dep in pdf.get_dependents(only_floating=False):
        print("{} {} Floating: {}".format(dep.name, zfit.run(dep), dep.floating))

# EOF

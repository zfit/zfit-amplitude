#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   dynamics.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   12.02.2019
# =============================================================================
"""Temporary functions to play with."""

import tensorflow as tf

import zfit
from numpy.core._multiarray_umath import dtype
from zfit import ztf

import zfit_amplitude.kinematics as kinematics


def relativistic_breit_wigner(m2, mres, wres):
    """
    Relativistic Breit-Wigner
    """
    # TODO: Check complex
    second_part = tf.complex(ztf.constant(0.), mres) * ztf.to_complex(wres)
    below_div = ztf.to_complex(mres ** 2 - m2) - second_part
    # real_part = tf.real(below_div)
    # imag_part = tf.imag(below_div)
    # result = tf.complex(1. / real_part, 1. / imag_part)
    # return result
    # return tf.cast(1. / tf.cast(below_div, dtype=tf.complex64), dtype=tf.complex128)
    return 1. / below_div


# class FancyAlbert(zfit.func.ZFunc):
#     def _func(self, x):
#         obs = ztf.unstack_x(x)
#         return self.FUNC(*obs, **self.params)


# class relativistic_breit_wigner(FancyAlbert):
#     PARAMS = ['mres', 'wres']
#     FUNC = relativistic_breit_wigner


# def make_func(func, params):
#     function = FancyAlbert
#     function.PARAMS = params
#     function.FUNC = func
#     return function

# relativistic_breit_wigner = make_func(func=relativistic_breit_wigner, params=['mres', 'wres'])

class RelativisticBreitWigner(zfit.func.BaseFunc):
    def __init__(self, obs, name, mres, wres, using_m_squared=False, dtype=zfit.settings.ztypes.complex):
        self.using_m_squared = using_m_squared
        # HACK to make it usable in while loop
        zfit.run._enable_parameter_autoconversion = False

        super().__init__(obs=obs, name=name, dtype=dtype,
                         params={'mres': mres, 'wres': wres})

        zfit.run._enable_parameter_autoconversion = True
        # HACK end

    def _func(self, x):
        var = ztf.unstack_x(x)
        if isinstance(var, list):
            m_sq = kinematics.mass_squared(tf.reduce_sum(
                [kinematics.lorentz_vector(kinematics.vector(px, py, pz), pe)
                 for px, py, pz, pe in zip(*[iter(var)] * 4)],
                axis=0))
        elif self.using_m_squared:
            m_sq = var
        else:
            m_sq = var * tf.math.conj(var)
        mres = self.params['mres']
        wres = self.params['wres']
        return relativistic_breit_wigner(m_sq, mres, wres)


class RelativisticBreitWignerReal(RelativisticBreitWigner):


    def __init__(self, obs, name, mres, wres, using_m_squared=False):
        super().__init__(obs=obs, name=name, mres=mres, wres=wres, using_m_squared=using_m_squared, dtype=zfit.settings.ztypes.float)

    def _func(self, x):
        propagator = super()._func(x)
        val = propagator * tf.math.conj(propagator)
        val = ztf.to_real(val)
        return val
        #return ztf.to_real(super()._func(x))


def blatt_weisskopf_ff(q, q0, d, l):
    """
    Blatt-Weisskopf formfactor for intermediate resonance
    """
    z = q * d
    z0 = q0 * d

    def hankel1(x):
        if l == 0:
            return ztf.constant(1.)
        if l == 1:
            return 1 + x ** 2
        if l == 2:
            x2 = x ** 2
            return 9 + x2 * (3. + x2)
        if l == 3:
            x2 = x ** 2
            return 225 + x2 * (45 + x2 * (6 + x2))
        if l == 4:
            x2 = x ** 2
            return 11025. + x2 * (1575. + x2 * (135. + x2 * (10. + x2)))

    return tf.sqrt(hankel1(z0) / hankel1(z))


def mass_dependent_width(m, m0, gamma0, p, p0, ff, l):
    """
    Mass-dependent width for BW amplitude
    """
    return gamma0 * ((p / p0) ** (2 * l + 1)) * (m0 / m) * (ff ** 2)


def orbital_barrier_factor(p, p0, l):
    """
    Orbital barrier factor
    """
    return (p / p0) ** l


def breit_wigner_line_shape(m2, m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld, barrier_factor=True):
    """
    Breit-Wigner amplitude with Blatt-Weisskopf formfactors, mass-dependent width and orbital
    barriers
    """
    m = tf.sqrt(m2)
    q = kinematics.two_body_momentum(md, m, mc)
    q0 = kinematics.two_body_momentum(md, m0, mc)
    p = kinematics.two_body_momentum(m, ma, mb)
    p0 = kinematics.two_body_momentum(m0, ma, mb)
    ffr = blatt_weisskopf_ff(p, p0, dr, lr)
    ffd = blatt_weisskopf_ff(q, q0, dd, ld)
    width = mass_dependent_width(m, m0, gamma0, p, p0, ffr, lr)
    bw = relativistic_breit_wigner(m2, m0, width)
    ff = ffr * ffd
    if barrier_factor:
        b1 = orbital_barrier_factor(p, p0, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        ff *= b1 * b2
    return bw * tf.complex(ff, ztf.constant(0.))


class BreitWignerLineshape(zfit.func.BaseFunc):
    """Func version of `breit_wigner_line_shape`."""

    def __init__(self, obs, name,
                 m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld, ma0, md0, barrier_factor, using_m_squared=False):
        self.barrier_factor = barrier_factor
        self.using_m_squared = using_m_squared
        super().__init__(obs=obs, name=name, dtype=zfit.settings.ztypes.complex,
                         params={'m0': m0, 'gamma0': gamma0, 'ma': ma, 'mb': mb, 'mc': mc, 'md': md,
                                 'dr': dr, 'dd': dd, 'lr': lr, 'ld': ld, 'ma0': ma0, 'md0': md0})

    def _func(self, x):
        def func(x):
            var = ztf.unstack_x(x)
            if isinstance(var, list):
                m_sq = kinematics.mass(tf.reduce_sum(
                    [kinematics.lorentz_vector(kinematics.vector(px, py, pz), pe)
                     for px, py, pz, pe in zip(*[iter(var)] * 4)],
                    axis=0))
            elif self.using_m_squared:
                m_sq = var
            else:
                m_sq = tf.pow(var, 2)
            m0 = self.params['m0']
            gamma0 = self.params['gamma0']
            ma = self.params['ma']
            mb = self.params['mb']
            mc = self.params['mc']
            md = self.params['md']
            dr = self.params['dr']
            dd = self.params['dd']
            lr = self.params['lr']
            ld = self.params['ld']
            ma0 = self.params['ma0']
            md0 = self.params['md0']
            return breit_wigner_line_shape(m_sq, m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld,
                                           self.barrier_factor, ma0, md0)

        return func(x)
        # return ztf.run_no_nan(func=func, x=x)


def spin_factor_oneplus_swave(pol0, p1, p2, p3, mresa, mresv):
    """Spin factor for decay chain BtoAV0_AtoVP1_VtoP2P3_Swave.

    p1 - p3 are 4-vectors for the final state hadrons
    pol0 is the photon polarisation vector (complex lorentz vector)
    mResA is the invariant mass of the axial-vector resonance (p1+p2+p3)
    mResV is the invariant mass of the scalar resonance (p2+p3)

    """
    pv = p2 + p3
    qv = p2 - p3
    pa = pv + p1
    zt = qv - tf.expand_dims(kinematics.lorentz_dot_product(qv, pv), axis=-1) * pv * (1. / mresv ** 2)
    spinsumv = (-zt + pa * tf.expand_dims(kinematics.lorentz_dot_product(pa, zt), axis=-1) / mresa ** 2)

    return tf.complex(kinematics.lorentz_dot_product(tf.real(tf.conj(pol0)), spinsumv),
                      kinematics.lorentz_dot_product(tf.imag(tf.conj(pol0)), spinsumv))


class SpinFactor(zfit.func.BaseFunc):
    """Order of observables: p1, p2, p3, pgamma"""

    SPIN_FACTORS = {("1+", "S"): spin_factor_oneplus_swave}

    def __init__(self, obs, spin, wave, helicity, mresa, mresv, name=None):
        self.mresa = mresa
        self.mresv = mresv
        self.helicity = helicity
        if not name:
            name = "SpinFactor_{}_{}_{}".format(spin.replace('+', 'plus').replace('-', 'minus'), wave, helicity)
        self._spin_factor_function = SpinFactor.SPIN_FACTORS[(spin, wave)]
        super().__init__(obs=obs, name=name, dtype=zfit.settings.ztypes.complex, params={})

    def _func(self, x):
        components = ztf.unstack_x(x)
        p1, p2, p3, pgamma = [kinematics.lorentz_vector(kinematics.vector(px, py, pz), pe)
                              for px, py, pz, pe in zip(*[iter(components)] * 4)]
        return self._spin_factor_function(kinematics.pol_vector(pgamma, self.helicity), p1, p2, p3,
                                          mresa=self.mresa, mresv=self.mresv)

# EOF

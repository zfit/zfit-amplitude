in/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   kinematics.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   12.02.2019
# =============================================================================
"""Basic kinematics."""

import tensorflow as tf

import zfit
from zfit import ztf


def zemach_tensor(m2ab, m2ac, m2bc, m2d, m2a, m2b, m2c, spin):
    """
    Zemach tensor for 3-body D->ABC decay
    """
    z = None
    if spin == 0:
        z = tf.complex(ztf.constant(1.), ztf.constant(0.))
    if spin == 1:
        z = tf.complex(m2ac - m2bc + (m2d - m2c) * (m2b - m2a) / m2ab, ztf.constant(0.))
    if spin == 2:
        z = tf.complex((m2bc - m2ac + (m2d - m2c) * (m2a - m2b) / m2ab) ** 2 - 1. / 3. * (
                m2ab - 2. * (m2d + m2c) + (m2d - m2c) ** 2 / m2ab) * (
                               m2ab - 2. * (m2a + m2b) + (m2a - m2b) ** 2 / m2ab), ztf.constant(0.))
    return z


def two_body_momentum(md, ma, mb, calc_complex_=False):
    """Momentum of two-body decay products D->AB in the D rest frame.
    Args:
        calc_complex (bool): If true, the output value is a complex number,
        allowing analytic continuation for the region below threshold.
    """
    squared_sum = (md ** 2 - (ma + mb) ** 2) * (md ** 2 - (ma - mb) ** 2) / (4 * md ** 2)
    if calc_complex_:
        squared_sum = ztf.to_complex(squared_sum)
    return tf.sqrt(squared_sum)


def spatial_component(vector):
    """
    Return spatial components of the input Lorentz vector
        vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    """
    #  return tf.slice(vector, [0, 0], [-1, 3])
    return vector[:, 0:3]


def time_component(vector):
    """
    Return time component of the input Lorentz vector
        vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    """
    #  return tf.unstack(vector, axis=1)[3]
    return vector[:, 3]


def x_component(vector):
    """
    Return spatial X component of the input Lorentz or 3-vector
        vector : input vector
    """
    #  return tf.unstack(vector, axis=1)[0]
    return vector[:, 0]


def y_component(vector):
    """
    Return spatial Y component of the input Lorentz or 3-vector
        vector : input vector
    """
    return vector[:, 1]


def z_component(vector):
    """
    Return spatial Z component of the input Lorentz or 3-vector
        vector : input vector
    """
    return vector[:, 2]


def vector(x, y, z):
    """
    Make a 3-vector from components
    x, y, z : vector components
    """
    return tf.stack([x, y, z], axis=1)


def scalar_product(vec1, vec2):
    """
    Calculate scalar product of two 3-vectors
    """
    return tf.reduce_sum(vec1 * vec2, 1)


def scalar(x):
    """
    Create a scalar (e.g. tensor with only one component) which can be used to e.g. scale a vector
    One cannot do e.g. Const(2.)*vector(x, y, z), needs to do scalar(Const(2))*vector(x, y, z)
    """
    return tf.stack([x], axis=1)


def mass_squared(vector):
    """Calculate the squared mass for a Lorentz 4-momentum."""
    return tf.reduce_sum(tf.transpose(vector * vector) * tf.reshape(metric_tensor(), (4,1)), axis=0)


def mass(vector):
    """
    Calculate mass scalar for Lorentz 4-momentum
        vector : input Lorentz momentum vector
    """
    return tf.sqrt(mass_squared(vector))
    # return tf.sqrt(tf.reduce_sum(vector * vector * metric_tensor(), axis=1))


def lorentz_vector(space, time):
    """
    Make a Lorentz vector from spatial and time components
        space : 3-vector of spatial components
        time  : time component
    """
    return tf.concat([space, tf.stack([time], axis=1)], axis=1)


def lorentz_boost(vector, boostvector):
    """
    Perform Lorentz boost
        vector :     4-vector to be boosted
        boostvector: boost vector. Can be either 3-vector or 4-vector (only spatial components
        are used)
    """
    boost = spatial_component(boostvector)
    b2 = scalar_product(boost, boost)

    def boost_fn():
        gamma = 1. / tf.sqrt(1. - b2)
        gamma2 = (gamma - 1.0) / b2
        ve = time_component(vector)
        vp = spatial_component(vector)
        bp = scalar_product(vp, boost)
        vp2 = vp + scalar(gamma2 * bp + gamma * ve) * boost
        ve2 = gamma * (ve + bp)
        return lorentz_vector(vp2, ve2)

    def no_boost_fn():
        return vector

    # if boost vector is zero, return the original vector
    boosted_vector = tf.cond(tf.equal(b2, 0.), true_fn=no_boost_fn, false_fn=boost_fn)
    return boosted_vector


def metric_tensor():
    """
    Metric tensor for Lorentz space (constant)
    """
    return ztf.constant([-1., -1., -1., 1.], dtype=zfit.settings.ztypes.float)


def lorentz_dot_product(vec1, vec2):
    """
    Dot product of two lorentz vectors
    return tf.tensordot(vec1,vec2,)?
    """
    return tf.reduce_sum(vec1 * vec2 * metric_tensor(), axis=-1)


def pol_vector(p, hel):
    """
    Polarisation vector (complex lorentz vector) for a massless spin-1 particle with four-vector p and helicity hel
    Non-massless case to be added?
    """

    px = p[:, 0] / p[:, 3]
    py = p[:, 1] / p[:, 3]
    pz = p[:, 2] / p[:, 3]
    r2 = px ** 2 + py ** 2
    s = tf.math.sin(tf.math.acos(pz))
    one_over_sqrt_two = ztf.to_complex((1 / tf.sqrt(ztf.constant(2.))))

    if hel == 1:
        c0 = one_over_sqrt_two * ztf.complex(((1 - pz) * py ** 2) / r2 + pz,
                                             px * py * (1 - pz) / r2)
        c1 = one_over_sqrt_two * ztf.complex(-px * py * (1 - pz) / r2,
                                             -((1 - pz) * px ** 2) / r2 - pz)
        c2 = one_over_sqrt_two * ztf.complex(-px * s / r2 ** 0.5,
                                             py * s / r2 ** 0.5)
        c3 = tf.zeros_like(c0, dtype=tf.complex128)
    if hel == -1:
        c0 = one_over_sqrt_two * ztf.complex(-(((1 - pz) * py ** 2) / r2 + pz),
                                             px * py * (1 - pz) / r2)
        c1 = one_over_sqrt_two * ztf.complex(px * py * (1 - pz) / r2,
                                             -((1 - pz) * px ** 2) / r2 - pz)
        c2 = one_over_sqrt_two * ztf.complex(px * s / r2 ** 0.5,
                                             py * s / r2 ** 0.5)
        c3 = tf.zeros_like(c0, dtype=tf.complex128)

    return lorentz_vector(vector(c0, c1, c2), c3)

# EOF

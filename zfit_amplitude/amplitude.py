#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   amplitude.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   26.03.2019
# =============================================================================
"""Base amplitude classes."""

from itertools import combinations

import tensorflow as tf
import zfit
from zfit import ztf
from zfit.core.interfaces import ZfitFunc

from zfit.models.functions import BaseFunctorFunc
from zfit.util.execution import SessionHolderMixin

from zfit_amplitude.utils import sanitize_string


class Decay:
    """Representation of a high-level decay, which can be made up of several amplitudes.

    Arguments:

    Raise:
        KeyError: If amplitude and coefficient list are not of the same length.
        ValueError: If all the amplitudes don't have the same top particle mass.

    """
    REQUIRED_ATTRIBUTES = tuple()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing_args = [attr for attr in Decay.REQUIRED_ATTRIBUTES
                        if not hasattr(cls, attr)]
        if missing_args:
            raise NotImplementedError("Decay attributes not implemented in subclass -> {}"
                                      .format(', '.join(missing_args)))

    def __init__(self, obs, amplitudes=None, coeffs=None):  # noqa: W107
        if amplitudes:
            if len(amplitudes) != len(coeffs):
                raise ValueError("Amplitude and coefficient lists must have the same length!")
            if len(set(amplitude.top_particle_mass for amplitude in amplitudes)) != 1:
                raise ValueError
        self._amplitudes = amplitudes if amplitudes else []
        self._coeffs = coeffs if coeffs else []
        self._obs = obs

    def add_amplitude(self, amplitude, coeff):
        """Add amplitude and its correspondig coefficient.

        Arguments:
            amplitude (:py:class:`Amplitude`): Amplitude to add.
            coeff (:py:class:~zfit.core.parameter.Parameter): Coefficient corresponding
                to the amplitude.

        Return:
            int: Total number of configured amplitudes.

        """
        self._amplitudes.append(amplitude)
        self._coeffs.append(coeff)
        return len(self._amplitudes)

    @property
    def obs(self):
        """:py:class:`~zfit.Space`: Observables."""
        return self._obs

    @property
    def amplitudes(self):
        """list[:py:class:`Amplitude`]: Amplitudes."""
        return self._amplitudes

    @property
    def coeffs(self):
        """list[:py:class:~zfit.core.parameter.Parameter]: Coefficients of each amplitude."""
        return self._coeffs

    def pdf(self, name):
        """Get the PDF representing the decay."""
        return self._pdf(name)

    def _pdf(self, name, external_integral=None):
        return SumAmplitudeSquaredPDF(obs=self.obs,
                                      name=name,
                                      amp_list=[amp.get_amplitude(self._obs)
                                                for amp in self._amplitudes],
                                      coef_list=self._coeffs,
                                      top_particle_mass=self._amplitudes[0].top_particle_mass,
                                      external_integral=external_integral)


class SumAmplitudeSquaredPDF(zfit.pdf.BasePDF):
    r"""Deal with squared sums of amplitudes in a smart way.

    This class takes amplitudes of the form

        M = sum_i a_i e^{i\phi_i} A_i

    where the A_i are amplitudes that don't change (ie, mass and width are fixed).
    The integrals of the cross A_i*A_j^* terms are cached to speed up calculations.

    Note:
        a_i e^{i\phi_i} are passed as `ComplexParameter`s.

    """

    def __init__(self, obs, amp_list, coef_list, top_particle_mass,
                 name="SumAmplitudeSquaredPDF", external_integral=None,
                 **kwargs):
        self._external_integral = external_integral
        self._amplitudes_combinations = [(frac1, frac2, AmplitudeProductCached(amp1=amp1, amp2=amp2))
                                         for (amp1, frac1), (amp2, frac2)
                                         in combinations(list(zip(amp_list, coef_list)), 2)]
        self._amplitudes = list(zip(coef_list, amp_list))
        self._particles = None
        self._top_at_rest = tf.stack((0.0, 0.0, 0.0, top_particle_mass), axis=-1)
        super().__init__(obs=obs, name=name, params={coef.name: coef for coef in coef_list}, **kwargs)
        self.update_integration_options(draws_per_dim=300000)

    def _unnormalized_pdf(self, x):
        def unnormalized_pdf_func(x):
            value = tf.reduce_sum([(frac1 * frac2.conj) * amps.func(x=x)
                                   for frac1, frac2, amps
                                   in self._amplitudes_combinations],
                                  axis=0)
            return ztf.to_real(value)

        result = ztf.run_no_nan(unnormalized_pdf_func, x)
        return result

    @zfit.supports()
    def _sample_and_weights(self, n_to_produce, limits, dtype):
        def flatten_gen(gen_list):
            if not self._particles:
                self._particles = gen_list.keys()
            return tf.concat([gen_list[part] for part in self._particles],
                             axis=1)

        pseudo_yields = []
        generators = []
        for frac, amp in self._amplitudes:
            pseudo_yields.append(frac * frac.conj * amp.integrate(limits=limits, norm_range=False))
            generators.append(amp.decay_phasespace())
        sum_yields = sum(pseudo_yields)
        n_to_generate = [tf.math.ceil(n_to_produce * pseudo_yield / sum_yields)
                         for pseudo_yield in pseudo_yields]

        norm_weights = []
        particles = []
        for amp_num, (_, amp) in enumerate(self._amplitudes):
            norm_weight, parts = generators[amp_num].generate(self._top_at_rest, n_to_generate[amp_num])
            norm_weights.append(norm_weight)
            particles.append(flatten_gen(parts))
        merged_particles = tf.concat(particles, axis=0)
        merged_weights = tf.concat(norm_weights, axis=0)
        thresholds = tf.random_uniform(shape=(n_to_produce,))
        return merged_particles, thresholds, merged_weights, sum_yields * 0.8, len(merged_weights)

    @zfit.supports()
    def _integrate(self, limits, norm_range):
        # raise NotImplementedError
        external_integral = self._external_integral
        if external_integral is not None:
            integral = self._external_integral(limits=limits, norm_range=norm_range)
        else:
            integral = tf.reduce_sum(
                [(frac1 * frac2.conj) * amps.integrate(limits=limits, norm_range=norm_range)
                 for frac1, frac2, amps in self._amplitudes_combinations],
                axis=0)
            integral = ztf.to_real(integral)
        return integral


class AmplitudeProductCached(BaseFunctorFunc, SessionHolderMixin):
    """Crude implementation of cached amplitude product.

    Note:
        This implementation currently assumes width and mean don't float, so the
        cache is never invalidated. This will be fixed soon.

    Arguments:
        amp1: :py:class:`~zfit.core.interfaces.ZfitFunc`: First amplitude.
        amp2: :py:class:`~zfit.core.interfaces.ZfitFunc`: Second amplitude.
        name (str, optional): Name of the object.

    """

    def __init__(self, amp1: ZfitFunc, amp2: ZfitFunc, name="AmplitudeProductCached", **kwargs):  # noqa: W107
        # from zfit.core.dimension import combine_spaces
        funcs = [amp1, amp2]
        # obs = combine_spaces([amp1.space, amp2.space])
        super().__init__(funcs=funcs, name=name, obs=None, **kwargs)

    def _func(self, x):
        amp1, amp2 = self.funcs
        def func(x):
            return amp1.func(x) * tf.math.conj(amp2.func(x))

        return ztf.run_no_nan(func=func, x=x)

    def _single_hook_integrate(self, limits, norm_range, name='_hook_integrate'):
        integral = self._cache.get("integral")
        if integral is None:
            self._cache['integral'] = {}

        # safer version
        if integral is not None:
            integral = integral.get((limits, norm_range))
        # safer version end

        if integral is None:
            integral = super()._single_hook_integrate(limits=limits, norm_range=norm_range, name=name)
            integral = self.sess.run(integral)
            integral_holder = tf.Variable(initial_value=integral, trainable=False,
                                          dtype=integral.dtype, use_resource=True)
            self.sess.run(integral_holder.initializer)
            # self._cache['integral'] = integral_holder
            # safer version
            self._cache['integral'][(limits, norm_range)] = integral_holder
            # safer version end
            integral = integral_holder
        return integral


class Amplitude:
    """Representation of a single amplitude.

    Arguments:
        decay_tree (tuple): Tree representation of the decay, represented by
        three items:
            + name
            + mass (fixed value or callable)
            + children, which are specified as a list of trees.

    Raise:
        KeyError: If the decay tree is badly specified.

    """

    def __init__(self, decay_tree):  # noqa: W107
        if len(decay_tree) != 3:
            raise KeyError("Badly specified decay tree")
        self._decay_tree = decay_tree
        self._top_mass = decay_tree[1]

    def __repr__(self):
        return "<Amplitude: {}>".format(self.get_decay_string())

    @property
    def top_particle_mass(self):
        """float: Mass of the top particle of the decay."""
        return self._top_mass

    def _decay_string(self):
        def get_children(tree):
            output = '->'
            for part_name, _, children in tree:
                if children:
                    output += '({}{})'.format(part_name, get_children(children))
                else:
                    output += part_name
            return output

        top_name, _, children = self._decay_tree
        return '{}{}'.format(top_name, get_children(children))

    def get_decay_string(self, sanitize=False):
        """Get a decay string.

        Arguments:
            sanitize (bool, optional): Sanitize to conform to tensorflow variable names.

        Return:
            str

        """
        decay_str = self._decay_string()
        if sanitize:
            decay_str = sanitize_string(decay_str)
        return decay_str

    def amplitude(self, obs, **kwargs):
        """Get the full decay amplitude.

        Arguments:
            obs: Observables.

        """
        return self._amplitude(obs, **kwargs)

    def _amplitude(self, obs, **kwargs):
        """Amplitude implementation.

        To be overridden.

        """
        raise NotImplementedError

    def decay_phasespace(self):
        """Get the phasespace generator."""
        return self._decay_phasespace()

    def _decay_phasespace(self):
        """Get the decay phasespace for sampling.

        The returned object must behave like a :py:class:`phasespace.Particle` object.

        """
        import phasespace as phsp

        def create_particles(tree):
            part_list = []
            for part_name, part_mass, part_tree in tree:
                part = phsp.Particle(part_name, mass=part_mass)
                if part_tree:
                    part.set_children(*create_particles(part_tree))
                part_list.append(part)
            return part_list

        top_name, _, top_tree = self._decay_tree
        return phsp.Particle(top_name).set_children(*create_particles(top_tree))

# EOF

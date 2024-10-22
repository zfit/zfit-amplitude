#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   amplitude.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   26.03.2019
# =============================================================================
"""Base amplitude classes."""

from types import MethodType

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
        ValueError: If the given amplitudes and coefficients are not well defined.
        KeyError: If the sampling and variable transformation functions lead to an inconsistent
        configuration.

    """

    def __init__(self, obs, amplitudes=None, coeffs=None, sampling_function=None,
                 variable_transformations=None):  # noqa: W107
        if amplitudes:
            if len(amplitudes) != len(coeffs):
                raise ValueError("Amplitude and coefficient lists must have the same length!")
            if len(set(amplitude.top_particle_mass for amplitude in amplitudes)) != 1:
                raise ValueError("The amplitudes have different top particle mass")
        self._amplitudes = amplitudes if amplitudes else []
        self._coeffs = coeffs if coeffs else []
        self._obs = obs
        # Do some checks for the sampling and variable transforms
        if not sampling_function:  # We do phasespace sampling
            if not variable_transformations:  # We need to have 4-momenta as obs!
                if len(obs.obs) < 8 or len(obs.obs) % 4:
                    raise KeyError(
                        "Requested sampling in phasespace but it doesn't seem like observables are 4-momenta")
        self._sampling_function = sampling_function
        self._var_transforms = variable_transformations

    def add_amplitude(self, amplitude, coeff):
        """Add amplitude and its correspondig coefficient.

        Arguments:
            amplitude (:py:class:`Amplitude`): Amplitude to add.
            coeff (:py:class:~zfit.core.parameter.Parameter): Coefficient corresponding
                to the amplitude.

        Return:
            int: Total number of configured amplitudes.

        Raise:
            ValueError: If the given amplitude is not compatible with previous ones due to
                a different final state.

        """

        def get_final_state_particles(amp):
            def recurse(leaves):
                parts = []
                for part, _, children in leaves:
                    if not children:
                        parts.append(part)
                    else:
                        parts.extend(recurse(children))
                return parts

            return recurse(amp.decay_tree[2])

        if self._amplitudes:
            if set(get_final_state_particles(self._amplitudes[-1])) != set(get_final_state_particles(amplitude)):
                raise ValueError("Incompatible Amplitude -> final state particles don't match")
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

    def pdf(self, name, **kwargs):
        """Get the PDF representing the decay."""
        return self._pdf(name, **kwargs)

    def _pdf(self, name, external_integral=None, **kwargs):
        pdf = SumAmplitudeSquaredPDF(obs=self.obs,
                                     name=name,
                                     amp_list=self._amplitudes,
                                     coef_list=self._coeffs,
                                     top_particle_mass=self._amplitudes[0].top_particle_mass,
                                     external_integral=external_integral,
                                     **kwargs)
        if self._sampling_function:
            pdf._do_sample = MethodType(self._sampling_function, pdf)
        if self._var_transforms:
            pdf._do_transform = MethodType(self._var_transforms, pdf)
        return pdf


# pylint: disable=W0212
def generator_sample_and_weights_factory(self):
    def sample_and_weights(n_to_produce, limits, dtype=None):
        if isinstance(limits, zfit.core.sample.EventSpace):
            limits.create_limits(n=n_to_produce)
        gen_parts, *output = self._do_sample(n_to_produce, limits)
        obs_vars = self._do_transform(gen_parts)
        if set(obs_vars.keys()) != set(self._obs.obs):
            raise ValueError(f"The obs_vars keys {obs_vars.keys()} do not match the observables {self._obs.obs}")
        obs_vars = tf.concat([obs_vars[obs] for obs in self._obs.obs], axis=1)
        return tuple([obs_vars] + output)

    return sample_and_weights


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
                 amplitude_extra_config=None,
                 **kwargs):
        self._external_integral = external_integral
        amp_extra_config = amplitude_extra_config if amplitude_extra_config else {}
        self._amplitudes = [(frac, amp, amp.amplitude(obs, **amp_extra_config))
                                 for frac, amp in zip(coef_list, amp_list)]
        self._amplitudes_combinations = [(frac1, frac2,
                                          AmplitudeProductCached(amp1=amp1,
                                                                 amp2=amp2))
                                         for (frac1, _, amp1), (frac2, _, amp2)
                                         in combinations(self._amplitudes, 2)]
        self._top_at_rest = tf.stack((0.0, 0.0, 0.0, ztf.to_real(top_particle_mass)), axis=-1)
        super().__init__(obs=obs, name=name, params={coef.name: coef for coef in coef_list}, **kwargs)
        self.update_integration_options(draws_per_dim=300000)
        self._sample_and_weights = MethodType(generator_sample_and_weights_factory, self)

    def _unnormalized_pdf(self, x):
        def unnormalized_pdf_func(x):
            value = tf.reduce_sum([(frac1 * frac2.conj) * amps.func(x=x)
                                   for frac1, frac2, amps
                                   in self._amplitudes_combinations],
                                  axis=0)
            return ztf.to_real(value)

        result = ztf.run_no_nan(unnormalized_pdf_func, x)
        return result

    def _do_sample(self, n_to_produce, limits):
        pseudo_yields = []
        generators = []
        for frac, amp, amp_func in self._amplitudes:
            pseudo_yields.append(ztf.to_real(frac * frac.conj *
                                             amp_func.integrate(limits=limits.get_subspace(amp_func.obs),
                                                                norm_range=False)))
            generators.append(amp.decay_phasespace())
        sum_yields = sum(pseudo_yields)
        n_to_generate = [tf.math.ceil(ztf.to_real(n_to_produce) * pseudo_yield / sum_yields)
                         for pseudo_yield in pseudo_yields]

        norm_weights = []
        particles = {}
        for amp_num in range(len(self._amplitudes)):
            print(f"Generating {amp_num}")
            norm_weight, parts = generators[amp_num].generate(self._top_at_rest, n_to_generate[amp_num])
            norm_weights.append(norm_weight)
            for part_name, gen_parts in parts.items():
                if part_name not in particles:
                    particles[part_name] = []
                particles[part_name].append(gen_parts)
                merged_particles = {part_name: tf.concat(particles, axis=0)
                                    for part_name, part_list in particles}
        merged_weights = tf.concat(norm_weights, axis=0)
        thresholds = ztf.random_uniform(shape=(n_to_produce,))
        return merged_particles, thresholds, merged_weights, sum_yields, len(norm_weights)

    def _do_transform(self, particle_dict):
        """Identity.

        In general, get a dictionary of particles as (4, n) tensors and transform it to
        a dictionary of observables.

        """
        return particle_dict

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
    def decay_tree(self):
        """tuple: Decay tree."""
        return self._decay_tree

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


class Resonance:
    """Resonance to be used for amplitude building and phasespace sampling.

    Arguments:
        particle (:py:class:`~particle.Particle`): Particle object corresponding to the
        resonance we want to model.
        resonance_model (:py:class:`~zfit.func.BaseFunc`): Class to model the resonance.
        model_config (dict): Configuration of `resonance_model`.

    """
    def __init__(self, particle, resonance_model, **model_config):  # noqa
        self.particle = particle
        self._model = resonance_model
        self._model_config = model_config

    def amplitude(self, name, obs, **extra_args):
        """Get amplitude to build PDFs.

        Arguments:
            name (str): Name of the amplitude.
            obs (:py:class:`~zfit.Space`): Observable space for the resonance.
            extra_args (dict): Extra configuration to pass to the resonance model.

        Return:
            :py:class:`~zfit.func.BaseFunc`

        """
        args = self._model_config.copy()
        args.update(extra_args)
        return self._model(obs=obs, name=sanitize_string(name), **args)

    def mass_sampler(self, name='', **extra_args):
        """Get mass sampler to use for phasespace generation.

        Build a function that can be used for mass sampling in the `phasespace` package.

        Arguments:
            name (str): Name of the amplitude.
            extra_args (dict): Extra configuration to pass to the resonance model.

        Return:
            Callable.

        """
        args = self._model_config.copy()
        args.update(extra_args)
        name = sanitize_string(name)

        def get_resonance_mass(mass_min, mass_max, n_events):
            def factory(n_to_generate):
                return mass_min(n_to_generate), mass_max(n_to_generate)

            space = zfit.core.sample.EventSpace(f'M({name})',
                                                limits=(lambda lim: lim[0],
                                                        lambda lim: lim[1]),
                                                factory=factory)
            return tf.reshape(self._model(obs=space,
                                          name=f'BW({name})',
                                          **args).sample(n_events),
                              (1, n_events))

        return get_resonance_mass

    @property
    def name(self):
        """str: Name of the resonance."""
        return self.particle.name

    @property
    def mass(self):
        """float: Mass of the resonance."""
        return self.particle.mass

    @property
    def width(self):
        """float: Width of the resonance."""
        return self.particle.width

# EOF


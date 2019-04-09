#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   decay.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   12.02.2019
# =============================================================================
"""Decay handler."""

import operator
import functools
from types import MethodType

import yaml

from particle import Particle
from particle.particle import literals as lp

import zfit
from zfit.core.parameter import Parameter, ComplexParameter

from zfit_amplitude.amplitude import Decay, Amplitude, SumAmplitudeSquaredPDF
import zfit_amplitude.dynamics as dynamics
from zfit_amplitude.utils import sanitize_string

# deactivating CUDA capable gpus
suppress_gpu = True  # doesn't really help currently, reciprocal complex not supported in tf on GPU
# suppress_gpu = False
if suppress_gpu:
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

WAVES = {'S': 0, 'D': 2}

# pylint: disable=E1101
V2PIPI = (lp.rho_770_0, lp.omega_782)
V2KPI = (lp.Kst_892_0, lp.K_0st_1430_0, Particle.from_pdgid(225))


# Mass functions
def kres_mass(kres_mass, kres_width, dec_string):
    def get_kres_mass(mass_min, mass_max, n_events):
        print_op = tf.print("DEBUG: nevents:", n_events)
        with tf.control_dependencies([print_op]):
            # HACK broadcast mass min max to nevents. Why can mass be 1 and nevents not?
            #mass_min = tf.broadcast_to(mass_min, (n_events,))
            #mass_max = tf.broadcast_to(mass_max, (n_events,))
            bw_res = dynamics.RelativisticBreitWignerReal(obs=zfit.core.sample.EventSpace(f'Kres_mass({dec_string})',
                                                                                          limits=(((mass_min,),), ((mass_max,),))),
                                                          name=f'Kres_BW({dec_string})',
                                                          mres=kres_mass, wres=kres_width).sample(n_events)
        return tf.reshape(bw_res, (1, n_events))

    return get_kres_mass


def vres_mass(vres_mass, vres_width, dec_string):
    def get_vres_mass(mass_min, mass_max, n_events):
        # HACK broadcast mass min max to nevents. Why can mass be 1 and nevents not?
        #mass_min = tf.broadcast_to(mass_min, (n_events,))
        #mass_max = tf.broadcast_to(mass_max, (n_events,))
        bw_v = dynamics.RelativisticBreitWignerReal(obs=zfit.core.sample.EventSpace(f'Vres_mass({dec_string})',
                                                                                    limits=(((mass_min,),), ((mass_max,),))),
                                                    name=f'Vres_BW({dec_string})',
                                                    mres=vres_mass, wres=vres_width).sample(n_events)
        return tf.reshape(bw_v, (1, n_events))

    return get_vres_mass


class B2KP1P2P3GammaAmplitude(Amplitude):
    """Guess the decay structure for B+ -> Kres+ (-> Vres (-> P2 P3) P1) gamma."""

    def __init__(self, kres, vres, wave='S'):
        """Configure the decay.

        B+ -> Kres+ (-> Vres (-> P2 P3) P1) gamma

        """
        kres_list = Particle.from_string_list(name=kres)
        if not kres_list:
            raise ValueError(f"Badly specified Kres -> {kres}")
        elif len(kres_list) > 1:
            raise ValueError("Ambiguous Kres specification -> {} (options: {})"
                             .format(kres, ','.join(res.fullname for res in kres_list)))
        else:
            self.kres = kres_list[0]
        vres_list = Particle.from_string_list(name=vres)
        if not vres_list:
            raise ValueError(f"Badly specified Vres -> {vres}")
        elif len(vres_list) > 1:
            raise ValueError("Ambiguous vres specification -> {} (options: {})"
                             .format(vres, ','.join(res.fullname for res in vres_list)))
        else:
            self.vres = vres_list[0]
        # pylint: disable=E1101
        if self.vres in V2PIPI:
            self.p1_part = lp.K_plus
            self.p2_part = lp.pi_plus
            self.p3_part = lp.pi_minus
            self.vres_children = slice(4, 13)  # ('K+', 'pi+', 'pi-', 'gamma')
        elif self.vres in V2KPI:
            self.p1_part = lp.pi_plus
            self.p2_part = lp.K_plus
            self.p3_part = lp.pi_minus
            self.vres_children = slice(0, 9)  # ('pi+', 'K+', 'pi-', 'gamma')
        else:
            raise ValueError(f"Vres not implemented! -> {self.vres.fullname}")
        if wave not in WAVES:
            raise ValueError(f"Unknown wave -> {wave}")
        self.wave = wave
        dec_string = sanitize_string(f"{self.kres.name}->{self.vres.name}{self.p1_part.name}")
        decay_tree = ('B+', lp.B_plus.mass, [
            (self.kres.name, kres_mass(self.kres.mass, self.kres.width, dec_string), [
                (self.vres.name, vres_mass(self.vres.mass, self.vres.width, dec_string), [
                    (self.p2_part.name, self.p2_part.mass, []),
                    (self.p3_part.name, self.p3_part.mass, [])]),
                (self.p1_part.name, self.p1_part.mass, [])]),
            ('gamma', 0.0, [])])
        super().__init__(decay_tree)

    def _amplitude(self, all_vectors_obs, chirality):
        """Get the full decay amplitude."""
        helicity = +1 if chirality == 'R' else -1
        spin_factor = dynamics.SpinFactor(all_vectors_obs, "1+", self.wave, helicity, self.kres.mass, self.vres.mass)
        kres_obs = all_vectors_obs.get_subspace(all_vectors_obs.obs[:-4])
        kres_bw = dynamics.RelativisticBreitWigner(obs=kres_obs, name="Kres_BW", mres=self.kres.mass,
                                                   wres=self.kres.width)
        vres_obs = all_vectors_obs.get_subspace(all_vectors_obs.obs[self.vres_children])
        vres_bw = dynamics.RelativisticBreitWigner(obs=vres_obs, name="Vres_BW", mres=self.vres.mass,
                                                   wres=self.vres.width)
        return spin_factor * kres_bw * vres_bw

    def _decay_string(self):
        """Build a decay string.

        Optionally, sanitize to conform to tensorflow variable names.

        """
        wave_str = "" if self.wave == 'S' else f'[{self.wave}]'
        decay_str = super()._decay_string()
        if wave_str:
            decay_str += wave_str
        return decay_str





SCALE = 1e4


class HackSampleSumPDF(zfit.pdf.SumPDF):
    @zfit.supports()
    def _sample(self, n, limits):
        # assuming only sum of 2 pdfs
        if len(self.pdfs) > 2 or self.is_extended:
            raise RuntimeError("This is a simple adhoc example, only use with 2 pdfs and non-extended.")
        n_sample1 = tf.ceil(n * self.fracs[0])
        n_sample2 = n - n_sample1
        print("using sampling")
        # HACK to use normal sampling
        # raise NotImplementedError
        sample = tf.concat([pdf.sample(n=n, limits=limits)
                            for n, pdf in zip((n_sample1, n_sample2), self.pdfs)],
                           axis=0)
        return sample


class Bp2KpipiGamma(Decay):
    """Generate a sum of amplitudes for photon polarization measurements.

    Configuration files have the following keys:
        - lambda_gamma, which can be either a value, with the addition of a True/False for floating it.
        - decays, a list of amplitudes to include with the format:
            Kres Vres f_real f_imag wave float_f_real float_f_imag
        (the last three are optional, but float_* need to be specified together).

    """

    # pylint: disable=E1101
    DEFAULT_PARTICLE_NAMES = {'gamma': 'gamma', 'K+': 'Kplus', 'pi-': 'piminus', 'pi+': 'piplus'}
    DEFAULT_COMPONENT_NAMES = {'x': '_PX', 'y': '_PY', 'z': '_PZ', 'e': '_PE'}
    DEFAULT_RANGES = {'x': (-SCALE, SCALE), 'y': (-SCALE, SCALE), 'z': (-SCALE, SCALE), 'e': (10, SCALE * 1e3)}

    def __init__(self, config_file):
        """Load fit config."""
        with open(config_file, 'r') as file_:
            config = yaml.load(file_)
        # Load lambda_gamma
        lambda_gamma_sp = config['lambda_gamma'].split()
        self.lambda_gamma = zfit.Parameter("lambda_gamma", float(lambda_gamma_sp[0]))
        if len(lambda_gamma_sp) > 1:
            self.lambda_gamma.floating = bool(int(lambda_gamma_sp[1]))
        # Load amplitudes
        amplitudes = []
        coeffs = []
        for amp in config['decays']:
            amp_sp = amp.split()
            kres, vres, f_a, f_phi = amp_sp[0:4]
            wave = amp_sp[4] if len(amp_sp) > 4 else "S"
            float_a, float_phi = map(bool, map(int, amp_sp[5:7])) if len(amp_sp) > 5 else (True, True)
            amplitude = B2KP1P2P3GammaAmplitude(kres, vres, wave)
            dec_string = amplitude.get_decay_string(True)
            a_i = Parameter(name=f"a_{dec_string}",
                            value=float(f_a), floating=float_a)
            phi_i = Parameter(name=f"phi_{dec_string}",
                              value=float(f_phi), floating=float_phi)
            fraction = ComplexParameter.from_polar(name=f"f_{dec_string}",
                                                   mod=a_i, arg=phi_i)
            amplitudes.append(amplitude)
            coeffs.append(fraction)
        obs_def = config.get('obs', {})
        obs = self._build_obs(obs_def)

        def var_transf(self, particles):
            return {tuple_name+tuple_comp: particles[phsp_name][comp_num]
                    for phsp_name, tuple_name in obs_def.get('particle-names',
                                                             Bp2KpipiGamma.DEFAULT_PARTICLE_NAMES).items()
                    for comp_num, tuple_comp in enumerate(obs_def.get('component-names',
                                                                      Bp2KpipiGamma.DEFAULT_COMPONENT_NAMES).values())}

        super().__init__(obs, amplitudes, coeffs, variable_transformations=var_transf)

    @staticmethod
    def _build_obs(obs_def):
        particle_names = obs_def.get('particle-names', Bp2KpipiGamma.DEFAULT_PARTICLE_NAMES)
        comp_names = obs_def.get('component-names', Bp2KpipiGamma.DEFAULT_COMPONENT_NAMES)
        return functools.reduce(operator.mul,
                                [functools.reduce(operator.mul,
                                                  [zfit.Space(obs=particle_names[particle] + comp_names[component],
                                                              limits=Bp2KpipiGamma.DEFAULT_RANGES[component])
                                                   for component in ('x', 'y', 'z', 'e')])
                                 for particle in ('K+', 'pi-', 'pi+', 'gamma')])

    def _pdf(self, name):
        """Build the PDF.

        Sum incoherently the coherent sum of L and R, separately.

        """
        right_pdf = SumAmplitudeSquaredPDF(obs=self.obs, name=f"{name}_R",
                                           amp_list=self._amplitudes,
                                           coef_list=self._coeffs,
                                           top_particle_mass=self._amplitudes[0].top_particle_mass,
                                           amplitude_extra_config={'chirality': "R"})
        left_pdf = SumAmplitudeSquaredPDF(obs=self.obs, name=f"{name}_L",
                                          amp_list=self._amplitudes,
                                          coef_list=self._coeffs,
                                          top_particle_mass=self._amplitudes[0].top_particle_mass,
                                          amplitude_extra_config={'chirality': "L"})
        if self._sampling_function:
            right_pdf._do_sample = MethodType(self._sampling_function, right_pdf)
            left_pdf._do_sample = MethodType(self._sampling_function, left_pdf)
        if self._var_transforms:
            right_pdf._do_transform = MethodType(self._var_transforms, right_pdf)
            left_pdf._do_transform = MethodType(self._var_transforms, left_pdf)
        safe_lambda_gamma = tf.minimum(self.lambda_gamma, 1.)
        return HackSampleSumPDF([right_pdf, left_pdf], [((1 + safe_lambda_gamma) / 2)], name="Sum_L_R")


if __name__ == "__main__":
    import tensorflow as tf
    import platform
    import zfit

    zfit.settings.set_verbosity(6)

    if platform.system() == 'Darwin':
        import matplotlib

        matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt

    decay = Bp2KpipiGamma('b2kpipigamma.yaml')
    lower = (tuple([-SCALE, -SCALE, -SCALE, 10] * 4),)  # last one is energy, only positive
    upper = (tuple([SCALE, SCALE, SCALE, SCALE] * 4),)
    limits = zfit.Space(obs=decay.obs, limits=(lower, upper))

    pdf = decay.pdf("Test")
    pdf.update_integration_options(draws_per_dim=3000)
    for dep in pdf.get_dependents(only_floating=False):
        print("{} {} Floating: {}".format(dep.name, zfit.run(dep), dep.floating))
    print("limits area", limits.area())
    zfit.settings.set_verbosity(6)

    sample = pdf.sample(n=3000, limits=limits)
    #branches_alias = {"_0_B#_E":}
    #sample = zfit.Data.from_root(path="kpipigamma_rest.root", treepath="DalitzEventList",
    #                             branches=)

    sample_np = zfit.run(sample)
    print("Shape sample produced: {}".format(sample_np.shape))
    for i, obs in enumerate(sample.obs):
        plt.figure()
        plt.title(obs)
        plt.hist(sample_np[:, i], bins=35)
    # plt.title(sample.obs[1])
    # plt.hist(zfit.run(sample)[:, 1])
    # plt.figure()
    # plt.title(sample.obs[2])
    # plt.hist(zfit.run(sample)[:, 2])
    # plt.figure()
    # plt.title(sample.obs[3])
    # plt.hist(zfit.run(sample)[:, 3])
    plt.show()
    nll = zfit.loss.UnbinnedNLL(model=pdf, data=sample, fit_range=limits)
    minimizer = zfit.minimize.MinuitMinimizer(verbosity=10)
    minimizer._use_tfgrad = False  # no analytic gradient, may has bugs in it
    for param in nll.get_dependents():
        param.load(0.8)
    result = minimizer.minimize(loss=nll)
    print(result.fmin)
    # sample_np = zfit.run(sample)
    # print(sample_np)
    # probs = pdf.pdf(x=sample, norm_range=limits)
    # print(zfit.run(probs))
    # x = np.random.uniform(high=np.ones(shape=16) * 50000, size=(10000, 16))
    # # probs = pdf.pdf(x=x)
    # # tf.add_check_numerics_ops()
    #
    # vals = pdf.unnormalized_pdf(x=x, component_norm_range=limits)
    # print([val for val in zfit.run(vals) if val != 0])
    # integral = pdf.integrate(limits=limits, norm_range=limits)
    # integral_np = zfit.run(integral)
    # print(integral_np)
    # probs_np = zfit.run(probs)
    # print(probs_np)

    # import tensorflow_probability as tfp

    # samples, _ = tfp.mcmc.sample_chain(
    #         num_results=1000,
    #         current_state=10000 * np.ones(shape=(1, 16)),
    #         kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
    #                 target_log_prob_fn=lambda x: tf.log(
    #                     ztf.constant(1e-81) + pdf.unnormalized_pdf(x=x, component_norm_range=limits)),
    #                 step_size=np.ones(shape=(1, 16)) * 100.,
    #                 seed=54),
    #         num_burnin_steps=30000,
    #         num_steps_between_results=1,  # Thinning.
    #         parallel_iterations=10)
    # # samples = tf.stack(samples, axis=-1)
    # samples_np = zfit.run(samples)
    # print(samples_np)

    # step_size = tf.Variable(name='step_size', initial_value=1., use_resource=True, trainable=False,
    #                         dtype=tf.float64)
    # zfit.run(step_size.initializer)
    # samples, _ = tfp.mcmc.sample_chain(
    #         num_results=1000,
    #         current_state=100 * np.ones(shape=(1, 16)),
    #         kernel=tfp.mcmc.HamiltonianMonteCarlo(
    #                 target_log_prob_fn=lambda x: tf.log(
    #                         ztf.constant(1e-88) + pdf.unnormalized_pdf(x=x, component_norm_range=limits)),
    #                 num_leapfrog_steps=3,
    #                 step_size=step_size,
    #                 step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy()),
    #         num_burnin_steps=1000,
    #         num_steps_between_results=1,  # Thinning.
    #         parallel_iterations=4)
    # # samples = tf.stack(samples, axis=-1)
    # samples_np = zfit.run(samples)
    # samples = ztf.convert_to_tensor(samples_np[:, 0, :])
    # samples_data = zfit.data.Data.from_tensor(obs=[str(i) for i in range(16)], tensor=samples)
    # probs = pdf.pdf(x=samples_data, norm_range=limits)
    # probs_np = zfit.run(probs)

    # print(samples_np)
    # print("probs", probs_np)

# EOF

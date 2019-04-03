#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   helpers.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   03.04.2019
# =============================================================================
"""Helper functions."""

import zfit


def sampling_resonance(resonance_model, model_config, name=''):
    """Calculate mass for sampling."""
    def get_resonance_mass(mass_min, mass_max, n_events):
        def factory(_):
            return mass_min(), mass_max()

        space = zfit.core.sample.EventSpace(f'M({name})',
                                            limits=(lambda lim: lim[0],
                                                    lambda lim: lim[1]),
                                            factory=factory)
        resonance = resonance_model(obs=space,
                             name=f'BW({name})',
                             **model_config).sample(n_events)
        return tf.reshape(bw, (1, n_events))
    return get_resonance_mass

# EOF

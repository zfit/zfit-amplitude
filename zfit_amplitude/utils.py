#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   utils.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   26.03.2019
# =============================================================================
"""Utilities."""

import re

_VALID_SCOPE_NAME_REGEX = re.compile("[A-Za-z0-9_./]*")


def sanitize_string(string):
    """Sanitize string for tensorflow."""
    return ''.join(item for item in _VALID_SCOPE_NAME_REGEX.findall(string
                                                                    .replace('->', '_')
                                                                    .replace(')', '_'))
                   if item).replace('__', '_')

# EOF

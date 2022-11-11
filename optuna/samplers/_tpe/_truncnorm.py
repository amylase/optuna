# This file contains the codes from SciPy project.
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import sys
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np


_norm_pdf_C = math.sqrt(2 * math.pi)
_norm_pdf_logC = math.log(_norm_pdf_C)


def _log_sum(log_p: float, log_q: float) -> float:
    if log_p > log_q:
        log_p, log_q = log_q, log_p
    return math.log1p(math.exp(log_p - log_q)) + log_q


def _log_diff(log_p: float, log_q: float) -> float:
    # returns log(q - p).
    # assuming that log_q is always greater than log_q
    return math.log1p(-math.exp(log_q - log_p)) + log_p


def _ndtr(a: float) -> float:
    x = a / 2**0.5
    z = abs(x)

    if z < 1 / 2**0.5:
        y = 0.5 + 0.5 * math.erf(x)
    else:
        y = 0.5 * math.erfc(z)
        if x > 0:
            y = 1.0 - y

    return y


def _log_ndtr(a: float) -> float:
    if a > 6:
        return -_ndtr(-a)
    if a > -20:
        return math.log(_ndtr(a))

    log_LHS = -0.5 * a**2 - math.log(-a) - 0.5 * math.log(2 * math.pi)
    last_total = 0.0
    right_hand_side = 1.0
    numerator = 1.0
    denom_factor = 1.0
    denom_cons = 1 / a**2
    sign = 1
    i = 0

    while abs(last_total - right_hand_side) > sys.float_info.epsilon:
        i += 1
        last_total = right_hand_side
        sign = -sign
        denom_factor *= denom_cons
        numerator *= 2 * i - 1
        right_hand_side += sign * numerator * denom_factor

    return log_LHS + math.log(right_hand_side)


def _norm_logpdf(x: float) -> float:
    return -(x**2) / 2.0 - _norm_pdf_logC


def _log_gauss_mass(a: float, b: float) -> float:
    """Log of Gaussian probability mass within an interval"""

    # Calculations in right tail are inaccurate, so we'll exploit the
    # symmetry and work only in the left tail

    def mass_case_left(a: float, b: float) -> float:
        return _log_diff(_log_ndtr(b), _log_ndtr(a))

    def mass_case_right(a: float, b: float) -> float:
        return mass_case_left(-b, -a)

    def mass_case_central(a: float, b: float) -> float:
        # Previously, this was implemented as:
        # left_mass = mass_case_left(a, 0)
        # right_mass = mass_case_right(0, b)
        # return _log_sum(left_mass, right_mass)
        # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
        # Correct for this with an alternative formulation.
        # We're not concerned with underflow here: if only one term
        # underflows, it was insignificant; if both terms underflow,
        # the result can't accurately be represented in logspace anyway
        # because sc.log1p(x) ~ x for small x.
        return math.log1p(-_ndtr(a) - _ndtr(-b))

    if b <= 0:
        return mass_case_left(a, b)
    elif a > 0:
        return mass_case_right(a, b)
    else:
        return mass_case_central(a, b)


_P0 = [
    -5.99633501014107895267e1,
    9.80010754185999661536e1,
    -5.66762857469070293439e1,
    1.39312609387279679503e1,
    -1.23916583867381258016e0,
]
_P1 = [
    4.05544892305962419923,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4,
]
_P2 = [
    3.23774891776946035970,
    6.91522889068984211695,
    3.93881025292474443415,
    1.33303460815807542389,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9,
]
_Q0 = [
    1.95448858338141759834e0,
    4.67627912898881538453e0,
    8.63602421390890590575e1,
    -2.25462687854119370527e2,
    2.00260212380060660359e2,
    -8.20372256168333339912e1,
    1.59056225126211695515e1,
    -1.18331621121330003142e0,
]
_Q1 = [
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4,
]
_Q2 = [
    6.02427039364742014255,
    3.67983563856160859403,
    1.37702099489081330271,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9,
]


def _polevl(x: float, coefs: List[float], degree: int) -> float:
    v = 0
    for i in range(degree):
        v += coefs[i]
        v *= x
    v += coefs[degree]
    return v


def _p1evl(x: float, coefs: List[float], degree: int) -> float:
    v = 1
    for i in range(degree):
        v *= x
        v += coefs[i]
    return v


def _ndtri(y0: float) -> float:
    if y0 == 0:
        return -math.inf
    if y0 == 1:
        return math.inf
    if y0 < 0 or y0 > 1:
        return math.nan

    code = 1
    y = y0
    if y > (1.0 - 0.13533528323661269189):  # 0.135... = exp(-2)
        y = 1.0 - y
        code = 0

    if y > 0.13533528323661269189:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * _polevl(y2, _P0, 4) / _p1evl(y2, _Q0, 8))
        x = x * 2.50662827463100050242e0  # = sqrt(2pi)
        return x

    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x

    z = 1.0 / x
    if x < 8.0:  # y > exp(-32) = 1.2664165549e-14
        x1 = z * _polevl(z, _P1, 8) / _p1evl(z, _Q1, 8)
    else:
        x1 = z * _polevl(z, _P2, 8) / _p1evl(z, _Q2, 8)
    x = x0 - x1
    if code != 0:
        x = -x
    return x


def _ndtri_exp_small_y(y: float) -> float:
    if y >= -sys.float_info.max * 0.5:
        x = math.sqrt(-2 * y)
    else:
        x = math.sqrt(2) * math.sqrt(-y)
    x0 = x - math.log(x) / x
    z = 1 / x
    if x < 8.0:
        x1 = z * _polevl(z, _P1, 8) / _p1evl(z, _Q1, 8)
    else:
        x1 = z * _polevl(z, _P2, 8) / _p1evl(z, _Q2, 8)
    return x1 - x0


def _ndtri_exp(y: float) -> float:
    if y < -sys.float_info.max:
        return -math.inf
    elif y < -2.0:
        return _ndtri_exp_small_y(y)
    elif y > math.log1p(-math.exp(-2)):
        return -_ndtri(-math.expm1(y))
    else:
        return _ndtri(math.exp(y))


@np.vectorize
def ppf(q: float, a: float, b: float) -> float:
    if a == b:
        return math.nan
    if q == 0:
        return a
    if q == 1:
        return b

    def ppf_left(q: float, a: float, b: float) -> float:
        log_Phi_x = _log_sum(_log_ndtr(a), math.log(q) + _log_gauss_mass(a, b))
        return _ndtri_exp(log_Phi_x)

    def ppf_right(q: float, a: float, b: float) -> float:
        log_Phi_x = _log_sum(_log_ndtr(-b), math.log1p(-q) + _log_gauss_mass(a, b))
        return -_ndtri_exp(log_Phi_x)

    if a < 0:
        return ppf_left(q, a, b)
    else:
        return ppf_right(q, a, b)


def rvs(
    a: np.ndarray,
    b: np.ndarray,
    loc: Union[np.ndarray, float] = 0,
    scale: Union[np.ndarray, float] = 1,
    size: int = 1,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    random_state = random_state or np.random.RandomState()
    percentiles = random_state.uniform(low=0, high=1, size=size)
    return ppf(percentiles, a, b) * scale + loc


@np.vectorize
def logpdf(x: float, a: float, b: float, loc: float = 0, scale: float = 1) -> float:
    x = (x - loc) / scale
    if a == b:
        return math.nan
    if x < a or b < x:
        return -math.inf
    return _norm_logpdf(x) - _log_gauss_mass(a, b)


def _logsf(x: float, a: float, b: float) -> float:
    logsf = _log_gauss_mass(x, b) - _log_gauss_mass(a, b)
    if logsf > -0.1:  # avoid catastrophic cancellation
        logsf = math.log1p(-math.exp(logcdf(x, a, b)))
    return logsf


@np.vectorize
def logcdf(x: float, a: float, b: float, loc: float = 0, scale: float = 1) -> float:
    if a == b:
        return math.nan
    x = (x - loc) / scale
    if x <= a:
        return -math.inf
    if x >= b:
        return 0
    logcdf = _log_gauss_mass(a, x) - _log_gauss_mass(a, b)
    if logcdf > -0.1:  # avoid catastrophic cancellation
        logcdf = math.log1p(-math.exp(_logsf(x, a, b)))
    return logcdf


@np.vectorize
def cdf(x: float, a: float, b: float, loc: float = 0, scale: float = 1) -> float:
    if a == b:
        return math.nan
    x = (x - loc) / scale
    if x <= a:
        return 0
    if x >= b:
        return 1
    return math.exp(logcdf(x, a, b))

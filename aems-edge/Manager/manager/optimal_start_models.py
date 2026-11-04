# -*- coding: utf-8 -*-
# ===----------------------------------------------------------------------===
#
#                 AEMS Application — Optimal Start Zone Models
#
# ===----------------------------------------------------------------------===
#
# Copyright 2026 Battelle Memorial Institute
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# ===----------------------------------------------------------------------===


"""
Optimal-start zone models.

Each model predicts how many minutes before occupancy a unit must start so the
zone reaches its occupied setpoint on time, and learns from each morning's
recovery how fast the zone responds.

Models
------
* :class:`Carrier`  - learned recovery rate (°F/min), scaled by outdoor air.
* :class:`Siemens`  - minutes linear in the zone gap plus an outdoor-air term.
* :class:`Johnson`  - minutes quadratic in the zone gap plus an offset.
* :class:`Sbs`      - first-order (exponential) decay of the setpoint error.
* :class:`RecoveryRegression` - robust regression of observed recovery times on
  the zone gap and outdoor air (the default deciding model).

All models derive from :class:`OptimalStartZoneModel`, which provides the
shared training pipeline (:meth:`OptimalStartZoneModel.train`), the training
window, mode detection and persistence. The coefficient models (Carrier,
Siemens, Johnson, Sbs) keep a short history of per-morning coefficients and
average it with :func:`~manager.utils.ema`; :class:`RecoveryRegression` keeps a
history of per-morning observations and refits at every prediction.

Heating and cooling are learned separately for every model. Only the fields in
each class's ``STATE_FIELDS`` are persisted and restored, so configuration
values are never loaded back from the config store.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime as dt
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from volttron.platform.agent.utils import format_timestamp

from . import OptimalStartConfig
from .points import DFPoints as dfpts
from .utils import (
    calculate_prestart_time,
    ema,
    get_operating_mode,
    get_time_target,
    get_time_temp_diff,
    offset_time,
    parse_df,
    trim,
)

_log = logging.getLogger(__name__)

_DEFAULT_TIMEZONE = "US/Pacific"
_TRAINING_END_OFFSET_MINUTES = 60

# Heat-transfer-rate lookup: temperature-drop thresholds 0..15 °F in 0.25 °F steps
_HTR_THRESHOLDS = np.linspace(0, 15, 61)


class OperatingMode(str, Enum):
    """
    Operating mode needed to recover the zone.

    The enum values (``"cooling"`` / ``"heating"``) are the same strings used by
    :func:`~manager.utils.get_operating_mode`, :func:`~manager.utils.parse_df` and
    the persisted state field names.

    :cvar COOLING: Zone is above the cooling setpoint.
    :cvar HEATING: Zone is below the heating setpoint.
    """
    COOLING = "cooling"
    HEATING = "heating"

    @property
    def prefix(self) -> str:
        """
        Coefficient attribute prefix for the mode.

        :return: ``"c"`` for cooling, ``"h"`` for heating (e.g. ``c1`` / ``h1``).
        :rtype: str
        """
        return "c" if self is OperatingMode.COOLING else "h"


@dataclass(frozen=True)
class ZoneConditions:
    """
    Snapshot of the zone and outdoor conditions at one instant.

    Immutable; built from one row of the device DataFrame with :meth:`from_row`
    or from the most recent row with :meth:`latest`.

    :ivar zone_temperature: Zone temperature (°F).
    :vartype zone_temperature: float
    :ivar cooling_setpoint: Occupied cooling setpoint (°F).
    :vartype cooling_setpoint: float
    :ivar heating_setpoint: Occupied heating setpoint (°F).
    :vartype heating_setpoint: float
    :ivar outdoor_air_temperature: Outdoor-air temperature (°F); NaN when missing
        or stale.
    :vartype outdoor_air_temperature: float
    """
    zone_temperature: float
    cooling_setpoint: float
    heating_setpoint: float
    outdoor_air_temperature: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "ZoneConditions":
        """
        Build conditions from one DataFrame row.

        :param row: Row with ``zonetemperature``, ``coolingsetpoint``,
            ``heatingsetpoint`` and ``outdoortemperature``.
        :type row: pd.Series
        :return: The conditions in that row.
        :rtype: ZoneConditions
        """
        return cls(
            zone_temperature=row["zonetemperature"],
            cooling_setpoint=row["coolingsetpoint"],
            heating_setpoint=row["heatingsetpoint"],
            outdoor_air_temperature=row["outdoortemperature"],
        )

    @classmethod
    def latest(cls, data: pd.DataFrame) -> "ZoneConditions":
        """
        Build conditions from the last row of ``data``.

        :param data: Time-indexed device data (oldest first).
        :type data: pd.DataFrame
        :return: The most recent conditions.
        :rtype: ZoneConditions
        """
        return cls.from_row(data.iloc[-1])

    def zone_error(self, mode: OperatingMode) -> float:
        """
        Degrees the zone must move to reach the setpoint for ``mode``.

        Positive when recovery is needed: ``zone - cooling SP`` for cooling and
        ``heating SP - zone`` for heating.

        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: Zone gap in °F.
        :rtype: float
        """
        if mode is OperatingMode.HEATING:
            return self.heating_setpoint - self.zone_temperature
        return self.zone_temperature - self.cooling_setpoint

    def outdoor_error(self, mode: OperatingMode) -> float:
        """
        Outdoor-air load relative to the setpoint for ``mode``.

        ``OAT - cooling SP`` for cooling and ``heating SP - OAT`` for heating, so a
        larger value always means a harder recovery.

        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: Outdoor-air difference in °F (NaN if outdoor air is missing).
        :rtype: float
        """
        if mode is OperatingMode.HEATING:
            return self.heating_setpoint - self.outdoor_air_temperature
        return self.outdoor_air_temperature - self.cooling_setpoint


def _coefficients_are_valid(
    values: dict[str, float], *, model_label: str, allow_zero: bool = False
) -> bool:
    """
    Check that learned coefficients are usable.

    :param values: Coefficient name -> value.
    :type values: dict[str, float]
    :param model_label: Label used in debug logging (e.g. ``"Carrier cooling"``).
    :type model_label: str
    :param allow_zero: Accept zero values (default: strictly positive required).
    :type allow_zero: bool
    :return: True if every value is finite and positive (or zero when allowed).
    :rtype: bool
    """
    for name, value in values.items():
        if not np.isfinite(value):
            _log.debug("%s: non-finite coefficient %s = %s", model_label, name, value)
            return False
        if value < 0 or (not allow_zero and value == 0):
            _log.debug("%s: non-positive coefficient %s = %s", model_label, name, value)
            return False
    return True


def _record(**fields: Any) -> dict:
    """
    Training record published after a training run.

    :param fields: Model-specific values to include.
    :return: ``{"date": <now>, **fields}``.
    :rtype: dict
    """
    return {"date": format_timestamp(dt.now()), **fields}


class OptimalStartZoneModel(ABC):
    """
    Base class for optimal-start models.

    Subclasses implement :meth:`_train_mode` (learn from one morning's recovery),
    :meth:`calculate_prestart` (minutes before occupancy to start),
    :attr:`is_trained` and :meth:`trained_for`. The shared :meth:`train` method
    slices the training window, detects the operating mode and hands the model
    its input (:meth:`_model_input`).

    Prediction conventions shared by all models: when no recovery is needed the
    model returns ``latest_start_time``; when it is not trained for the needed
    mode it returns ``earliest_start_time``.

    :cvar STATE_FIELDS: Attributes persisted to / restored from the config store.
    :cvar SEPARATE_WEEKEND_MODEL: Whether the manager keeps a separate copy of the
        model for the day after a weekend/holiday.
    :ivar record: Summary of the most recent training run (published by the
        manager); empty when the last run learned nothing.
    :vartype record: dict
    :ivar cfg: Optimal-start configuration in effect.
    :vartype cfg: OptimalStartConfig
    :ivar latest_start_time: Minimum prestart in minutes (configured).
    :vartype latest_start_time: float
    :ivar earliest_start_time: Maximum prestart in minutes (configured).
    :vartype earliest_start_time: float
    :ivar setpoint_deviation: Allowable deviation from setpoint (°F) used for mode
        detection and as SBS's target.
    :vartype setpoint_deviation: float
    :ivar training_period_window: Number of coefficient history entries kept.
    :vartype training_period_window: int
    :ivar tz: Local timezone name (``LOCAL_TZ`` environment variable).
    :vartype tz: str
    """

    #: Attributes persisted to / restored from the config store.
    STATE_FIELDS: tuple[str, ...] = ()
    #: Whether the manager keeps a separate copy of this model for the day after a weekend/holiday.
    SEPARATE_WEEKEND_MODEL: bool = True

    def __init__(self, config: OptimalStartConfig, get_current_schedule_fn: Callable):
        """
        :param config: Optimal-start configuration.
        :type config: OptimalStartConfig
        :param get_current_schedule_fn: Callable returning today's
            :class:`~manager.Schedule` (or None).
        :type get_current_schedule_fn: Callable
        """
        self._get_current_schedule = get_current_schedule_fn
        self.record: dict = {}
        self.update_config(config)

    def update_config(self, config: OptimalStartConfig) -> None:
        """
        Apply a new optimal-start configuration (learned state is kept).

        :param config: New configuration.
        :type config: OptimalStartConfig
        """
        self.cfg = config
        self.latest_start_time: float = config.latest_start_time
        self.earliest_start_time: float = config.earliest_start_time
        self.setpoint_deviation: float = config.allowable_setpoint_deviation
        self.training_period_window: int = config.training_period_window
        self.tz: str = os.environ.get("LOCAL_TZ", _DEFAULT_TIMEZONE)

    # -- Persistence --------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """
        Learned state to persist.

        :return: ``STATE_FIELDS`` attribute name -> value (learned values only,
            never configuration).
        :rtype: dict[str, Any]
        """
        return {name: getattr(self, name) for name in self.STATE_FIELDS}

    def load_model(self, model_dict: dict[str, Any]) -> None:
        """
        Restore learned state saved by :meth:`get_state`.

        Only keys listed in ``STATE_FIELDS`` are applied; unknown or legacy keys
        (including old configuration values) are ignored.

        :param model_dict: Persisted state (may be None).
        :type model_dict: dict[str, Any]
        """
        for name, value in (model_dict or {}).items():
            if name in self.STATE_FIELDS:
                setattr(self, name, value)

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """
        Whether the model has learned anything (for either mode).

        :return: True once at least one coefficient / observation is stored.
        :rtype: bool
        """

    @abstractmethod
    def trained_for(self, mode: OperatingMode) -> bool:
        """
        Whether the model has learned what it needs to predict ``mode``.

        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: True if a prediction for ``mode`` uses learned values.
        :rtype: bool
        """

    def ready_for(self, data: pd.DataFrame) -> bool:
        """
        Whether this model's prediction for ``data`` is a real prediction.

        Used by the manager to exclude untrained models from the start decision.

        :param data: Current device data (stale readings already masked).
        :type data: pd.DataFrame
        :return: True when no recovery is needed or the model is trained for the
            mode the zone needs now; False on empty data.
        :rtype: bool
        """
        if data is None or data.empty:
            return False
        mode = self._determine_mode(ZoneConditions.latest(data))
        return True if mode is None else self.trained_for(mode)

    # -- Training -----------------------------------------------------------

    def train(self, data: pd.DataFrame, prestart: Optional[float] = None) -> None:
        """
        Learn from this morning's recovery.

        Slices the training window from ``prestart`` minutes before occupancy to
        occupancy + 60 min, keeps the supply-fan-on rows (dropping the start-up
        transition row), detects the operating mode, builds the model input with
        :meth:`_model_input` and passes it to :meth:`_train_mode`. Days without a
        timed schedule, fan operation, active heating/cooling or usable data are
        skipped. :attr:`record` is reset and then filled by the model.

        :param data: Today's device data, time indexed (UTC or tz-aware).
        :type data: pd.DataFrame
        :param prestart: Minutes before occupancy the training window starts
            (normally the committed prestart + a buffer); defaults to
            ``earliest_start_time``.
        :type prestart: float | None
        """
        self.record = {}
        if prestart is None:
            prestart = self.earliest_start_time
        if data is None or data.empty:
            return

        schedule = self._get_current_schedule()
        if schedule is None or schedule.start is None:
            _log.debug("No timed schedule today — skipping training.")
            return

        window = self._training_window(data, schedule, prestart)
        fan_on = window[window["supplyfanstatus"] != 0].iloc[1:]  # drop the start-up transition row
        if fan_on.empty:
            _log.debug("Supply fan was off for the entire training window.")
            return

        mode = get_operating_mode(fan_on)
        if mode is None:
            _log.debug("No active heating or cooling during training period.")
            return
        mode = OperatingMode(mode)
        model_input = self._model_input(window, fan_on, mode)
        if model_input.empty:
            _log.debug("No usable %s data for training.", mode.value)
            return
        self._train_mode(model_input, mode)

    def _model_input(
        self, window: pd.DataFrame, fan_on: pd.DataFrame, mode: OperatingMode
    ) -> pd.DataFrame:
        """
        Data passed to :meth:`_train_mode`.

        The default is the active recovery only (compressor/heat on, until setpoint is
        reached) from :func:`~manager.utils.parse_df`. The coefficient models measure
        elapsed time from timestamps, so the off-cycle gaps parse_df leaves behind do
        not affect them.

        :param window: All rows of the training window (local time).
        :type window: pd.DataFrame
        :param fan_on: Window rows with the supply fan on (first row dropped).
        :type fan_on: pd.DataFrame
        :param mode: Detected operating mode.
        :type mode: OperatingMode
        :return: Training input; empty if nothing usable.
        :rtype: pd.DataFrame
        """
        return parse_df(fan_on, mode.value)

    @abstractmethod
    def _train_mode(self, data: pd.DataFrame, mode: OperatingMode) -> None:
        """
        Learn coefficients from one morning's recovery.

        :param data: Output of :meth:`_model_input`.
        :type data: pd.DataFrame
        :param mode: Operating mode of the recovery.
        :type mode: OperatingMode
        """
        ...

    @abstractmethod
    def calculate_prestart(self, data: pd.DataFrame) -> float:
        """
        Minutes before occupancy the unit should start.

        :param data: Current device data; the last row is "now".
        :type data: pd.DataFrame
        :return: Prestart in minutes (``latest_start_time`` if no recovery is needed,
            ``earliest_start_time`` if empty data or not trained for the mode).
        :rtype: float
        """
        ...

    # -- Shared helpers -----------------------------------------------------

    def _training_window(self, data: pd.DataFrame, schedule, prestart: float,
                         end_offset: int = _TRAINING_END_OFFSET_MINUTES) -> pd.DataFrame:
        """
        Rows between (occupancy - prestart) and (occupancy + ``end_offset``), local time.

        :param data: Device data (index converted to ``tz``).
        :type data: pd.DataFrame
        :param schedule: Today's schedule (uses ``schedule.start``).
        :param prestart: Minutes before occupancy the window starts.
        :type prestart: float
        :param end_offset: Minutes after occupancy the window ends.
        :type end_offset: int
        :return: Window rows with a local-time index.
        :rtype: pd.DataFrame
        """
        window_start = calculate_prestart_time(schedule.start, prestart)
        window_end = offset_time(schedule.start, end_offset)
        _log.debug("Training window: %s → %s", window_start, window_end)

        data = data.copy()
        data.index = pd.to_datetime(data.index, utc=True).tz_convert(self.tz)
        return data.between_time(window_start, window_end)

    def _compute_heat_transfer_rate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Recovery milestones used by the coefficient models.

        Finds the first row at which the temperature difference (``tempdiff``) has
        dropped by each threshold in ``_HTR_THRESHOLDS`` (0 to 15 °F in 0.25 °F
        steps). If only the zero threshold was crossed but the zone started outside
        the allowable deviation, the last row is added as the end point. NaN values
        never qualify.

        :param data: Parsed recovery data with a ``tempdiff`` column.
        :type data: pd.DataFrame
        :return: Milestone rows, deduplicated, in time order (empty if none).
        :rtype: pd.DataFrame
        """
        temp_diff = data[dfpts.tempdiff.value].to_numpy(dtype=float)
        if len(temp_diff) == 0:
            return pd.DataFrame()
        initial_diff = temp_diff[0]

        # Running max of the drop from the initial value; NaNs never qualify.
        drop = np.nan_to_num(initial_diff - temp_diff, nan=-np.inf)
        running_max = np.maximum.accumulate(drop)
        positions = np.searchsorted(running_max, _HTR_THRESHOLDS, side="left")
        positions = positions[positions < len(data)]

        if len(positions) == 0:
            return pd.DataFrame()
        # Only the zero threshold was crossed but the zone was outside tolerance:
        # use the last row as the end point.
        if len(positions) == 1 and initial_diff > self.setpoint_deviation:
            positions = np.append(positions, len(data) - 1)

        htr = data.iloc[np.unique(positions)].copy()
        return htr[~htr.index.duplicated(keep="first")]

    def _determine_mode(self, conditions: ZoneConditions) -> Optional[OperatingMode]:
        """
        Mode needed to bring the zone within the allowable deviation of its setpoints.

        :param conditions: Zone conditions.
        :type conditions: ZoneConditions
        :return: HEATING if the zone is more than the deviation below the heating
            setpoint, COOLING if more than the deviation above the cooling setpoint,
            otherwise None (no recovery needed).
        :rtype: OperatingMode | None
        """
        if conditions.zone_temperature + self.setpoint_deviation < conditions.heating_setpoint:
            return OperatingMode.HEATING
        if conditions.zone_temperature - self.setpoint_deviation > conditions.cooling_setpoint:
            return OperatingMode.COOLING
        return None


class Carrier(OptimalStartZoneModel):
    """
    Rate-based model: minutes = degrees to recover / learned recovery rate.

    Each morning's recovery gives a rate in °F/min (temperature drop between the
    heat-transfer milestones divided by the elapsed time). The prediction divides
    the zone gap by the EMA of recent rates, scaled by outdoor air::

        oat_ratio = (reference - OAT_now) / (reference - OAT_training)
        minutes   = gap / (oat_ratio * rate)

    with reference 100 °F for cooling and 0 °F for heating: hotter (colder)
    weather than during training lowers the effective rate and lengthens the
    prediction.

    :ivar c1: Cooling rate history (``[timestamp, °F/min]``).
    :vartype c1: list
    :ivar h1: Heating rate history.
    :vartype h1: list
    :ivar oat_clg: Mean outdoor air during cooling training mornings.
    :vartype oat_clg: list
    :ivar oat_htg: Mean outdoor air during heating training mornings.
    :vartype oat_htg: list
    :ivar adjust_time: Reserved adjustment (minutes); not used in prediction.
    :vartype adjust_time: float
    """

    STATE_FIELDS = ("c1", "h1", "oat_clg", "oat_htg", "adjust_time")
    _OAT_REFERENCE = {OperatingMode.HEATING: 0.0, OperatingMode.COOLING: 100.0}
    _OAT_FIELD = {OperatingMode.HEATING: "oat_htg", OperatingMode.COOLING: "oat_clg"}

    def __init__(self, config: OptimalStartConfig, schedule_fn: Callable):
        """
        :param config: Optimal-start configuration.
        :type config: OptimalStartConfig
        :param schedule_fn: Callable returning today's schedule.
        :type schedule_fn: Callable
        """
        super().__init__(config, schedule_fn)
        self.c1: list = []
        self.h1: list = []
        self.oat_clg: list = []
        self.oat_htg: list = []
        self.adjust_time: float = 0.0

    @property
    def is_trained(self) -> bool:
        """
        :return: True once a cooling or heating rate has been learned.
        :rtype: bool
        """
        return bool(self.c1 or self.h1)

    def trained_for(self, mode: OperatingMode) -> bool:
        """
        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: True if a rate has been learned for ``mode``.
        :rtype: bool
        """
        return bool(getattr(self, f"{mode.prefix}1"))

    def _train_mode(self, data: pd.DataFrame, mode: OperatingMode) -> None:
        """
        Learn this morning's recovery rate and the outdoor air it was learned at.

        :param data: Parsed recovery data (``parse_df`` output).
        :type data: pd.DataFrame
        :param mode: Operating mode of the recovery.
        :type mode: OperatingMode
        """
        htr = self._compute_heat_transfer_rate(data)
        if htr.empty:
            _log.debug("Carrier %s: HTR table is empty.", mode.value)
            return

        time_diff, temp_diff = get_time_temp_diff(htr, self.setpoint_deviation)
        if not time_diff:
            _log.debug("Carrier %s: time_diff is zero.", mode.value)
            return

        rate = temp_diff / time_diff
        if not _coefficients_are_valid({"rate": rate}, model_label=f"Carrier {mode.value}"):
            return

        coeff_field = f"{mode.prefix}1"
        oat_field = self._OAT_FIELD[mode]
        history = trim(getattr(self, coeff_field), rate, self.training_period_window)
        setattr(self, coeff_field, history)
        setattr(self, oat_field, trim(getattr(self, oat_field),
                                      htr["outdoortemperature"].mean(),
                                      self.training_period_window))
        self.record = _record(**{coeff_field: rate, f"{coeff_field}_array": history})

    def calculate_prestart(self, data: pd.DataFrame) -> float:
        """
        Minutes needed: zone gap / (outdoor-air ratio x EMA of learned rates).

        :param data: Current device data; the last row is "now".
        :type data: pd.DataFrame
        :return: Prestart in minutes (not clamped; the manager clamps).
        :rtype: float
        """
        if data.empty:
            return self.earliest_start_time

        cond = ZoneConditions.latest(data)
        mode = self._determine_mode(cond)
        if mode is None:
            return self.latest_start_time

        history = getattr(self, f"{mode.prefix}1")
        if not history:
            return self.earliest_start_time

        oat_history = getattr(self, self._OAT_FIELD[mode])
        oat_ref = ema(oat_history) if oat_history else cond.outdoor_air_temperature
        reference = self._OAT_REFERENCE[mode]
        # Scale the learned rate by how far today's OAT is from the reference
        # compared with the OAT seen during training.
        oat_ratio = (reference - cond.outdoor_air_temperature) / (reference - oat_ref)
        return cond.zone_error(mode) / (oat_ratio * ema(history))


class Siemens(OptimalStartZoneModel):
    """
    Linear model with an outdoor-air interaction::

        minutes = 60 * (k1 * gap + k2 * gap * oat_gap / 10) + adjust_time

    ``gap`` is the zone gap and ``oat_gap`` the outdoor-air difference (both
    floored at 0). ``k1`` (hours per °F) and ``k2`` are fitted per morning from
    the recovery and averaged with an EMA.

    :ivar c1: Cooling ``k1`` history.
    :vartype c1: list
    :ivar c2: Cooling ``k2`` history.
    :vartype c2: list
    :ivar h1: Heating ``k1`` history.
    :vartype h1: list
    :ivar h2: Heating ``k2`` history.
    :vartype h2: list
    :ivar adjust_time: Constant added to the prediction (minutes).
    :vartype adjust_time: float
    """

    STATE_FIELDS = ("c1", "c2", "h1", "h2", "adjust_time")
    _OSP_SCALE_FACTOR = 10.0

    def __init__(self, config: OptimalStartConfig, schedule_fn: Callable):
        """
        :param config: Optimal-start configuration.
        :type config: OptimalStartConfig
        :param schedule_fn: Callable returning today's schedule.
        :type schedule_fn: Callable
        """
        super().__init__(config, schedule_fn)
        self.c1: list = []
        self.c2: list = []
        self.h1: list = []
        self.h2: list = []
        self.adjust_time: float = 0.0

    @property
    def is_trained(self) -> bool:
        """
        :return: True once a cooling or heating ``k1`` has been learned.
        :rtype: bool
        """
        return bool(self.c1 or self.h1)

    def trained_for(self, mode: OperatingMode) -> bool:
        """
        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: True if both ``k1`` and ``k2`` have been learned for ``mode``.
        :rtype: bool
        """
        return bool(getattr(self, f"{mode.prefix}1")) and bool(getattr(self, f"{mode.prefix}2"))

    def _train_mode(self, data: pd.DataFrame, mode: OperatingMode) -> None:
        """
        Fit ``k1`` / ``k2`` for this morning and append them to the histories.

        :param data: Parsed recovery data (``parse_df`` output).
        :type data: pd.DataFrame
        :param mode: Operating mode of the recovery.
        :type mode: OperatingMode
        """
        htr = self._compute_heat_transfer_rate(data)
        if htr.empty:
            _log.debug("Siemens %s: HTR table is empty.", mode.value)
            return

        first = ZoneConditions.from_row(htr.iloc[0])
        coeffs = self._fit_coefficients(
            data, htr, first.zone_error(mode), first.outdoor_error(mode), mode.value
        )
        if coeffs is None:
            return

        p = mode.prefix
        self.record = _record()
        for idx, value in enumerate(coeffs, start=1):
            field = f"{p}{idx}"
            history = trim(getattr(self, field), value, self.training_period_window)
            setattr(self, field, history)
            self.record[field] = value
            self.record[f"{field}_array"] = history

    def _fit_coefficients(
        self, data: pd.DataFrame, htr: pd.DataFrame,
        zone_sp_diff: float, outdoor_sp_diff: float, label: str,
    ) -> Optional[tuple[float, float]]:
        """
        Coefficients for one morning's recovery.

        ``k1`` = recovery hours per °F; ``k2`` is the remaining time explained by the
        outdoor-air term (clamped at 0 rather than rejected).

        :param data: Parsed recovery data.
        :type data: pd.DataFrame
        :param htr: Heat-transfer milestones (:meth:`_compute_heat_transfer_rate`).
        :type htr: pd.DataFrame
        :param zone_sp_diff: Zone gap at the first milestone (°F).
        :type zone_sp_diff: float
        :param outdoor_sp_diff: Outdoor-air difference at the first milestone (°F).
        :type outdoor_sp_diff: float
        :param label: Mode label for logging.
        :type label: str
        :return: ``(k1, k2)``, or None if the morning is unusable.
        :rtype: tuple[float, float] | None
        """
        time_diff, temp_diff = get_time_temp_diff(htr, self.setpoint_deviation)
        if not time_diff:
            _log.debug("Siemens %s: time_diff is zero.", label)
            return None

        if np.isnan(get_time_target(data, 1.0)):
            _log.debug("Siemens %s: time_one_degree is NaN.", label)
            return None

        coeff1 = (time_diff / temp_diff) / 60.0
        coeff2 = (
            (time_diff / 60.0 - coeff1 * zone_sp_diff)
            * self._OSP_SCALE_FACTOR
            / (outdoor_sp_diff * zone_sp_diff)
        )

        model_label = f"Siemens {label}"
        if not _coefficients_are_valid({"coeff1": coeff1}, model_label=model_label):
            return None
        # coeff2 is clamped rather than rejected.
        if not np.isfinite(coeff2):
            _log.debug("%s: non-finite coeff2.", model_label)
            return None
        return coeff1, max(float(coeff2), 0.0)

    def calculate_prestart(self, data: pd.DataFrame) -> float:
        """
        Minutes needed from the EMA of ``k1`` / ``k2`` and today's gaps.

        :param data: Current device data; the last row is "now".
        :type data: pd.DataFrame
        :return: Prestart in minutes (not clamped; the manager clamps).
        :rtype: float
        """
        if data.empty:
            return self.earliest_start_time

        cond = ZoneConditions.latest(data)
        mode = self._determine_mode(cond)
        if mode is None:
            return self.latest_start_time

        k1_hist = getattr(self, f"{mode.prefix}1")
        k2_hist = getattr(self, f"{mode.prefix}2")
        if not k1_hist or not k2_hist:
            return self.earliest_start_time

        zsp = max(0.0, cond.zone_error(mode))
        osp = max(0.0, cond.outdoor_error(mode))
        k1, k2 = ema(k1_hist), ema(k2_hist)
        return (k1 * zsp + k2 * zsp * osp / self._OSP_SCALE_FACTOR) * 60.0 + self.adjust_time


class Johnson(OptimalStartZoneModel):
    """
    Quadratic model::

        minutes = k1 * gap**2 + k2

    ``k2`` is the time to recover the first 1 °F (a fixed start-up cost) and
    ``k1`` scales the rest of the recovery with the square of the gap. Both are
    fitted per morning and averaged with an EMA.

    :ivar c1_list: Cooling ``k1`` history.
    :vartype c1_list: list
    :ivar c2_list: Cooling ``k2`` history.
    :vartype c2_list: list
    :ivar h1_list: Heating ``k1`` history.
    :vartype h1_list: list
    :ivar h2_list: Heating ``k2`` history.
    :vartype h2_list: list
    """

    STATE_FIELDS = ("c1_list", "c2_list", "h1_list", "h2_list")

    def __init__(self, config: OptimalStartConfig, schedule_fn: Callable):
        """
        :param config: Optimal-start configuration.
        :type config: OptimalStartConfig
        :param schedule_fn: Callable returning today's schedule.
        :type schedule_fn: Callable
        """
        super().__init__(config, schedule_fn)
        self.c1_list: list = []
        self.c2_list: list = []
        self.h1_list: list = []
        self.h2_list: list = []

    @property
    def is_trained(self) -> bool:
        """
        :return: True once a cooling or heating ``k1`` has been learned.
        :rtype: bool
        """
        return bool(self.c1_list or self.h1_list)

    def trained_for(self, mode: OperatingMode) -> bool:
        """
        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: True if both ``k1`` and ``k2`` have been learned for ``mode``.
        :rtype: bool
        """
        return bool(getattr(self, f"{mode.prefix}1_list")) and bool(getattr(self, f"{mode.prefix}2_list"))

    def _train_mode(self, data: pd.DataFrame, mode: OperatingMode) -> None:
        """
        Fit ``k1`` / ``k2`` for this morning and append them to the histories.

        :param data: Parsed recovery data (``parse_df`` output).
        :type data: pd.DataFrame
        :param mode: Operating mode of the recovery.
        :type mode: OperatingMode
        """
        initial_diff = ZoneConditions.from_row(data.iloc[0]).zone_error(mode)
        coeffs = self._fit_coefficients(data, initial_diff, mode.value)
        if coeffs is None:
            return

        p = mode.prefix
        self.record = _record()
        for idx, value in enumerate(coeffs, start=1):
            field = f"{p}{idx}"
            history = trim(getattr(self, f"{field}_list"), value, self.training_period_window)
            setattr(self, f"{field}_list", history)
            self.record[field] = value
            self.record[f"{field}_array"] = history

    def _fit_coefficients(
        self, data: pd.DataFrame, initial_diff: float, label: str
    ) -> Optional[tuple[float, float]]:
        """
        Coefficients for one morning's recovery.

        :param data: Parsed recovery data.
        :type data: pd.DataFrame
        :param initial_diff: Zone gap at the start of the recovery (°F).
        :type initial_diff: float
        :param label: Mode label for logging.
        :type label: str
        :return: ``(k1, k2)`` (both strictly positive), or None if unusable.
        :rtype: tuple[float, float] | None
        """
        htr = self._compute_heat_transfer_rate(data)
        if htr.empty:
            _log.debug("Johnson %s: HTR table is empty.", label)
            return None

        time_diff, temp_diff = get_time_temp_diff(htr, self.setpoint_deviation)
        if not time_diff:
            _log.debug("Johnson %s: time_diff is zero.", label)
            return None

        coeff2 = get_time_target(data, 1.0)
        if np.isnan(coeff2):
            _log.debug("Johnson %s: time_one_degree is NaN.", label)
            return None

        if (time_diff - coeff2) >= 1:
            coeff1 = (time_diff - coeff2) / (initial_diff ** 2)
        else:
            coeff1 = time_diff / (temp_diff * 10)

        if not _coefficients_are_valid(
            {"coeff1": coeff1, "coeff2": coeff2}, model_label=f"Johnson {label}"
        ):
            return None
        return coeff1, coeff2

    def calculate_prestart(self, data: pd.DataFrame) -> float:
        """
        Minutes needed: ``EMA(k1) * gap**2 + EMA(k2)``.

        :param data: Current device data; the last row is "now".
        :type data: pd.DataFrame
        :return: Prestart in minutes (not clamped; the manager clamps).
        :rtype: float
        """
        if data.empty:
            return self.earliest_start_time

        cond = ZoneConditions.latest(data)
        mode = self._determine_mode(cond)
        if mode is None:
            return self.latest_start_time

        k1_hist = getattr(self, f"{mode.prefix}1_list")
        k2_hist = getattr(self, f"{mode.prefix}2_list")
        if not k1_hist or not k2_hist:
            return self.earliest_start_time

        zsp = cond.zone_error(mode)
        return ema(k1_hist) * zsp * zsp + ema(k2_hist)


class Sbs(OptimalStartZoneModel):
    """
    First-order (exponential) decay model of the setpoint error.

    The deadband-adjusted setpoint error ``e`` is assumed to decay by a fixed
    factor each minute, ``e[t+1] = alpha * e[t]``. Alpha is fitted per morning by
    least squares on every minute from unit start (including compressor/fan
    off-cycles, as in the original sbs-application) and averaged with an EMA.
    The prediction is the minutes for today's error to decay to the allowable
    deviation::

        minutes = ln(deviation / error) / ln(alpha)

    clamped to [latest, earliest]. Alpha is learned separately for cooling and
    heating; the fit starts after the start-up transient and only real recoveries
    are used (see :meth:`_model_input`).

    :ivar alpha_list_cooling: Cooling alpha history (``[timestamp, alpha]``).
    :vartype alpha_list_cooling: list
    :ivar alpha_list_heating: Heating alpha history.
    :vartype alpha_list_heating: list
    :ivar alpha: Alpha used by the latest prediction (None before one is made).
    :vartype alpha: float | None
    :ivar sp_error_occ: Predicted error remaining at the earliest start horizon
        (diagnostic).
    :vartype sp_error_occ: float
    """

    STATE_FIELDS = ("alpha_list_cooling", "alpha_list_heating")
    _ALPHA_LOWER = 0.001
    _ALPHA_UPPER = 0.999
    _SBS_COLUMNS = ["zonetemperature", "coolingsetpoint", "heatingsetpoint"]
    _MAX_FILL_MINUTES = 5
    #: Minutes after unit start searched for the peak error (end of the start-up transient).
    _STARTUP_MINUTES = 20

    def __init__(self, config: OptimalStartConfig, schedule_fn: Callable):
        """
        :param config: Optimal-start configuration.
        :type config: OptimalStartConfig
        :param schedule_fn: Callable returning today's schedule.
        :type schedule_fn: Callable
        """
        super().__init__(config, schedule_fn)
        self.alpha_list_cooling: list = []
        self.alpha_list_heating: list = []
        self.alpha: Optional[float] = None          # alpha used by the latest prediction
        self.sp_error_occ: float = 0.0

    @property
    def is_trained(self) -> bool:
        """
        :return: True once an alpha has been learned for either mode.
        :rtype: bool
        """
        return bool(self.alpha_list_cooling or self.alpha_list_heating)

    def _history(self, mode: OperatingMode) -> list:
        """
        Alpha history for ``mode``.

        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: ``alpha_list_cooling`` or ``alpha_list_heating``.
        :rtype: list
        """
        return getattr(self, f"alpha_list_{mode.value}")

    def trained_for(self, mode: OperatingMode) -> bool:
        """
        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: True if an alpha has been learned for ``mode``.
        :rtype: bool
        """
        return bool(self._history(mode))

    def load_model(self, model_dict: dict[str, Any]) -> None:
        """
        Restore learned state.

        A legacy single ``alpha_list`` (saved before alpha was split by mode) seeds
        both the cooling and heating histories, so deployed agents keep what they
        learned.

        :param model_dict: Persisted state (may be None).
        :type model_dict: dict[str, Any]
        """
        super().load_model(model_dict)
        legacy = (model_dict or {}).get("alpha_list")
        if legacy and not self.is_trained:
            self.alpha_list_cooling = list(legacy)
            self.alpha_list_heating = list(legacy)

    # -- Training -----------------------------------------------------------

    def _errors(self, data: pd.DataFrame) -> np.ndarray:
        """
        Deadband-adjusted setpoint error per row.

        The error is measured from the setpoint midpoint and reduced by half the
        deadband, so it equals ``cooling SP - zone`` (negative) when the zone is above
        the cooling setpoint and ``heating SP - zone`` (positive) when below the
        heating setpoint, and 0 inside the deadband.

        :param data: Rows with zone temperature and both setpoints.
        :type data: pd.DataFrame
        :return: Error per row (°F); NaN where data is missing.
        :rtype: np.ndarray
        """
        midpoint = (data["coolingsetpoint"] + data["heatingsetpoint"]) / 2.0
        deadband = data["coolingsetpoint"] - data["heatingsetpoint"]
        return self._apply_deadband(
            (midpoint - data["zonetemperature"]).to_numpy(dtype=float), deadband.to_numpy(dtype=float)
        )

    def _model_input(
        self, window: pd.DataFrame, fan_on: pd.DataFrame, mode: OperatingMode
    ) -> pd.DataFrame:
        """
        Minute-by-minute data for the alpha fit.

        Every minute from the first fan start to the end of the window, including
        compressor/fan off cycles; short sample gaps (up to ``_MAX_FILL_MINUTES``) are
        forward filled, longer gaps stay NaN and are skipped by the fit.

        The fit starts at the largest error within the first ``_STARTUP_MINUTES``:
        some zone sensors keep moving away from setpoint for a few minutes after the
        fan starts, which would otherwise read as alpha > 1. Mornings whose error at
        that point is within the allowable deviation are not a real recovery and are
        not used (they would fit alpha ~ 1).

        :param window: All rows of the training window (local time).
        :type window: pd.DataFrame
        :param fan_on: Window rows with the supply fan on.
        :type fan_on: pd.DataFrame
        :param mode: Detected operating mode (not used: the error sign carries it).
        :type mode: OperatingMode
        :return: 1-minute rows from the peak error onward; empty if not usable.
        :rtype: pd.DataFrame
        """
        empty = window.iloc[0:0][self._SBS_COLUMNS]
        first_on = window.index[window["supplyfanstatus"] != 0][0]
        started = window.loc[first_on:, self._SBS_COLUMNS]
        started = started.resample("1min").mean().ffill(limit=self._MAX_FILL_MINUTES)

        head = np.abs(self._errors(started)[: self._STARTUP_MINUTES])
        if not np.isfinite(head).any():
            return empty
        peak = int(np.nanargmax(head))
        if head[peak] <= self.setpoint_deviation:
            _log.debug("SBS: no recovery needed (error %.2f) - not training.", head[peak])
            return empty
        return started.iloc[peak:]

    def _train_mode(self, data: pd.DataFrame, mode: OperatingMode) -> None:
        """
        Fit this morning's alpha and append it to the history for ``mode``.

        Least squares of ``e[t] = alpha * e[t-1]`` (one step = one minute), with
        ``e[-1] = 0``; pairs touching a data gap are skipped. The estimate is clipped
        to [``_ALPHA_LOWER``, ``_ALPHA_UPPER``]; no update if the fit is degenerate.

        :param data: Output of :meth:`_model_input`.
        :type data: pd.DataFrame
        :param mode: Operating mode of the recovery.
        :type mode: OperatingMode
        """
        errors = self._errors(data)
        # Least-squares fit of e[t] = alpha * e[t-1] (one step = one minute),
        # with e[-1] = 0; pairs touching a data gap are skipped.
        previous = np.concatenate(([0.0], errors[:-1]))
        valid = np.isfinite(previous) & np.isfinite(errors)
        sum_x2 = float(np.sum(previous[valid] ** 2))
        sum_xy = float(np.sum(previous[valid] * errors[valid]))

        new_alpha = None
        history = self._history(mode)
        if sum_x2 > 0 and sum_xy > 0:
            new_alpha = float(np.clip(sum_xy / sum_x2, self._ALPHA_LOWER, self._ALPHA_UPPER))
            history = trim(history, new_alpha, self.training_period_window)
            setattr(self, f"alpha_list_{mode.value}", history)
        self.record = _record(mode=mode.value, new_alpha=new_alpha,
                              alpha=ema(history) if history else None, alpha_array=history)

    # -- Prediction ---------------------------------------------------------

    def calculate_prestart(self, data: pd.DataFrame) -> float:
        """
        Minutes for today's error to decay to the allowable deviation.

        :param data: Current device data; the last row is "now".
        :type data: pd.DataFrame
        :return: Prestart in minutes, clamped to [latest, earliest].
        :rtype: float
        """
        if data.empty:
            return self.earliest_start_time

        cond = ZoneConditions.latest(data)
        mode = self._determine_mode(cond)
        if mode is None:
            return self.latest_start_time
        history = self._history(mode)
        if not history:
            return self.earliest_start_time            # not trained for this mode yet

        self.alpha = ema(history)
        midpoint = (cond.cooling_setpoint + cond.heating_setpoint) / 2.0
        deadband = cond.cooling_setpoint - cond.heating_setpoint
        error = abs(float(self._apply_deadband(midpoint - cond.zone_temperature, deadband)))

        if error > 0:
            prestart = np.log(self.setpoint_deviation / error) / np.log(self.alpha)
        else:
            prestart = self.latest_start_time
        prestart = float(np.clip(prestart, self.latest_start_time, self.earliest_start_time))
        self.sp_error_occ = error * np.power(self.alpha, self.earliest_start_time)
        _log.debug("SBS %s: error=%.2f  alpha=%.4f  prestart=%.1f", mode.value, error, self.alpha, prestart)
        return prestart

    @staticmethod
    def _apply_deadband(error, deadband):
        """
        Zero the error inside ±deadband/2 and shrink it by deadband/2 outside.

        Works element-wise on arrays and on scalars.

        :param error: Error from the setpoint midpoint (°F).
        :param deadband: Cooling SP - heating SP (°F).
        :return: Deadband-adjusted error (float for scalar input, else ndarray).
        """
        error = np.asarray(error, dtype=float)
        half = np.asarray(deadband, dtype=float) / 2.0
        result = np.where(np.abs(error) <= half, 0.0, error - np.sign(error) * half)
        return result if result.ndim else float(result)


class RecoveryRegression(OptimalStartZoneModel):
    """
    Learns the start time directly from observed recoveries.

    Each morning gives one observation for the mode the zone needed: the gap to
    setpoint and the outdoor-air difference at unit start, and the minutes the
    zone took to get within ``TARGET_DEVIATION`` of setpoint. The prediction is a
    robust (Huber) least-squares fit over the last ``HISTORY`` observations::

        minutes = a + b * gap + c * gap * oat_diff + margin

    * ``a`` - fixed start-up overhead (fan/compressor start, final approach);
    * ``b`` - minutes per °F of gap in mild weather;
    * ``c`` - how outdoor air changes the cost of each degree;
    * ``oat_diff`` - OAT - SP for cooling, SP - OAT for heating (larger = harder);
    * ``margin`` - the ``MARGIN_QUANTILE`` quantile of the fit residuals: a
      learned bias toward starting slightly early, since a late start costs
      comfort.

    Without a usable outdoor-air reading (missing or stale), or with fewer than
    ``MIN_OBSERVATIONS`` observations that have one, the fit drops the outdoor
    term (``a + b * gap``). The model is untrained for a mode until it has
    ``MIN_OBSERVATIONS`` observations. Coefficients are recomputed at every
    prediction, so the observation history is the complete state.

    The model keeps one history for every day (``SEPARATE_WEEKEND_MODEL`` is
    False). Training must run at least ``OBSERVATION_HORIZON_MINUTES`` after
    occupancy; a recovery not finished by then is recorded at the horizon.

    :cvar TARGET_DEVIATION: Target closeness to setpoint (°F) that ends a recovery.
    :cvar HISTORY: Observations kept per mode.
    :cvar MIN_OBSERVATIONS: Observations needed before the model is trained.
    :cvar MARGIN_QUANTILE: Residual quantile added as the early-start margin.
    :cvar OBSERVATION_HORIZON_MINUTES: Minutes after occupancy a recovery is
        observed.
    :ivar observations_cooling: Cooling observations,
        ``[timestamp, gap, oat_diff or None, minutes]``, oldest first.
    :vartype observations_cooling: list
    :ivar observations_heating: Heating observations (same layout).
    :vartype observations_heating: list
    """

    STATE_FIELDS = ("observations_cooling", "observations_heating")
    # One history for every day: Mondays are just more observations (the gap and
    # outdoor-air terms describe the conditions); a weekend-only copy would see
    # too few mornings to fit.
    SEPARATE_WEEKEND_MODEL = False
    TARGET_DEVIATION = 0.5
    HISTORY = 20
    MIN_OBSERVATIONS = 5
    MARGIN_QUANTILE = 0.7
    OBSERVATION_HORIZON_MINUTES = 90
    _HUBER_K = 1.345

    def __init__(self, config: OptimalStartConfig, schedule_fn: Callable):
        """
        :param config: Optimal-start configuration.
        :type config: OptimalStartConfig
        :param schedule_fn: Callable returning today's schedule.
        :type schedule_fn: Callable
        """
        super().__init__(config, schedule_fn)
        # Each observation: [timestamp, gap (F), oat_diff (F) or None, minutes]
        self.observations_cooling: list = []
        self.observations_heating: list = []

    def _history(self, mode: OperatingMode) -> list:
        """
        Observation history for ``mode``.

        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: ``observations_cooling`` or ``observations_heating``.
        :rtype: list
        """
        return getattr(self, f"observations_{mode.value}")

    @property
    def is_trained(self) -> bool:
        """
        :return: True once either mode has ``MIN_OBSERVATIONS`` observations.
        :rtype: bool
        """
        return any(self.trained_for(m) for m in OperatingMode)

    def trained_for(self, mode: OperatingMode) -> bool:
        """
        :param mode: Operating mode.
        :type mode: OperatingMode
        :return: True if ``mode`` has at least ``MIN_OBSERVATIONS`` observations.
        :rtype: bool
        """
        return len(self._history(mode)) >= self.MIN_OBSERVATIONS

    # -- Training -----------------------------------------------------------

    def train(self, data: pd.DataFrame, prestart: Optional[float] = None) -> None:
        """
        Record this morning's recovery as an observation.

        Uses the window from ``prestart`` minutes before occupancy to occupancy +
        ``OBSERVATION_HORIZON_MINUTES`` (see :meth:`_observe`) and appends the
        observation to the history for its mode, keeping the last ``HISTORY``.
        :attr:`record` holds the observation, or stays empty if none was recorded.

        :param data: Today's device data, time indexed.
        :type data: pd.DataFrame
        :param prestart: Minutes before occupancy the window starts (normally the
            committed prestart + a buffer); defaults to ``earliest_start_time``.
        :type prestart: float | None
        """
        self.record = {}
        if prestart is None:
            prestart = self.earliest_start_time
        if data is None or data.empty:
            return
        schedule = self._get_current_schedule()
        if schedule is None or schedule.start is None:
            return

        window = self._training_window(data, schedule, prestart, end_offset=self.OBSERVATION_HORIZON_MINUTES)
        observation = self._observe(window, schedule)
        if observation is None:
            return
        mode, gap, oat_diff, minutes = observation
        history = self._history(mode) + [[format_timestamp(dt.now()), gap, oat_diff, minutes]]
        setattr(self, f"observations_{mode.value}", history[-self.HISTORY:])
        self.record = _record(mode=mode.value, gap=gap, oat_diff=oat_diff, minutes=minutes,
                              observations=len(self._history(mode)))

    def _train_mode(self, data: pd.DataFrame, mode: OperatingMode) -> None:
        """
        Not used: observations are recorded by :meth:`train`.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError("RecoveryRegression records observations in train()")

    def _observe(self, window: pd.DataFrame, schedule):
        """
        This morning's observation, if it is a usable recovery.

        The unit start is the first supply-fan-on minute in the window; conditions
        are taken from the last reading before it. The recovery ends at the first
        minute the zone is within ``TARGET_DEVIATION`` of setpoint, or at the
        observation horizon if not reached.

        :param window: Training window (local time), through the horizon.
        :type window: pd.DataFrame
        :param schedule: Today's schedule (uses ``schedule.start``).
        :return: ``(mode, gap, oat_diff or None, minutes)``; None if the fan never
            ran, the zone reading is missing/stale, no recovery was needed, or the
            data ends before the horizon without reaching the target.
        :rtype: tuple[OperatingMode, float, float | None, float] | None
        """
        fan_on = window.index[window["supplyfanstatus"] != 0]
        if len(fan_on) == 0:
            return None
        start = fan_on[0]
        before = window[window.index < start].ffill()
        if before.empty:
            return None
        cond = ZoneConditions.from_row(before.iloc[-1])
        if not np.isfinite(cond.zone_temperature):
            return None                                   # stale / missing zone reading
        mode = self._determine_mode(cond)
        if mode is None:
            return None                                   # no recovery needed this morning

        occupancy = start.replace(hour=schedule.start.hour, minute=schedule.start.minute, second=0, microsecond=0)
        horizon = occupancy + pd.Timedelta(minutes=self.OBSERVATION_HORIZON_MINUTES)
        after = window[window.index >= start]
        if mode is OperatingMode.COOLING:
            hit = after[after["zonetemperature"] <= after["coolingsetpoint"] + self.TARGET_DEVIATION]
        else:
            hit = after[after["zonetemperature"] >= after["heatingsetpoint"] - self.TARGET_DEVIATION]
        if len(hit):
            minutes = (hit.index[0] - start).total_seconds() / 60.0
        elif after.index[-1] >= horizon - pd.Timedelta(minutes=5):
            minutes = (horizon - start).total_seconds() / 60.0   # not reached by the horizon
        else:
            return None                                   # data ends before the horizon: incomplete
        oat_diff = cond.outdoor_error(mode)
        return mode, float(cond.zone_error(mode)), (float(oat_diff) if np.isfinite(oat_diff) else None), float(minutes)

    # -- Prediction ---------------------------------------------------------

    def calculate_prestart(self, data: pd.DataFrame) -> float:
        """
        Minutes needed for today's gap and outdoor air.

        :param data: Current device data; the last row is "now".
        :type data: pd.DataFrame
        :return: Prestart in minutes, clamped to [latest, earliest]
            (``latest_start_time`` if no recovery is needed, ``earliest_start_time``
            if not trained for the mode).
        :rtype: float
        """
        if data.empty:
            return self.earliest_start_time
        cond = ZoneConditions.latest(data)
        mode = self._determine_mode(cond)
        if mode is None:
            return self.latest_start_time
        if not self.trained_for(mode):
            return self.earliest_start_time
        oat_diff = cond.outdoor_error(mode)
        prestart = self._fit_predict(self._history(mode), cond.zone_error(mode), oat_diff)
        return float(np.clip(prestart, self.latest_start_time, self.earliest_start_time))

    def _fit_predict(self, history: list, gap: float, oat_diff: float) -> float:
        """
        Fit the history and predict one morning, including the early margin.

        :param history: Observations for the mode.
        :type history: list
        :param gap: Today's zone gap (°F).
        :type gap: float
        :param oat_diff: Today's outdoor-air difference (°F); NaN if unavailable.
        :type oat_diff: float
        :return: Predicted minutes (unclamped).
        :rtype: float
        """
        obs = np.array([[o[1], np.nan if o[2] is None else o[2], o[3]] for o in history], dtype=float)
        with_oat = np.isfinite(oat_diff) and np.isfinite(obs[:, 1]).sum() >= self.MIN_OBSERVATIONS
        if with_oat:
            obs = obs[np.isfinite(obs[:, 1])]
            design = np.column_stack([np.ones(len(obs)), obs[:, 0], obs[:, 0] * obs[:, 1]])
            x = np.array([1.0, gap, gap * oat_diff])
        else:
            design = np.column_stack([np.ones(len(obs)), obs[:, 0]])
            x = np.array([1.0, gap])
        coef, residuals = self._huber_fit(design, obs[:, 2])
        return float(x @ coef + np.quantile(residuals, self.MARGIN_QUANTILE))

    @classmethod
    def _huber_fit(cls, design: np.ndarray, y: np.ndarray, iterations: int = 5):
        """
        Robust least squares (iteratively reweighted, Huber weights).

        Each pass fits weighted least squares, estimates the typical residual from
        the median absolute deviation, and down-weights observations whose residual
        exceeds ``_HUBER_K`` times that scale, so one unusual morning cannot pull the
        fit far.

        :param design: Design matrix (one row per observation).
        :type design: np.ndarray
        :param y: Observed minutes.
        :type y: np.ndarray
        :param iterations: Reweighting passes.
        :type iterations: int
        :return: ``(coefficients, residuals)`` of the final pass.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        weights = np.ones(len(y))
        for _ in range(iterations):
            sw = np.sqrt(weights)
            coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
            residuals = y - design @ coef
            scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals))) or 1.0
            k = cls._HUBER_K * scale
            weights = np.where(np.abs(residuals) <= k, 1.0, k / np.maximum(np.abs(residuals), 1e-9))
        return coef, residuals

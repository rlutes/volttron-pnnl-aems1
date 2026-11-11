"""Config-store entries the Manager reacts to (main_manager.config_store_handlers)."""
import gevent
import pytest

from manager.main_manager import LEGACY_SET_POINTS_ENTRY, SET_POINTS_ENTRY, config_store_handlers


class FakeProxy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("set_"):
            raise AttributeError(name)

        def record(contents, update_store=True):
            self.calls.append((name, contents, update_store))
            return True
        record.__name__ = name
        return record


SETPOINTS = {"UnoccupiedHeatingSetPoint": 60.0, "UnoccupiedCoolingSetPoint": 85.0, "DeadBand": 3, "OccupiedSetPoint": 71}


def test_the_stored_set_points_entry_is_applied_when_it_changes():
    # The manager stores, validates and restores setpoints under SET_POINTS_ENTRY,
    # so storing that entry must apply them right away (not at the next validation).
    proxy = FakeProxy()
    handlers = config_store_handlers(proxy)
    assert SET_POINTS_ENTRY == "set_points"
    handlers[SET_POINTS_ENTRY](SETPOINTS, update_store=False)
    assert proxy.calls == [("set_temperature_setpoints", SETPOINTS, False)]


def test_legacy_entry_is_applied_and_persisted_as_set_points():
    # Storing the legacy name must also update SET_POINTS_ENTRY (update_store=True),
    # otherwise the periodic validator re-applies the old setpoints.
    proxy = FakeProxy()
    handlers = config_store_handlers(proxy)
    handlers[LEGACY_SET_POINTS_ENTRY](SETPOINTS, update_store=False)
    assert proxy.calls == []                     # deferred: the store cannot be written from a config callback
    gevent.sleep(0)
    assert proxy.calls == [("set_temperature_setpoints", SETPOINTS, True)]


def test_every_entry_has_a_handler_in_apply_order():
    names = list(config_store_handlers(FakeProxy()))
    assert names[0] == "schedule"
    assert set(names) == {"schedule", "set_points", "temperature_setpoints", "holidays", "optimal_start",
                          "occupancy_overrides", "location", "control"}

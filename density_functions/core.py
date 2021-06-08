
from enum import Enum

class SimulatorEvent(Enum):
  BEFORE_CALL = 1
  CALL_RESULT = 2

class Simulator:

  def __init__(self, name, domain, fn):
    self.name = name
    self.domain = domain
    self._callbacks_by_event = {}
    self._simulate = fn

  def __call__(self, theta):
    self._notify_before(theta=theta)
    data = self._simulate(theta)
    self._notify_result(theta=theta, data=data)
    return self._simulate(theta)

  def add_callback(self, event: SimulatorEvent, callback):
    assert isinstance(event, SimulatorEvent)
    if event not in self._callbacks_by_event:
      self._callbacks_by_event[event] = []
    self._callbacks_by_event[event].append(callback)

  def _notify_before(self, theta):
    if SimulatorEvent.BEFORE_CALL not in self._callbacks_by_event:
      return
    for callback in self._callbacks_by_event[SimulatorEvent.BEFORE_CALL]:
      callback(theta)

  def _notify_result(self, theta, data):
    if SimulatorEvent.CALL_RESULT not in self._callbacks_by_event:
      return
    for callback in self._callbacks_by_event[SimulatorEvent.CALL_RESULT]:
      callback(theta, data)
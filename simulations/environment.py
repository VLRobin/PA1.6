import math
from dataclasses import dataclass

@dataclass
class Environment:
    base: float = 5.0
    amplitude: float = 2.0
    period_s: float = 7200.0
    door_start_s: float = 1200.0
    door_duration_s: float = 120.0
    door_exchange_tau_s: float = 600.0

    def T_out(self, t: float) -> float:
        ''' Calculate the outside temperature at time t (seconds).
            The temperature follows a sinusoidal pattern with a specified base, amplitude, and period.
            Additionally, there is a temporary drop in temperature to simulate a door opening event.
            t: Time in seconds.
            Returns the outside temperature in degrees Celsius.
        '''
        if self.period_s <= 0.0:
            raise ValueError("period_s must be positive")

        return self.base + self.amplitude * math.sin(
            2.0 * math.pi * t / self.period_s
        )

    def is_door_open(self, t: float) -> bool:
        """Return whether the door is open at time t."""
        return self.door_start_s <= t < self.door_start_s + self.door_duration_s

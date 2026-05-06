import numpy as np

CC_CV_THRESHOLD = 0.80  # SoC fraction where taper begins

class EV:
    def __init__ (self, ev_id: str, battery_capacity_kwh: float, max_power_kw: float, cc_cv: bool = True):
        # EV specifications
        self.ev_id = ev_id
        self.battery_capacity_kwh = battery_capacity_kwh
        self.max_power_kw = max_power_kw
        self.cc_cv = cc_cv

        # Session variables
        self.current_energy_kwh = 0.0
        self.target_energy_kwh = 0.0
        self.arrival_time = 0
        self.departure_time = 0

    def reset_session(self, arrival_time: int, departure_time: int, initial_soc: float, target_soc: float):
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.current_energy_kwh = initial_soc * self.battery_capacity_kwh
        self.target_energy_kwh = target_soc * self.battery_capacity_kwh

    def transfer_energy(self, requested_power_kw: float) -> float:
        realized_power_kw = np.clip(requested_power_kw, -self.max_power_kw, self.max_power_kw)

        # CC-CV taper: only affects charging (positive power) above threshold
        if self.cc_cv and realized_power_kw > 0:
            soc = self.current_energy_kwh / self.battery_capacity_kwh
            if soc > CC_CV_THRESHOLD:
                taper = max(0.0, (1.0 - soc) / (1.0 - CC_CV_THRESHOLD))
                realized_power_kw *= taper

        if realized_power_kw > 0:
            # CHARGING: Cannot exceed maximum battery capacity
            space_left = self.battery_capacity_kwh - self.current_energy_kwh
            actual_energy_transfer = min(realized_power_kw, space_left)
        else:
            # DISCHARGING: Cannot discharge below 0 kWh
            actual_energy_transfer = max(realized_power_kw, -self.current_energy_kwh)

        self.current_energy_kwh += actual_energy_transfer
        return actual_energy_transfer
    
    def get_soc(self) -> float:
        """Returns current State of Charge as a percentage (0-100)."""
        return (self.current_energy_kwh / self.battery_capacity_kwh) * 100.0

    def get_target_soc(self) -> float:
        """Returns target State of Charge as a percentage (0-100)."""
        return (self.target_energy_kwh / self.battery_capacity_kwh) * 100.0

    def get_energy_deficit(self) -> float:
        """Returns how many kWh are still needed to reach the user's target. Zero if at or above target."""
        return max(0.0, self.target_energy_kwh - self.current_energy_kwh)
    
    def get_soc_delta(self) -> float:
        """Returns the percentage difference between current SoC and target SoC."""
        return max(0.0, self.get_target_soc() - self.get_soc())

    def get_time_remaining(self, current_time: int) -> int:
        """Returns timesteps left until the user unplugs."""
        return max(0, self.departure_time - current_time)
    def get_arrival_time(self) -> int:
        """Returns the arrival time of the EV."""
        return self.arrival_time

class TeslaModel3LR(EV):
    def __init__(self, ev_id: str, cc_cv: bool = True):
        super().__init__(ev_id, battery_capacity_kwh=75.0, max_power_kw=11.0, cc_cv=cc_cv)

class TeslaModelS(EV):
    def __init__(self, ev_id: str, cc_cv: bool = True):
        super().__init__(ev_id, battery_capacity_kwh=100.0, max_power_kw=16.5, cc_cv=cc_cv)

class NissanLeaf(EV):
    def __init__(self, ev_id: str, cc_cv: bool = True):
        super().__init__(ev_id, battery_capacity_kwh=40.0, max_power_kw=6.6, cc_cv=cc_cv)

class VolvoXC90PHEV(EV):
    def __init__(self, ev_id: str, cc_cv: bool = True):
        super().__init__(ev_id, battery_capacity_kwh=11.6, max_power_kw=3.7, cc_cv=cc_cv)

class RenaultZoe(EV):
    def __init__(self, ev_id: str, cc_cv: bool = True):
        super().__init__(ev_id, battery_capacity_kwh=52.0, max_power_kw=22.0, cc_cv=cc_cv)

class CustomEV(EV):
    def __init__(self, ev_id, battery_capacity_kwh = 100.0, max_power_kw = 10.0, cc_cv: bool = True):
        super().__init__(ev_id, battery_capacity_kwh, max_power_kw, cc_cv=cc_cv)

EV_MODELS = {
    "Tesla Model 3 LR": TeslaModel3LR,
    "Tesla Model S": TeslaModelS,
    "Nissan Leaf": NissanLeaf,
    "Volvo XC90 PHEV": VolvoXC90PHEV,
    "Renault Zoe": RenaultZoe,
    "CustomEV": CustomEV

}

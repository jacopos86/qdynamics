import numpy as np
from src.io_module.input.core import AbstractInput
from src.io_module.input.base_inp_variables import EvolutionParameters
from src.utilities.log import log

#
#   different RT input classes
#

class InputRT(AbstractInput):
    """base class needed for all real time calculations"""
    _ALLOWED_ENVELOPES = {"gaussian", "cw", "delta", "gaussian+cw"}
    def __init__(self):
        # time parameters
        self.evol_params = None
        # ext. pot. params
        self.ext_pot_params = None
    # parsing data
    def _parse_data(self):
        """Parse YAML file and extract dynamical variables"""
        # time evolution parameters
        self.evol_params = EvolutionParameters.from_dict(self._data.get("evolution_params"))
        # external vec. pot. parameters
        ext_Vecpot_dict = self._data.get("external_vec_pot")
        if ext_Vecpot_dict is not None:
            enabled = ext_Vecpot_dict.get("enabled", False)
            # --- external vector potential ---
            # --- envelope --------------------
            env_dict = ext_Vecpot_dict.get("envelope", {})
            time_envelope = self._parse_time_envelope_function(env_dict)
            # ---- amplitude (A0) ----
            ampl = self.parse_quantity_from_dict(
                input_dict=ext_Vecpot_dict.get("amplitude"),
                required_keys=("units", "value"),
                desc="vector potential amplitude"
            )
            self.ext_Vecpot_params = {
                "enabled": enabled,
                "envelope": time_envelope,
                "amplitude": ampl,
            }
    # parse time envelope
    def _parse_time_envelope_function(self, env_dict):
        if not isinstance(env_dict, dict):
            log.error("envelope must be a dictionary")
        env_type = env_dict.get("type")
        if env_type is None:
            log.error("envelope type not defined")
        t0 = self.parse_quantity_from_dict(
            input_dict=env_dict.get("t0"),
            required_keys=("units", "value"),
            desc="envelope t0"
        )
        sigma_t = self.parse_quantity_from_dict(
            input_dict=env_dict.get("sigma_t"),
            required_keys=("units", "value"),
            desc="envelope sigma_t"
        )
        omega_d = self.parse_quantity_from_dict(
            input_dict=env_dict.get("omega_d"),
            required_keys=("units", "value"),
            desc="envelope omega_d"
        )
        phase = env_dict.get("phase", 0.0)
        return {
            "type": env_type,
            "t0": t0,
            "sigma_t": sigma_t,
            "omega_d": omega_d,
            "phase": phase
        }
    # validation
    def _validate(self):
        # RT mode
        self._check_RTmode_consistency()
        if not isinstance(self.TR_SYM, bool):
            log.error(f"TR_SYM either true or false - got: {self.TR_SYM}")
        # RT parameters
        self.evol_params._validate()
        # --- external vector potential validation ---
        self._validate_extVecPot()
        # --- external ph. drive validation ---
        self._validate_extPhDrive()
    # check consistency dynamical mode
    def _check_RTmode_consistency(self):
        modes = {
            "ee": (self.ee_mode, self._EE_MODES),
            "eph": (self.eph_mode, self._EPh_MODES),
            "erad": (self.erad_mode, self._ERAD_MODES),
        }
        for name, (val, allowed) in modes.items():
            if val not in allowed:
                log.error(f"{name}={val!r} not allowed. Must be one of {allowed}")
    def _validate_extVecPot(self):
        if self.ext_Vecpot_params is not None:
            # ---- enabled ----
            if not isinstance(self.ext_Vecpot_params.get("enabled"), bool):
                log.error("external_vec_pot.enabled must be boolean")
            if self.ext_Vecpot_params["enabled"]:
                # ---- envelope ----
                env = self.ext_Vecpot_params.get("envelope")
                if not isinstance(env, dict):
                    log.error("external_vec_pot.envelope must be a dictionary")
                env_type = env.get("type")
                if env_type is None:
                    log.error("env_type must be given")
                if env_type.lower() not in self._ALLOWED_ENVELOPES:
                    log.error(
                        f"Invalid envelope type {env_type!r}. "
                        f"Allowed: {self._ALLOWED_ENVELOPES}"
                    )
                # t0 → time
                t0 = env.get("t0")
                if t0 is not None:
                    if not isinstance(t0, Q_):
                        log.error("envelope.t0 must be a Quantity")
                    if not t0.check("second"):
                        log.error(f"t0 must have dimensions of time, got {t0.units}")
                # sigma_t → time
                sigma_t = env.get("sigma_t")
                if sigma_t is not None:
                    if not isinstance(sigma_t, Q_):
                        log.error("envelope.sigma_t must be a Quantity")
                    if not sigma_t.check("second"):
                        log.error(f"sigma_t must have dimensions of time, got {sigma_t.units}")
                # omega_d → frequency
                omega_d = env.get("omega_d")
                if omega_d is not None:
                    if not isinstance(omega_d, Q_):
                        log.error("envelope.omega_d must be a Quantity")
                    if not omega_d.check("1 / second"):
                        log.error(f"omega_d must have dimensions of frequency, got {omega_d.units}")
                # ---- A0 ----
                ampl = self.ext_Vecpot_params.get("amplitude")
                if not isinstance(ampl, Q_):
                    log.error("ampl must be a Quantity")
                # expected dimension: eV * ps / length
                if not ampl.check("eV * ps / meter"):
                    log.error(
                        f"A0 must have dimensions of eV * ps / length, got {ampl.units}"
                    )
                # if vector → must be 3D
                if isinstance(ampl.magnitude, np.ndarray):
                    if ampl.magnitude.shape != (3,):
                        log.error("A0 vector must have shape (3,)")
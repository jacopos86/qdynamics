import os
from src.utilities.input_parser import parser
from src.utilities.log import log
from src.io_module.input.input_Builder import InputCombiner
from src.io_module.input.input_Base import BaseInput
from src.io_module.input.input_ElecSys import ElecSysInput
from src.io_module.input.input_QuantEncode import QuantumEncodingInput
from src.io_module.input.input_VQE import QuantumVQEInput
from src.io_module.input.input_OutData import OutDataInput
from src.io_module.input.input_Psi4 import Psi4Input
from src.io_module.input.input_HHModel import HHModelInput

# parameters proxy class

class param_proxy:
    def __init__(self):
        self._real_p = None

    def set_input_arguments(self):
        is_testing = os.getenv('PYDEPHASING_TESTING') == '1'
        if is_testing is True:
            args = parser.parse_args(args=[])
        else:
            args = parser.parse_args()
        return args
    
    def _init(self):
        # input parameters object initialization
        args = self.set_input_arguments()
        co = parser.parse_args().co[0]
        ct1 = args.ct1[0]
        if co == 'spin-sys':
            if ct1 == "RT":
                self._real_p = Q_real_time_input()
            elif ct1 == "GS":
                self._real_p = Q_ground_state_input()
            else:
                log.error(f"ONLY RT / GS IMPLEMENTED")
        elif co == 'elec-sys':
            ct2 = args.ct2
            if ct1 == "RT":
                self._real_p = Q_real_time_input()
            elif ct1 == "GS":
                if ct2 == "PSI4":
                    self._real_p = InputCombiner(
                        BaseInput,
                        ElecSysInput,
                        QuantumEncodingInput,
                        QuantumVQEInput,
                        Psi4Input
                    )
                elif ct2 == "PYSCF":
                    self._real_p = Q_pyscf_input()
                elif ct2 == "HH_MODEL":
                    self._real_p = InputCombiner(
                        BaseInput,
                        HHModelInput,
                        QuantumEncodingInput,
                        QuantumVQEInput,
                        OutDataInput
                    )
                else:
                    log.error(f"Unknown ct2 value: {ct2!r}")
            else:
                log.error(f"ONLY RT / GS IMPLEMENTED")
        else:
            log.error(f"ONLY SPIN / ELEC SYSTEMS IMPLEMENTED")
        if self._real_p is None:
            log.error(f"Failed to initialize parameter object with ct1={ct1!r}, ct2={ct2!r}")
        self._real_p.sep = "*"*94

    def __getattr__(self, attr):
        if self._real_p is None:
            self._init()
        return getattr(self._real_p, attr)
    
p = param_proxy()
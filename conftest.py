import sys
import types


if 'jsbsim' not in sys.modules:
    sys.modules['jsbsim'] = types.SimpleNamespace(FGFDMExec=object)

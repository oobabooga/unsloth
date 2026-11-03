from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location(
    "nf4_pytest_criteria", Path(__file__).with_name("pytest_no_regression.py")
)
_pytest = module_from_spec(_spec)
_spec.loader.exec_module(_pytest)

MODE = _pytest.MODE
TITLE = "NF4 decode ROCm fallback, base versus head"
gates = _pytest.gates
head_is_worse = _pytest.head_is_worse
table = _pytest.table
NEEDS = ["rocm", "gpu"]

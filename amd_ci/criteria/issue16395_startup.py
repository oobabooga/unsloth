# SPDX-License-Identifier: AGPL-3.0-only
from pytest_no_regression import gates, head_is_worse, table

TITLE = "Injected startup recovery on Windows AMD, base versus head"
MODE = "regression"
NEEDS = ["windows", "vulkan", "nvidia", "rocm", "multi_gpu", "mlx"]

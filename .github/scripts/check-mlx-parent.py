# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise parent MLX computation before and after an isolated Studio tool."""
import os
import tempfile
import mlx.core as mx

os.environ["UNSLOTH_STUDIO_HOME"] = tempfile.mkdtemp(prefix="mlx-parent-")
metal = mx.metal.is_available()
mx.set_default_device(mx.gpu if metal else mx.cpu)
def calculate():
    value = (mx.ones((32,32)) @ mx.ones((32,32))).sum()
    mx.eval(value)
    return value.item()
before = calculate()
from core.inference import tools
from core.inference.os_sandbox import capability_snapshot
capability = capability_snapshot()
assert capability.available, capability.reason
output = tools.execute_tool("python", {"code":"print('MLX_PARENT_TOOL_OK')"},
                            session_id="__LOCALID_mlx_parent", timeout=30)
assert "MLX_PARENT_TOOL_OK" in output, output
assert tools._last_tool_execution_record.os_isolation
after = calculate()
assert before == after == 32768
print({"mlx_parent_device": "Metal" if metal else "CPU", "before":before,"after":after,
       "tool_isolated":True})
if not metal:
    print("UNVERIFIED: this runner exposes no Metal device; CPU MLX and dispatch checks only")

# SPDX-License-Identifier: AGPL-3.0-only
import argparse
import ast
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    obs = {'state': args.state, 'cases': []}
    try:
        import torch
        obs.update(torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.is_available())
        if not obs['gpu']:
            raise RuntimeError('No accelerator available')
        obs['device'] = torch.cuda.get_device_name()
        source = args.checkout / 'unsloth/models/llama.py'
        names = {'_round_rope_size', '_PerDeviceRopeCache', 'LlamaRotaryEmbedding'}
        nodes = [n for n in ast.parse(source.read_text(encoding='utf-8')).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        ns = dict(torch=torch, math=math, DEVICE_COUNT=torch.cuda.device_count(),
                  DEVICE_TYPE_TORCH='cuda', get_current_device=lambda: 0)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
        for dtype in (torch.float32, torch.float16):
            row = {'dtype': str(dtype)}
            try:
                torch._dynamo.reset()
                rope = ns['LlamaRotaryEmbedding'](dim=8, max_position_embeddings=16)
                reference = ns['LlamaRotaryEmbedding'](dim=8, max_position_embeddings=16)
                x = torch.ones(1, 4, 8, device='cuda', dtype=dtype, requires_grad=True)
                ids = torch.tensor([[0, 15, 16, 32768]], device='cuda')
                cos, sin = torch.compile(rope, fullgraph=True, dynamic=True)(x, ids)
                rc, rs = reference(x, ids)
                row['max_error'] = float(torch.maximum((cos-rc).abs().max(), (sin-rs).abs().max()).detach())
                ((cos+sin)*x).sum().backward()
                row.update(executed=True, finite_grad=bool(torch.isfinite(x.grad).all()),
                           grad_error=float((x.grad-(cos+sin)).abs().max().detach()))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            obs['cases'].append(row)
    except Exception as exc:
        obs['error'] = f'{type(exc).__name__}: {exc}'
    args.out.write_text(json.dumps(obs, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

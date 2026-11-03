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
        gemma_source = args.checkout / 'unsloth/models/gemma.py'
        gemma_nodes = [n for n in ast.parse(gemma_source.read_text(encoding='utf-8')).body
                       if isinstance(n, ast.ClassDef) and n.name in {'GemmaFixedRotaryEmbedding', 'GemmaFixedLinearScalingRotaryEmbedding'}]
        exec(compile(ast.Module(body=gemma_nodes, type_ignores=[]), str(gemma_source), 'exec'), ns)
        for kind, dtype in __import__('itertools').product(('LlamaRotaryEmbedding', 'GemmaFixedRotaryEmbedding', 'GemmaFixedLinearScalingRotaryEmbedding'), (torch.float32, torch.float16)):
            row = {'kind': kind, 'dtype': str(dtype)}
            try:
                torch._dynamo.reset()
                kwargs = dict(dim=128, max_position_embeddings=16)
                if kind == "GemmaFixedLinearScalingRotaryEmbedding":
                    kwargs["scaling_factor"] = 1.3
                rope = ns[kind](**kwargs)
                reference = ns[kind](**kwargs)
                x = torch.ones(1, 4, 128, device='cuda', dtype=dtype, requires_grad=True)
                ids = torch.tensor([[0, 15, 32767, 131071]], device='cuda')
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

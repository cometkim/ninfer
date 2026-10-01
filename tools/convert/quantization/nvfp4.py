"""Fork NVFP4_MAXABS_DIVISOR_RNE_V1 conversion method.

The global divisor belongs to the complete packed parent, not each logical slice
or streaming chunk. Values are FP32; block scales are E4M3FN and codes E2M1 RNE.
"""
import struct

import torch

from tools.convert.methods import AuxiliaryValue


def e2m1_rne_codes(values):
    magnitude = values.abs()
    codes = torch.zeros_like(values, dtype=torch.uint8)
    for boundary, tie_up in zip((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.),
                                (False, True, False, True, False, True, False)):
        codes += ((magnitude > boundary) | ((magnitude == boundary) & tie_up)).to(torch.uint8)
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def encode_block(values, divisor):
    rows, columns = values.shape
    blocks = (values.float() * divisor).reshape(rows, columns // 16, 16)
    scales = (blocks.abs().amax(dim=2) / 6.).clamp(max=448.).to(torch.float8_e4m3fn)
    decoded = scales.float()
    nonzero = decoded > 0
    ratios = torch.where(nonzero[..., None],
                         blocks / torch.where(nonzero, decoded, 1.)[..., None], 0.)
    codes = e2m1_rne_codes(ratios).reshape(rows, columns // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4), scales.view(torch.uint8)


def nvfp4_maxabs(request):
    if request.target.format != 'nvfp4' or len(request.target.shape) != 2:
        raise ValueError('nvfp4_maxabs requires an NVFP4 matrix')
    if request.parameters:
        raise ValueError('nvfp4_maxabs accepts no numerical parameters')
    n, k = request.target.shape
    if n % 128 or k % 16 or request.source_offsets[-1] != n * k:
        raise ValueError('NVFP4 parent requires complete 128-row and 16-column tiles')
    if request.rows_per_chunk <= 0:
        raise ValueError('rows_per_chunk must be positive')
    chunk = max(128, request.rows_per_chunk // 128 * 128)
    auxiliaries = {}
    for item in request.inputs:
        for use in item.uses:
            key = (*use, 'activation_input_divisor')
            if request.policies[use] == 'AllowA4':
                if key not in request.auxiliary_overrides:
                    raise ValueError(f'{use}: calibrated activation divisor required')
                auxiliaries[key] = request.auxiliary_overrides[key]

    def produce(output):
        maximum = 0.
        for begin in range(0, n, chunk):
            values = request.values(begin*k, min(n, begin+chunk)*k).float()
            if not torch.isfinite(values).all():
                raise ValueError('NVFP4 source contains NaN or infinity')
            maximum = max(maximum, values.abs().max().item())
        raw = struct.pack('<f', 2688. / maximum if maximum else 1.)
        AuxiliaryValue.activation_divisor(raw)  # same positive finite FP32 contract
        divisor = struct.unpack('<f', raw)[0]
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            values = request.values(begin*k, end*k).reshape(end-begin, k).to(request.device)
            codes, scales = encode_block(values, divisor)
            output.write_codes(begin, codes.cpu(), scales.cpu(), raw)

    return request.job(produce=produce, auxiliaries=auxiliaries)

"""V^3 PVM 볼륨(.pvm, 예: Carp/Lobster/Foot — Stefan Roettger의 Volume Library)을 NIfTI(.nii)로 변환한다.

PVM 파일은 보통 "DDS v3d" / "DDS v3e" 무손실 압축으로 감싸져 있다. 이 스크립트는 V^3 ddsbase.cpp의
readDDSfile 알고리즘(MSB 우선 비트열, 런 단위 비트 폭, 행 간 차분 예측, 성분 interleave 복원)을 그대로 따라
푼 뒤 PVM 헤더(PVM/PVM2/PVM3: 크기, voxel 간격, 성분 수)를 읽는다.

    1 성분 = uint8, 2 성분 = uint16(big-endian 저장) — 출력 .nii는 (X, Y, Z) 순서, voxel 간격은 헤더 값.

사용:
    .venv\\Scripts\\python.exe scripts\\pvm_to_nii.py <입력.pvm> [<입력2.pvm> ...] [--out-dir resources/Volume_Data]
출력: <out-dir>/<이름>_<X>x<Y>x<Z>_<uint8|uint16>.nii
"""
import argparse
import os
import re

import numpy as np

DDS_ID1, DDS_ID2 = b"DDS v3d\n", b"DDS v3e\n"
DDS_RL = 7
DDS_INTERLEAVE = 1 << 24


class BitReader:
    """MSB 우선 비트열(ddsbase의 big-endian 32bit 버퍼와 같은 순서)."""

    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
        self.pos = 0

    def read(self, n):
        if n == 0:
            return 0
        v = 0
        for b in self.bits[self.pos:self.pos + n]:
            v = (v << 1) | int(b)
        self.pos += n
        return v

    def read_many(self, count, n):
        """n비트 값 count개를 한 번에 읽는다(n=0이면 0)."""
        if n == 0:
            return np.zeros(count, dtype=np.int64)
        chunk = self.bits[self.pos:self.pos + count * n]
        if len(chunk) < count * n:
            raise ValueError("DDS 비트열이 예상보다 짧다")
        self.pos += count * n
        return chunk.reshape(count, n).astype(np.int64) @ (1 << np.arange(n - 1, -1, -1, dtype=np.int64))


def dds_decode_bits(code):
    return code + 1 if code >= 1 else code


def interleave_restore(data, skip, block):
    """ddsbase DDS_interleave(=deinterleave restore): 성분별로 모아 저장된 바이트를 원래 순서로 되돌린다."""
    if skip <= 1:
        return data
    out = data.copy()

    def restore(seg):
        res = np.empty_like(seg)
        pos = 0
        for i in range(skip):
            n = len(range(i, len(seg), skip))
            res[i::skip] = seg[pos:pos + n]
            pos += n
        return res

    if block == 0:
        return restore(data)
    size = skip * block
    k = len(data) // size
    for b in range(k):
        out[b * size:(b + 1) * size] = restore(data[b * size:(b + 1) * size])
    if len(data) - k * size:
        out[k * size:] = restore(data[k * size:])
    return out


def read_dds(raw):
    """DDS 압축을 푼 바이트(np.uint8)를 돌려준다. DDS가 아니면 None."""
    if raw.startswith(DDS_ID1):
        version = 1
    elif raw.startswith(DDS_ID2):
        version = 2
    else:
        return None
    br = BitReader(raw[len(DDS_ID1):])
    skip = br.read(2) + 1
    strip = br.read(16) + 1

    buf = np.zeros(1 << 22, dtype=np.int64)
    cnt, act = 0, 0
    while True:
        run = br.read(DDS_RL)
        if run == 0:
            break
        bits = dds_decode_bits(br.read(3))
        delta = br.read_many(run, bits) - ((1 << bits) // 2)
        if cnt + run > len(buf):
            buf = np.concatenate([buf, np.zeros(max(len(buf), run), dtype=np.int64)])
        idx = np.arange(cnt, cnt + run)
        if strip == 1 or cnt + run - 1 <= strip:
            # 모든 샘플이 단순 차분: act += delta
            vals = (act + np.cumsum(delta)) % 256
        elif cnt > strip and strip >= run:
            # 모든 샘플이 행 차분이고 참조(cnt-strip)가 이미 풀린 값: act += (d[i-strip]-d[i-strip-1]) + delta
            pred = buf[idx - strip] - buf[idx - strip - 1]
            vals = (act + np.cumsum(pred + delta)) % 256
        else:
            vals = np.empty(run, dtype=np.int64)
            a = act
            for j, c in enumerate(idx):
                if strip == 1 or c <= strip:
                    a += delta[j]
                else:
                    a += buf[c - strip] - buf[c - strip - 1] + delta[j]
                a %= 256
                buf[c] = a
                vals[j] = a
        buf[cnt:cnt + run] = vals
        act = int(vals[-1])
        cnt += run
    data = buf[:cnt].astype(np.uint8)
    return interleave_restore(data, skip, 0 if version == 1 else DDS_INTERLEAVE)


def read_pvm(path):
    """Returns: (volume (X,Y,Z) uint8|uint16, spacing (3,), info dict)"""
    raw = open(path, "rb").read()
    dec = read_dds(raw)
    data = dec.tobytes() if dec is not None else raw
    m = re.match(rb"PVM(\d?)\n", data)
    if not m:
        raise ValueError(f"PVM 헤더가 없다: {data[:16]!r}")
    version = int(m.group(1) or 1)
    lines, pos = [], m.end()
    need = 3 if version >= 2 else 2           # (크기, 간격, 성분) / (크기, 성분)
    for _ in range(need):
        end = data.index(b"\n", pos)
        lines.append(data[pos:end].decode("ascii"))
        pos = end + 1
    w, h, d = map(int, lines[0].split())
    spacing = tuple(map(float, lines[1].split())) if version >= 2 else (1.0, 1.0, 1.0)
    comps = int(lines[-1].split()[0])
    n = w * h * d * comps
    body = np.frombuffer(data, dtype=np.uint8, count=n, offset=pos)
    if comps == 1:
        vol = body.reshape(d, h, w)
    elif comps == 2:
        vol = body.view(">u2").reshape(d, h, w).astype(np.uint16)
    else:
        raise ValueError(f"성분 수 {comps}는 지원하지 않는다(1=8bit, 2=16bit)")
    extra = data[pos + n:].split(b"\0")
    info = {"version": version, "compressed": dec is not None, "components": comps,
            "notes": [s.decode("latin-1").strip() for s in extra if s.strip()]}
    return np.ascontiguousarray(vol.transpose(2, 1, 0)), np.asarray(spacing), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pvm", nargs="+")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "resources", "Volume_Data"))
    args = ap.parse_args()
    import sys
    import nibabel as nib
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.pvm:
        vol, spacing, info = read_pvm(path)
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, f"{name}_{'x'.join(map(str, vol.shape))}_{vol.dtype}.nii")
        img = nib.Nifti1Image(vol, np.diag([*spacing, 1.0]))
        img.header.set_zooms(tuple(spacing))
        nib.save(img, out)
        print(f"{os.path.basename(path)}: PVM{info['version']} {'DDS 압축' if info['compressed'] else '무압축'}, "
              f"shape {vol.shape}, {vol.dtype}, 간격 {tuple(spacing)}, 값 {vol.min()}~{vol.max()}")
        for s in info["notes"]:
            print(f"    {s}")
        print(f"    → {out}")


if __name__ == "__main__":
    main()

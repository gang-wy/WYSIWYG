"""받은 볼륨을 main.py / INR-DVR가 여는 NIfTI(.nii)로 변환한다.

입력 종류(자동 판별):
  - DICOM 폴더: 폴더 안 .dcm을 ImagePositionPatient(없으면 InstanceNumber) 순으로 쌓는다.
                값 = 저장값 × RescaleSlope + RescaleIntercept, 간격 = PixelSpacing + 인접 slice 위치 차이.
  - MetaImage(.mha/.mhd): DimSize, ElementType, ElementByteOrderMSB, ElementSpacing, ElementDataFile을 따른다.
  - PVM(.pvm): scripts/pvm_to_nii.py의 read_pvm을 쓴다.
출력 배열은 (X, Y, Z) 순서이고, 값이 정수면 저장값 범위에 맞는 정수형(uint8/uint16/int16)으로 둔다.

사용:
    .venv\\Scripts\\python.exe scripts\\convert_to_nii.py <입력> [<입력2> ...] [--out-dir resources/Volume_Data] [--name 이름]
출력: <out-dir>/<이름>_<X>x<Y>x<Z>_<dtype>.nii (이름 기본값 = 폴더/파일 이름)
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET_TYPES = {"MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_USHORT": "u2", "MET_SHORT": "i2",
             "MET_UINT": "u4", "MET_INT": "i4", "MET_FLOAT": "f4", "MET_DOUBLE": "f8"}


def read_dicom_folder(folder):
    import pydicom
    files = sorted(glob.glob(os.path.join(folder, "*.dcm"))) or sorted(
        f for f in glob.glob(os.path.join(folder, "*")) if os.path.isfile(f))
    ds = [pydicom.dcmread(f) for f in files]
    ds = [d for d in ds if "PixelData" in d]
    if not ds:
        raise ValueError(f"{folder}: 픽셀이 있는 DICOM이 없다")

    def pos(d):
        if "ImagePositionPatient" in d and "ImageOrientationPatient" in d:
            o = np.asarray(d.ImageOrientationPatient, dtype=float)
            return float(np.dot(np.cross(o[:3], o[3:]), np.asarray(d.ImagePositionPatient, dtype=float)))
        return float(getattr(d, "InstanceNumber", 0))
    ds.sort(key=pos)
    slices = [d.pixel_array.astype(np.float64) * float(getattr(d, "RescaleSlope", 1) or 1)
              + float(getattr(d, "RescaleIntercept", 0) or 0) for d in ds]
    vol = np.stack(slices, axis=-1)                       # (Rows, Cols, Z)
    vol = vol.transpose(1, 0, 2)                          # (X=Cols, Y=Rows, Z)
    row_sp, col_sp = (float(v) for v in getattr(ds[0], "PixelSpacing", [1, 1]))
    zs = np.diff([pos(d) for d in ds])
    z_sp = float(np.median(zs)) if len(zs) and np.median(zs) > 0 else float(getattr(ds[0], "SliceThickness", 1) or 1)
    info = {"slices": len(ds), "modality": getattr(ds[0], "Modality", "?")}
    return vol, np.array([col_sp, row_sp, z_sp]), info


def read_metaimage(path):
    hdr = {}
    for line in open(path, "r", encoding="latin-1"):
        if "=" in line:
            k, v = line.split("=", 1)
            hdr[k.strip()] = v.strip()
    dims = [int(x) for x in hdr["DimSize"].split()]
    dtype = np.dtype((">" if hdr.get("ElementByteOrderMSB", hdr.get("BinaryDataByteOrderMSB", "False")) == "True"
                      else "<") + MET_TYPES[hdr["ElementType"]])
    spacing = [float(x) for x in hdr.get("ElementSpacing", hdr.get("ElementSize", "1 1 1")).split()]
    data_file = hdr["ElementDataFile"]
    if data_file == "LOCAL":
        raw = open(path, "rb").read()
        data = np.frombuffer(raw[len(raw) - int(np.prod(dims)) * dtype.itemsize:], dtype=dtype)
    else:
        data = np.fromfile(os.path.join(os.path.dirname(path), data_file), dtype=dtype)
    vol = data.reshape(dims[::-1]).transpose(2, 1, 0)       # 파일은 X가 가장 빠름 → (X, Y, Z)
    return vol.astype(dtype.newbyteorder("=")), np.array(spacing), {}


def to_int_if_possible(vol):
    if vol.dtype.kind == "f" and np.all(np.mod(vol, 1) == 0):
        lo, hi = vol.min(), vol.max()
        for t in (np.uint8, np.uint16, np.int16, np.int32):
            if np.iinfo(t).min <= lo and hi <= np.iinfo(t).max:
                return vol.astype(t)
    return vol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "resources", "Volume_Data"))
    ap.add_argument("--name", help="출력 이름(입력이 하나일 때)")
    args = ap.parse_args()
    import nibabel as nib
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.inputs:
        path = os.path.abspath(path)
        ext = os.path.splitext(path)[1].lower()
        if os.path.isdir(path):
            vol, spacing, info = read_dicom_folder(path)
        elif ext in (".mha", ".mhd"):
            vol, spacing, info = read_metaimage(path)
        elif ext == ".pvm":
            from pvm_to_nii import read_pvm
            vol, spacing, info = read_pvm(path)
        else:
            raise ValueError(f"{path}: DICOM 폴더, .mha/.mhd, .pvm만 지원")
        vol = to_int_if_possible(np.ascontiguousarray(vol))
        name = args.name if args.name and len(args.inputs) == 1 else os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, f"{name}_{'x'.join(map(str, vol.shape))}_{vol.dtype}.nii")
        img = nib.Nifti1Image(vol, np.diag([*spacing, 1.0]))
        img.header.set_zooms(tuple(float(s) for s in spacing))
        nib.save(img, out)
        extra = ", ".join(f"{k} {v}" for k, v in info.items() if k in ("slices", "modality"))
        print(f"{os.path.basename(path)}: shape {vol.shape}, {vol.dtype}, 간격 {tuple(round(float(s), 4) for s in spacing)}, "
              f"값 {vol.min()}~{vol.max()}" + (f", {extra}" if extra else "") + f"\n    → {out}")


if __name__ == "__main__":
    main()

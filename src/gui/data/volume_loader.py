"""볼륨 파일(.nii/.nii.gz/.npy/.raw/.dat) 로드와 전처리(3차원화, NaN 제거, [0, 1] 정규화)."""

import numpy as np
import nibabel as nib

RAW_DTYPES = {
    'uint8': np.uint8, 'int8': np.int8,
    'uint16': np.uint16, 'int16': np.int16,
    'uint32': np.uint32, 'int32': np.int32,
    'float32': np.float32, 'float64': np.float64
}


def load_nifti(file_path):
    """Returns: (volume, voxel_spacing(헤더 zoom))"""
    nii_img = nib.load(file_path)
    return np.array(nii_img.get_fdata(), order='F'), nii_img.header.get_zooms()[:3]


def load_npy(file_path):
    """Returns: (volume, (1, 1, 1))"""
    return np.load(file_path), (1.0, 1.0, 1.0)


def load_raw(file_path, shape, dtype_str, endian, voxel_spacing):
    """raw 바이너리를 Fortran order로 reshape한다. 데이터가 shape보다 길면 자르고 짧으면 오류.

    Returns: (volume float32, voxel_spacing)
    """
    endian_char = {'little': '<', 'big': '>'}.get(endian, '<')
    dtype = np.dtype(RAW_DTYPES[dtype_str]).newbyteorder(endian_char)
    raw_data = np.fromfile(file_path, dtype=dtype)

    expected_size = int(np.prod(shape))
    if len(raw_data) < expected_size:
        raise ValueError(f"데이터가 부족합니다. 예상 {expected_size}, 실제 {len(raw_data)} — 차원을 확인하세요.")
    if len(raw_data) > expected_size:
        print(f"⚠️ raw 데이터가 예상보다 김({len(raw_data)} > {expected_size}) — 앞부분만 사용")
        raw_data = raw_data[:expected_size]

    return raw_data.reshape(shape, order='F').astype(np.float32), voxel_spacing


def process_volume(volume_data):
    """float32 Fortran 배열로 바꾸고 3차원으로 맞춘 뒤 NaN/inf 제거, [0, 1] min-max 정규화.

    1D는 정육면체로 reshape, 2D는 앞에 축 추가, 4D 이상은 앞 3축만 사용.
    Returns: (X, Y, Z) float32
    """
    volume_data = np.array(volume_data, dtype=np.float32, order='F')

    if volume_data.ndim == 1:
        cube_size = int(np.round(len(volume_data) ** (1 / 3)))
        if cube_size ** 3 != len(volume_data):
            raise ValueError(f"1차원 데이터를 3차원으로 변환할 수 없습니다. 길이: {len(volume_data)}")
        volume_data = volume_data.reshape(cube_size, cube_size, cube_size, order='F')
    elif volume_data.ndim == 2:
        volume_data = volume_data[np.newaxis, :, :]
    while volume_data.ndim > 3:
        volume_data = volume_data[..., 0]

    if any(dim < 2 for dim in volume_data.shape):
        raise ValueError(f"볼륨 차원이 너무 작습니다: {volume_data.shape}")

    if not np.all(np.isfinite(volume_data)):
        volume_data = np.nan_to_num(volume_data, nan=0.0, posinf=1.0, neginf=0.0)

    v_min, v_max = volume_data.min(), volume_data.max()
    if v_max > v_min:
        volume_data = (volume_data - v_min) / (v_max - v_min)

    print(f"   Volume {volume_data.shape}, 정규화 {volume_data.min():.3f} ~ {volume_data.max():.3f}")
    return np.asfortranarray(volume_data)


class VolumeLoader:
    """확장자에 맞는 로더로 읽고 process_volume으로 전처리한다."""

    def load(self, file_path, raw_params=None):
        """raw_params: raw 파일일 때 {'shape', 'dtype_str', 'endian', 'voxel_spacing'}

        Returns: (volume (X,Y,Z) float32 [0,1], voxel_spacing)
        """
        lower = file_path.lower()
        if lower.endswith(('.nii', '.nii.gz')):
            volume_data, voxel_spacing = load_nifti(file_path)
        elif lower.endswith('.npy'):
            volume_data, voxel_spacing = load_npy(file_path)
        elif lower.endswith(('.raw', '.dat')):
            if raw_params is None:
                raise ValueError("Raw 파일 로딩에는 파라미터가 필요합니다.")
            volume_data, voxel_spacing = load_raw(
                file_path, raw_params['shape'], raw_params['dtype_str'],
                raw_params['endian'], raw_params['voxel_spacing'])
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path}")

        return process_volume(volume_data), voxel_spacing

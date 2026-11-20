"""Stage A(ray 샘플)와 Stage B(voxel)가 공유하는 opacity / 투과율 계산 규칙."""
import numpy as np

# ray 하나를 박스 교차 구간 길이와 무관하게 나누는 샘플 수
RAY_NUM_SAMPLES = 256

# TF opacity의 단위 거리(world). 렌더러와 같아야 한다: VTK ScalarOpacityUnitDistance 기본값,
# Unity BasicDVRRendererFix의 _OpacityUnitDistance 모두 1.
OPACITY_UNIT_DISTANCE = 1.0

# visibility feature 열에 넣을 값: "V" = alpha × 앞쪽 투과율(논문 Eq.6), "T" = 앞쪽 투과율만.
# best 점수 S는 항상 V(화면에 보이는 양)로 계산한다.
# shortcut: 실험용 모듈 전역 스위치(run_gt_evaluation --visibility-feature), 채택되면 정의를 하나로 정리
VISIBILITY_FEATURE = "V"


def intensity_to_alpha(values, tf_lut, v_min, v_max):
    """intensity → TF opacity. 볼륨 min/max로 0~255 정규화 후 LUT 조회. Returns: values와 같은 shape"""
    norm_v = np.clip(((values - v_min) / (v_max - v_min + 1e-8) * 255), 0, 255).astype(int)
    return np.asarray(tf_lut)[norm_v]


def correct_opacity(alphas, step):
    """TF opacity → 길이 step 샘플의 opacity. 렌더러와 같은 보정 1 - (1 - opacity)^(step / 단위 거리).

    보정 없이 합성하면 step이 단위 거리와 다른 ray마다 TF를 바꿔 분석하는 것과 같다.
    """
    return 1.0 - np.power(1.0 - np.asarray(alphas, dtype=np.float64),
                          np.asarray(step, dtype=np.float64) / OPACITY_UNIT_DISTANCE)


def prev_transmittance(alphas):
    """마지막 축(ray 방향)을 따라 각 샘플 앞까지의 누적 투과율. 첫 샘플은 1.

    visibility(ΔO) = 보정 opacity × prev_transmittance (논문 Eq.6). alphas는 correct_opacity를 거친 값.
    """
    trans = np.cumprod(1.0 - alphas, axis=-1)
    prev = np.roll(trans, 1, axis=-1)
    prev[..., 0] = 1.0
    return prev


if __name__ == "__main__":
    # 자체 확인: 같은 두께(10)의 opacity 0.1 판은 step이 달라도 누적 불투명도 O가 같다(1 - 0.9^10)
    for n in (5, 25, 100):
        a = correct_opacity(np.full(n, 0.1), 10.0 / n)
        O = 1.0 - np.prod(1.0 - a)
        assert abs(O - (1 - 0.9 ** 10)) < 1e-9, (n, O)
    assert np.allclose(correct_opacity([0.3], 0.0), 0.0) and np.allclose(correct_opacity([0.3], 1.0), 0.3)
    print("sampling_rules self-check ok")

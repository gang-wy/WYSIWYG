"""스트로크(선) → SAM positive 포인트. Unity MobileSamProcessing.SampleStrokePoints와 같은 규칙."""

import numpy as np

# Unity SegController.samPointCount (모든 데이터셋 씬에서 4)
STROKE_SAM_POINT_COUNT = 4


def sample_stroke_points(stroke, count=STROKE_SAM_POINT_COUNT):
    """스트로크를 따라 호 길이 기준으로 균등하게 count개 점을 뽑는다(양 끝은 피해서 (i+0.5)/count 위치).

    stroke: [(x, y), ...] 화면 좌표
    Returns: [(x, y), ...] float — 스트로크가 비었으면 빈 목록, 점 하나거나 길이가 0이면 첫 점 하나
    """
    if not stroke or count <= 0:
        return []
    pts = np.asarray(stroke, dtype=np.float64).reshape(-1, 2)
    first = (float(pts[0, 0]), float(pts[0, 1]))
    if len(pts) == 1:
        return [first]
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = cumulative[-1]
    if total <= 0:
        return [first]

    result = []
    seg = 1
    for k in range(count):
        target = total * (k + 0.5) / count
        while seg < len(pts) - 1 and cumulative[seg] < target:
            seg += 1
        length = cumulative[seg] - cumulative[seg - 1]
        t = (target - cumulative[seg - 1]) / length if length > 0 else 0.0
        p = pts[seg - 1] + t * (pts[seg] - pts[seg - 1])
        result.append((float(p[0]), float(p[1])))
    return result

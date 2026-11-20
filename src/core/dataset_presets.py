"""데이터셋 프리셋: 볼륨 / INR feature / TF / 카메라 / SAM 포인트 파일 경로, 선택 항목 "clipping"
({'x'|'y'|'z': [min, max]}, 0~1 정규화 — TF 패널 Volume Clipping 슬라이더 값).

선택 항목 "rois" (TF 패널 Experiment 그룹 — INR vs intensity 비교 촬영용):
  {ROI 이름: {"points": SAM 포인트 파일, "another": 그 ROI를 측면에서 보는 카메라 파일(없으면 None)}}
  첫 항목이 "points"와 같은 기본 ROI. 기본 시점은 모든 ROI가 "camera"를 쓴다.

TF 패널의 Dataset Presets 버튼과 scripts/가 사용한다. raw 볼륨은 "raw"에 RawDataDialog.get_parameters()
형식의 파라미터를 둔다.

INR "*_gradfix.nii"는 gradient head 출력을 ReLU에서 Softplus로 바꿔 재학습한 feature, "inr_old"는 그 전 feature다.
원본은 INR-DVR resources/Volume_Feature의 학습 시각 파일:
calix 20261112_165625, tooth 20261112_190000, engine 20261113_132906, bonsai 20261113_144646.
"""

DATASET_PRESETS = {
    "calix": {
        "label": "CALIX",
        "volume": "resources/Volume_Data/CALIX.nii",
        "inr": "resources/Volume_Feature/CALIX_8d_neuvolex_gradfix.nii",
        "inr_old": "resources/Volume_Feature/CALIX_8d_neuvolex.nii",
        "tf": "resources/TFs/calix2.json",
        "camera": "resources/Camera/calix_camera.json",
        "points": "resources/Points/points_calix.json",
        "rois": {
            "calix": {"points": "resources/Points/points_calix.json",
                      "another": "resources/Camera/calix_camera_another.json"},
            "left kidney": {"points": "resources/Points/points_calix_left kidney.json",
                            "another": "resources/Camera/calix_camera_another2.json"},
            "spine": {"points": "resources/Points/points_calix_spine.json",
                      "another": "resources/Camera/calix_camera_another3.json"},
        },
    },
    "tooth": {
        "label": "Tooth",
        "volume": "resources/Volume_Data/tooth.nii",
        "inr": "resources/Volume_Feature/tooth_8d_neuvolex_gradfix.nii",
        "inr_old": "resources/Volume_Feature/tooth_8d_neuvolex.nii",
        "tf": "resources/TFs/Tooth.json",
        "camera": "resources/Camera/Tooth_camera.json",
        "points": "resources/Points/points_tooth.json",
        "k": 4,   # 실험 k (나머지 데이터셋은 6) — 기록용, 코드에서 읽지 않음
        "rois": {
            "tooth": {"points": "resources/Points/points_tooth.json",
                      "another": "resources/Camera/Tooth_camera_another.json"},
            "crown": {"points": "resources/Points/points_tooth_crown.json",
                      "another": "resources/Camera/Tooth_camera_another2.json"},
            "dentin": {"points": "resources/Points/points_tooth_dentin.json",
                       "another": "resources/Camera/Tooth_camera_another3.json"},
        },
    },
    "nucleon": {
        "label": "Nucleon",
        "volume": "resources/Volume_Data/nucleon.nii",
        "inr": "resources/Volume_Feature/nucleon_8d_neuvolex.nii",
        "tf": "resources/TFs/nucleon.json",
        "camera": "resources/Camera/nucleon_camera.json",
        "points": "resources/Points/points_nucleon.json",
    },
    "engine": {
        "label": "Engine",
        "volume": "resources/Volume_Data/engine.nii",
        "inr": "resources/Volume_Feature/engine_8d_neuvolex_gradfix.nii",
        "inr_old": "resources/Volume_Feature/engine_8d_neuvolex.nii",
        "tf": "resources/TFs/Engine.json",
        "camera": "resources/Camera/Engine_camera.json",
        "points": "resources/Points/points_engine.json",
        # another 카메라가 하나뿐이라 모든 ROI가 같이 쓴다
        "rois": {
            "engine": {"points": "resources/Points/points_engine.json",
                       "another": "resources/Camera/Engine_camera_another.json"},
            "engine2": {"points": "resources/Points/points_engine2.json",
                        "another": "resources/Camera/Engine_camera_another.json"},
            "nose": {"points": "resources/Points/points_engine_nose.json",
                     "another": "resources/Camera/Engine_camera_another.json"},
            "plate": {"points": "resources/Points/points_engine_plate.json",
                      "another": "resources/Camera/Engine_camera_another.json"},
        },
    },
    "3Dircad2.1": {
        "label": "3Dircadb2.1",
        "volume": "resources/Volume_Data/3Dircadb2.1.nii",
        "inr": "resources/Volume_Feature/3Dircadb2.1_8d_neuvolex.nii",
        "tf": "resources/TFs/3Dircad2.1.json",
        "camera": "resources/Camera/3Dircad2.1_camera.json",
        "points": "resources/Points/points_3Dircad2.1.json",
    },
    "bonsai": {
        "label": "Bonsai",
        "volume": "resources/Volume_Data/bonsai_256x256x256_uint8.raw",
        "raw": {"shape": (256, 256, 256), "dtype_str": "uint8", "endian": "little",
                "voxel_spacing": (1.0, 1.0, 1.0)},
        "inr": "resources/Volume_Feature/Bonsai_8d_neuvolex_gradfix.nii",
        "inr_old": "resources/Volume_Feature/Bonsai_8d_neuvolex.nii",
        "tf": "resources/TFs/Bonsai.json",
        "camera": "resources/Camera/Bonsai_camera.json",
        "points": "resources/Points/points_bonsai.json",
        "rois": {   # another 카메라가 하나뿐이라 두 ROI가 같이 쓴다
            "bonsai": {"points": "resources/Points/points_bonsai.json",
                       "another": "resources/Camera/Bonsai_camera_another.json"},
            "pot": {"points": "resources/Points/points_bonsai_pot.json",
                    "another": "resources/Camera/Bonsai_camera_another.json"},
        },
    },
    "carp": {
        "label": "Carp",
        # Carp DICOM 변환, voxel 간격은 Carp.pvm 헤더 값(0.78125, 0.390625, 1.0)
        "volume": "resources/Volume_Data/Carp_256x256x512_uint16.nii",
        "inr": "resources/Volume_Feature/Carp_8d_neuvolex_gradfix.nii",
        # Voreen carp3.tfi(domain 0~2871)에서 변환한 TF
        "tf": "resources/TFs/Carp.json",
        "camera": "resources/Camera/Carp_camera.json",
        "points": "resources/Points/points_carp_backfin.json",
        "rois": {"backfin": {"points": "resources/Points/points_carp_backfin.json",
                             "another": None}},   # another 카메라는 아직 없음
    },
    "vix": {
        "label": "VIX",
        # VIX Pied-cheville1.0mmstd5_12bits.mhd(양발 발목 CT) 변환, 간격은 헤더 값(0.40234375, 0.40234375, 1.0)
        "volume": "resources/Volume_Data/VIX_512x512x250_uint16.nii",
        "inr": "resources/Volume_Feature/VIX_8d_neuvolex_gradfix.nii",   # INR-DVR 20261116_172948
        # Voreen vix.tfi(domain 0~2717 = 볼륨 min-max)에서 변환한 TF
        "tf": "resources/TFs/VIX.json",
        "camera": "resources/Camera/Vix_camera.json",
        "clipping": {"z": [0.30, 1.0]},               # 발목 위(z 아래쪽 30%)를 잘라낸다, 0~1 정규화
        "points": None,
    },
    "tornado": {
        "label": "Tornado",
        # tornado_0_12bits.raw(big-endian) 변환
        "volume": "resources/Volume_Data/tornado_256x256x256_uint16.nii",
        "inr": "resources/Volume_Feature/Tornado_8d_neuvolex_gradfix.nii",  # INR-DVR 20261116_171843
        "tf": "resources/TFs/Tornado.json",
        "camera": "resources/Camera/Tornado_camera.json",
        "points": None,
    },
    "lobster": {
        "label": "Lobster",
        "volume": "resources/Volume_Data/Lobster_301x324x56_uint8.nii",
        "inr": "resources/Volume_Feature/Lobster_8d_neuvolex_gradfix.nii",  # INR-DVR 20261116_192236
        # Voreen lobster.tfi에서 변환한 TF
        "tf": "resources/TFs/Lobster.json",
        "camera": None,                               # 카메라 아직 없음
        "points": None,
    },
    "manix": {
        "label": "MANIX",
        "volume": "resources/Volume_Data/MANIX_256x256x230_uint16.nii",
        # gradient head 수정 전에 학습한 feature(gradfix 없음)
        "inr": "resources/Volume_Feature/Manix_8d_neuvolex.nii",
        "tf": "resources/TFs/Manix_wysiwyg.json",
        "camera": None,                               # 카메라 아직 없음
        "points": None,
    },
    "cthead": {
        "label": "CThead",
        "volume": "resources/Volume_Data/CThead_16bit_256x256x113.raw",
        "raw": {"shape": (256, 256, 113), "dtype_str": "uint16", "endian": "little",
                "voxel_spacing": (1.0, 1.0, 2.0)},
        "inr": "resources/Volume_Feature/CThead_8d_neuvolex.nii",
        "tf": "resources/TFs/CThead.json",
        "camera": "resources/Camera/CTHead_camera.json",
        "points": "resources/Points/points_cthead.json",
    },
    "brainix": {
        "label": "BRAINIX",
        # FLAIR(sT2WFLAIR401_enlarged)와 같은 파일. TF는 Voreen .vws에서 변환(x = key×4095/1012)
        "volume": "resources/Volume_Data/brainix.nii",
        "inr": "resources/Volume_Feature/brainix_8d_neuvolex.nii",
        "tf": "resources/TFs/BRAINIX/brainix_initial_black_cuboid.json",
        "camera": "resources/Camera/brainix_camera.json",
        "points": "resources/Points/points_brainix.json",
    },
}

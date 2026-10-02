"""
PatchCore 기반 이상 탐지 실습 코드
인공지능 융합 프로젝트 / Vision AI 실습 - Anomaly Detection

개요
    정상 이미지만으로 메모리 뱅크를 구축하고, 테스트 이미지의 패치와
    메모리 뱅크 사이의 최근접 이웃 거리를 이상 점수로 사용한다.
    역전파 학습이 없으므로 1회 순회로 학습이 완료된다.

    일반적인 분류 모델과 달리 결함 이미지를 학습에 쓰지 않는다.
    실제 공장에서는 불량 샘플이 드물고 불량의 종류를 미리 알 수 없기 때문에,
    "정상이 아닌 것"을 찾는 방식이 더 현실적이다.

동작 원리
    1) 특징 추출
        ImageNet 으로 학습된 CNN 에 정상 이미지를 통과시켜 중간 특징을 얻는다.
        이미지 한 장이 격자 형태의 패치 특징 여러 개로 바뀐다.
    2) 메모리 뱅크 구축
        모든 정상 이미지의 패치 특징을 한곳에 모은다.
        그대로 두면 양이 많으므로 coreset 기법으로 대표 패치만 남긴다.
        서로 멀리 떨어진 것을 우선 고르므로 정상 분포의 다양성이 유지된다.
    3) 이상 점수 계산
        테스트 이미지의 각 패치에 대해 메모리 뱅크에서 가장 가까운 이웃을 찾는다.
        그 거리가 곧 해당 위치의 이상 정도가 된다.
        정상이라면 비슷한 패치가 뱅크에 있어 거리가 짧고,
        결함이라면 닮은 것이 없어 거리가 길어진다.
    4) 판정과 시각화
        패치별 거리를 이미지 크기로 확대하면 이상 맵(히트맵)이 되고,
        그중 최대값이 이미지 한 장의 이상 점수가 된다.

평가 지표
    image_AUROC   정상과 결함 이미지를 얼마나 잘 가르는지 (1.0 이 최고)
    pixel_AUROC   결함 위치를 화소 단위로 얼마나 정확히 짚는지
    image_F1Score 임계값을 적용했을 때의 정밀도와 재현율의 조화평균

사전 준비
    pip install "anomalib[cpu]"        # CPU 환경
    pip install "anomalib[cu126]"      # CUDA 환경 (버전은 설치 환경에 맞춘다)
    pip install matplotlib

데이터셋 위치
    ../00_Dataset 아래에 MVTec AD 구조로 둔다. 기본 카테고리는 bottle 이다.
        00_Dataset/bottle/train/good/        정상 이미지 (학습용)
        00_Dataset/bottle/test/good/         정상 이미지 (평가용)
        00_Dataset/bottle/test/<결함이름>/   결함 이미지 (평가용)
        00_Dataset/bottle/ground_truth/      결함 마스크

실행 예시
    python patchcore_anomaly_detection.py                        # 기본값으로 학습 + 평가
    python patchcore_anomaly_detection.py train   --category bottle
    python patchcore_anomaly_detection.py custom  --root ./dataset/my_parts
    python patchcore_anomaly_detection.py predict --ckpt results/.../model.ckpt --images ./test_images
    python patchcore_anomaly_detection.py export  --ckpt results/.../model.ckpt
"""

import argparse
import os
from pathlib import Path

# ----------------------------------------------------------------------
# 경로 설정
# ----------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
Dataset_dir = os.path.abspath(os.path.join(script_dir, "../00_Dataset"))
Videos_dir = os.path.abspath(os.path.join(script_dir, "../00_Sample_Video"))

# 사전학습 백본(wide_resnet50_2)을 00_Models 아래에 받아 두고 다음 실행부터 재사용한다.
# timm 은 huggingface_hub 를 통해 가중치를 내려받고, 저장 위치를 아래 환경 변수로 정한다.
# huggingface_hub 와 torch 는 import 시점에 이 값을 읽으므로,
# 반드시 anomalib 을 import 하기 전에 지정해야 한다. (그래서 import 문 사이에 위치한다)
os.makedirs(Models_dir, exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", Models_dir)   # timm / huggingface_hub 경로
os.environ.setdefault("TORCH_HOME", Models_dir)     # torch.hub 로 받는 경우의 경로
# Windows 에서는 캐시가 심볼릭 링크 대신 파일 복사로 동작한다. 정상이므로 경고만 끈다.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

# anomalib : 이상 탐지 모델과 학습 절차를 묶어 제공하는 라이브러리
#   MVTecAD       - MVTec AD 규격 폴더를 읽어주는 데이터 모듈
#   Folder        - 직접 찍은 이미지를 정상/결함 폴더 단위로 읽어주는 데이터 모듈
#   PredictDataset- 정답 없이 추론만 할 이미지 폴더를 읽어주는 데이터셋
#   Engine        - 학습, 평가, 추론, 내보내기를 실행하는 실행기
#   Patchcore     - 이 실습에서 사용할 모델
from anomalib.data import Folder, MVTecAD, PredictDataset
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ----------------------------------------------------------------------
# 공통 : 모델 생성
# ----------------------------------------------------------------------
def build_model(coreset_ratio: float = 0.1, num_neighbors: int = 9) -> Patchcore:
    """PatchCore 모델을 생성한다.

    backbone
        ImageNet 사전학습 WideResNet-50 을 사용한다.
    layers
        layer2 와 layer3 의 중간 특징을 사용한다. 최종층은 ImageNet 분류에
        치우쳐 있어 제외한다.
    coreset_sampling_ratio
        메모리 뱅크에 남길 패치의 비율이다. 값을 낮추면 추론이 빨라지고
        메모리 사용량이 줄어드는 대신 정확도가 하락할 수 있다.
    num_neighbors
        이상 점수를 산정할 때 참조하는 최근접 이웃의 개수이다.
    """
    # 여기서는 설정만 담은 객체를 만든다.
    # 백본 가중치는 실제로 학습/추론이 시작될 때 내려받아 올라간다.
    return Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        # layer1 은 너무 저수준(모서리, 색)이라 결함 구분력이 약하고,
        # layer4 는 "이것은 병이다" 같은 분류 정보에 치우쳐 위치 정보가 흐려진다.
        # 중간층이 질감과 형태를 함께 담고 있어 결함 검출에 적합하다.
        coreset_sampling_ratio=coreset_ratio,
        num_neighbors=num_neighbors,
    )


# ----------------------------------------------------------------------
# 1. MVTec AD 학습 및 평가
# ----------------------------------------------------------------------
def run_train(args) -> None:
    """MVTec AD 의 한 카테고리로 학습하고 평가 지표를 출력한다.

    데이터가 없으면 anomalib 이 자동으로 내려받는다.
    학습 세트에는 정상 이미지만 포함되어 있어야 한다.
    """
    # 데이터 모듈은 폴더 구조를 읽어 학습/평가용 데이터로더를 만들어 준다.
    # 학습 데이터로더에는 train/good 만, 평가 데이터로더에는 test 폴더 전체가 들어간다.
    datamodule = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.coreset_ratio, args.num_neighbors)

    # PatchCore 는 역전파 학습이 없으므로 max_epochs 는 1 로 고정한다.
    engine = Engine(max_epochs=1)

    # fit 이라는 이름이지만 가중치를 갱신하지 않는다.
    # 정상 이미지를 한 번 훑어 패치 특징을 모으고, coreset 으로 추려 메모리 뱅크에 저장한다.
    # 그래서 에폭을 늘려도 결과가 달라지지 않는다.
    engine.fit(model=model, datamodule=datamodule)

    # test 는 평가용 정상/결함 이미지를 모두 통과시켜 점수를 매기고 지표를 계산한다.
    # 결함 마스크(ground_truth)가 있으면 화소 단위 지표까지 나온다.
    results = engine.test(model=model, datamodule=datamodule)

    print("\n[평가 결과]")
    for row in results:
        for key, value in row.items():
            print(f"  {key:<28} {value:.4f}" if isinstance(value, float) else f"  {key:<28} {value}")
    print("\n체크포인트와 결과 이미지는 results/ 폴더 아래에 저장된다.")


# ----------------------------------------------------------------------
# 2. 자체 촬영 데이터 학습
# ----------------------------------------------------------------------
def run_custom(args) -> None:
    """직접 수집한 이미지로 학습한다.

    요구되는 폴더 구조
        <root>/
        ├─ good/       정상 이미지 (학습에 사용)
        ├─ defect/     결함 이미지 (평가에만 사용, 선택)
        └─ mask/       결함 마스크 (선택, 화소 단위 평가 시 필요)

    good 폴더에 결함 이미지가 섞이면 정상 분포가 오염되어 검출률이 떨어진다.
    촬영 시 조명과 카메라 거리를 고정해야 성능이 안정된다.
    """
    # 정상 이미지가 없으면 메모리 뱅크를 만들 수 없으므로 먼저 확인한다.
    root = Path(args.root)
    if not (root / args.normal_dir).exists():
        raise FileNotFoundError(f"정상 이미지 폴더가 없다: {root / args.normal_dir}")

    # MVTecAD 와 달리 폴더 이름을 직접 지정한다. 구조만 맞으면 어떤 이름이든 된다.
    datamodule = Folder(
        name=root.name,
        root=root,
        normal_dir=args.normal_dir,
        abnormal_dir=args.abnormal_dir,
        mask_dir=args.mask_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.coreset_ratio, args.num_neighbors)
    engine = Engine(max_epochs=1)

    engine.fit(model=model, datamodule=datamodule)

    # 결함 이미지가 준비된 경우에만 평가가 의미를 가진다.
    if args.abnormal_dir and (root / args.abnormal_dir).exists():
        engine.test(model=model, datamodule=datamodule)
    else:
        print("결함 이미지 폴더가 없어 평가 단계를 건너뛴다.")


# ----------------------------------------------------------------------
# 3. 추론 및 히트맵 저장
# ----------------------------------------------------------------------
def run_predict(args) -> None:
    """학습된 체크포인트로 폴더 안의 이미지를 판정한다.

    출력
        이미지별 이상 점수와 판정 결과를 표준 출력에 표시하고,
        이상 맵을 원본 위에 겹친 이미지를 out 폴더에 저장한다.
    """
    import matplotlib

    # Agg 는 화면 없이 파일로만 그리는 백엔드다.
    # 창을 띄우지 않으므로 원격 접속이나 서버 환경에서도 동작한다.

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = PredictDataset(path=args.images, image_size=(256, 256))
    # 추론 전용 데이터셋이라 정답 라벨이 필요 없다.
    model = build_model(args.coreset_ratio, args.num_neighbors)
    engine = Engine()

    # 체크포인트에는 학습 때 만든 메모리 뱅크가 들어 있다.
    # 이것을 불러와야 비교 대상이 생기므로 ckpt_path 는 필수다.
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=args.ckpt)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'파일명':<32}{'이상 점수':>12}  판정")
    print("-" * 58)

    for batch in predictions:
        # 결과는 배치 단위로 반환되므로 요소 단위로 풀어서 처리한다.
        paths = _as_list(batch.image_path)
        scores = _as_list(batch.pred_score)
        labels = _as_list(batch.pred_label)
        maps = batch.anomaly_map

        for i, path in enumerate(paths):
            score = float(np.asarray(scores[i]).item())
            label = int(np.asarray(labels[i]).item())
            verdict = "이상" if label == 1 else "정상"
            print(f"{Path(path).name:<32}{score:>12.4f}  {verdict}")

            if maps is None:
                continue

            # 이상 맵은 특징 격자 크기라 원본보다 작다. 겹쳐 그리려면 확대가 필요하다.
            anomaly_map = np.asarray(maps[i]).squeeze()
            image = plt.imread(path)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image)
            axes[0].set_title("input")
            axes[1].imshow(anomaly_map, cmap="jet")
            axes[1].set_title("anomaly map")
            axes[2].imshow(image)
            axes[2].imshow(
                _resize_map(anomaly_map, image.shape[:2]), cmap="jet", alpha=0.45
            )
            axes[2].set_title(f"{verdict}  score={score:.3f}")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{Path(path).stem}_result.png", dpi=120)
            plt.close(fig)

    print(f"\n결과 이미지는 {out_dir} 폴더에 저장되었다.")


# ----------------------------------------------------------------------
# 4. OpenVINO 내보내기
# ----------------------------------------------------------------------
def run_export(args) -> None:
    """엣지 배포용으로 모델을 내보낸다.

    OpenVINO 형식으로 변환하면 CPU 환경에서의 추론 속도가 개선된다.
    변환 전후의 FPS 를 비교해 보는 것을 권장한다.
    """
    model = build_model(args.coreset_ratio, args.num_neighbors)
    engine = Engine()
    # OpenVINO 는 인텔 CPU 에 최적화된 추론 형식이다.
    # 학습용 프레임워크 없이도 돌아가므로 현장 장비에 올리기 쉽다.
    engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        ckpt_path=args.ckpt,
    )
    print("내보내기가 완료되었다. results/ 폴더 아래의 weights 경로를 확인한다.")


# ----------------------------------------------------------------------
# 보조 함수
# ----------------------------------------------------------------------
def _as_list(value):
    """스칼라와 배치 출력을 동일한 형태로 취급한다."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    arr = np.asarray(value)
    return arr.tolist() if arr.ndim > 0 else [arr]


def _resize_map(anomaly_map: np.ndarray, shape) -> np.ndarray:
    """이상 맵을 원본 이미지 크기로 확대한다."""
    try:
        import cv2

        return cv2.resize(anomaly_map, (shape[1], shape[0]))
    except ImportError:
        ys = (np.linspace(0, anomaly_map.shape[0] - 1, shape[0])).astype(int)
        xs = (np.linspace(0, anomaly_map.shape[1] - 1, shape[1])).astype(int)
        return anomaly_map[np.ix_(ys, xs)]


# ----------------------------------------------------------------------
# 진입점
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    # 서브커맨드(train/custom/predict/export)로 실행 모드를 나눈다.
    # 각 모드는 set_defaults(func=...) 로 자신이 호출할 함수를 지정해 두고,
    # 맨 아래 진입점에서 args.func(args) 한 줄로 실행된다.
    # 공통 인자 (모든 모드에서 같이 쓰인다)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--coreset-ratio", type=float, default=0.1, help="메모리 뱅크 축소 비율")
    common.add_argument("--num-neighbors", type=int, default=9, help="점수 산정에 사용할 이웃 수")
    common.add_argument("--batch-size", type=int, default=8, help="배치 크기")
    common.add_argument("--num-workers", type=int, default=4, help="데이터 로더 워커 수")

    # train 모드 인자
    # 서브커맨드를 생략했을 때도 그대로 쓰이므로 따로 묶어서 양쪽에 붙인다.
    # Dataset_dir 아래의 <category> 폴더를 찾는다. 즉 00_Dataset/bottle 을 사용하게 된다.
    train_args = argparse.ArgumentParser(add_help=False)
    train_args.add_argument("--category", default="bottle", help="MVTec AD 카테고리 이름")
    train_args.add_argument("--data-root", default=Dataset_dir,
                            help="카테고리 폴더가 들어있는 상위 폴더")

    parser = argparse.ArgumentParser(
        description="PatchCore 기반 이상 탐지 실습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common, train_args],
    )

    # 서브커맨드를 생략하고 그냥 실행하면 train 모드(학습 + 평가)로 동작한다.
    #   python patchcore_anomaly_detection.py
    parser.set_defaults(func=run_train)

    sub = parser.add_subparsers(dest="mode")

    p_train = sub.add_parser("train", parents=[common, train_args], help="MVTec AD 학습 및 평가")
    p_train.set_defaults(func=run_train)

    p_custom = sub.add_parser("custom", parents=[common], help="자체 촬영 데이터 학습")
    p_custom.add_argument("--root", required=True, help="데이터 최상위 폴더")
    p_custom.add_argument("--normal-dir", default="good", help="정상 이미지 폴더 이름")
    p_custom.add_argument("--abnormal-dir", default="defect", help="결함 이미지 폴더 이름")
    p_custom.add_argument("--mask-dir", default=None, help="결함 마스크 폴더 이름")
    p_custom.set_defaults(func=run_custom)

    p_pred = sub.add_parser("predict", parents=[common], help="이미지 폴더 추론")
    p_pred.add_argument("--ckpt", required=True, help="학습된 체크포인트 경로")
    p_pred.add_argument("--images", required=True, help="추론할 이미지 폴더")
    p_pred.add_argument("--out", default="./predict_out", help="결과 이미지 저장 폴더")
    p_pred.set_defaults(func=run_predict)

    p_exp = sub.add_parser("export", parents=[common], help="OpenVINO 내보내기")
    p_exp.add_argument("--ckpt", required=True, help="학습된 체크포인트 경로")
    p_exp.set_defaults(func=run_export)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)

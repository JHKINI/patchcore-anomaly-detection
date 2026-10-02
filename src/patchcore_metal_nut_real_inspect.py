# 학습된 PatchCore 체크포인트로 같은 폴더의 metal_nut_real 이미지의 정상/불량을 판별하는 스크립트
"""
patchcore_anomaly_detection.py 로 학습한 뒤 생성된 metal_nut_model.ckpt 를 불러와
실사 metal_nut 이미지 폴더를 판정한다. 재학습은 하지 않는다.

동작 순서
    1) 같은 폴더의 metal_nut_model.ckpt 를 불러온다.
       체크포인트에는 학습 때 만든 메모리 뱅크(정상 패치 모음)와
       판정에 필요한 모델 정보가 들어 있다.

    2) metal_nut_real 폴더의 이미지를 한 장씩 모델에 통과시킨다.
       각 패치를 메모리 뱅크와 비교해 이상 맵과 이상 점수를 얻는다.

    3) 점수가 임계값을 넘으면 불량, 넘지 않으면 정상으로 판정한다.

    4) 판정표를 터미널에 출력하고,
       이미지별 결과 그림을 metal_nut_real_predict_out 에 저장한다.

    5) 저장한 결과 그림을 창에 한 장씩 띄운다.
       (아무 키 = 다음, q/ESC = 종료)

준비할 파일
    17_Anomoly_Detection/metal_nut_model.ckpt
        학습 결과
        results/.../weights/lightning/model.ckpt
        를 복사해서 이름을 변경해 둔다.

    17_Anomoly_Detection/metal_nut_real/
        판별할 실제 metal_nut 이미지

실행 예시
    python patchcore_metal_nut_real_inspect.py

    python patchcore_metal_nut_real_inspect.py ^
        --ckpt results/Patchcore/MVTecAD/metal_nut/v1/weights/lightning/model.ckpt

    python patchcore_metal_nut_real_inspect.py ^
        --images D:/my_images ^
        --out ./my_out
"""

import argparse
from pathlib import Path


# ----------------------------------------------------------------------
# 학습 스크립트에서 공통 기능 가져오기
# ----------------------------------------------------------------------
# 학습 때와 같은 설정으로 모델을 만들어야 체크포인트가 정상적으로 올라간다.
#
# Engine         - 추론을 실행하는 anomalib 실행기
# PredictDataset - 정답 라벨 없이 이미지 폴더만 읽는 데이터셋
# _as_list       - 배치 출력을 리스트로 풀어 주는 보조 함수
# _resize_map    - 이상 맵을 원본 이미지 크기로 늘리는 보조 함수
# build_model    - 학습 때와 같은 설정의 PatchCore 모델 생성 함수
# np             - numpy
# script_dir     - 학습 스크립트가 있는 폴더
#
from patchcore_anomaly_detection import (
    Engine,
    PredictDataset,
    _as_list,
    _resize_map,
    build_model,
    np,
    script_dir,
)


# ----------------------------------------------------------------------
# 기본 경로
# ----------------------------------------------------------------------

# metal_nut 학습 결과 체크포인트
Model_path = Path(script_dir) / "metal_nut_model.ckpt"

# 실제 촬영한 metal_nut 이미지 폴더
Real_dir = Path(script_dir) / "metal_nut_real"


# ----------------------------------------------------------------------
# 판별
# ----------------------------------------------------------------------
def inspect(args) -> None:
    """
    학습된 metal_nut PatchCore 체크포인트로
    실제 이미지를 판정하고 결과를 출력·저장한다.

    출력
        터미널
            이미지별 이상 점수와 판정
            전체 정상/불량 개수

        out 폴더
            <파일명>_result.png

            [원본]
            [이상 맵]
            [원본 + 이상 맵 Overlay]
    """

    import matplotlib

    # 화면에 직접 띄우지 않고 파일 저장용으로 사용
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # 입력 확인
    # ------------------------------------------------------------------

    ckpt = Path(args.ckpt)

    if not ckpt.exists():
        raise FileNotFoundError(
            f"체크포인트가 없다: {ckpt}\n"
            "metal_nut 학습 결과의 model.ckpt를 "
            "이 폴더에 metal_nut_model.ckpt로 복사한다."
        )

    images = Path(args.images)

    if not images.exists():
        raise FileNotFoundError(
            f"이미지 폴더가 없다: {images}"
        )

    print(f"체크포인트 : {ckpt}")
    print(f"이미지 폴더: {images}")

    # ------------------------------------------------------------------
    # 이미지 데이터셋 생성
    # ------------------------------------------------------------------

    # 학습 때와 같은 256 x 256 크기로 맞춘다.
    dataset = PredictDataset(
        path=images,
        image_size=(256, 256)
    )

    # 학습 때와 동일한 설정의 PatchCore 모델 생성
    model = build_model(
        args.coreset_ratio,
        args.num_neighbors
    )

    # anomalib 추론 실행기
    engine = Engine()

    # --------------------------------------------------------------
    # 체크포인트를 불러와 추론
    # --------------------------------------------------------------
    # 여기서 재학습하지 않는다.
    #
    # metal_nut 정상 데이터로 구축한
    # PatchCore memory bank를 체크포인트에서 불러온다.
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt)
    )

    # 결과 저장 폴더
    out_dir = Path(args.out)
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    # ------------------------------------------------------------------
    # 이미지별 결과 처리
    # ------------------------------------------------------------------

    for batch in predictions:

        # 배치 결과를 이미지별 리스트로 변환
        paths = _as_list(batch.image_path)
        scores = _as_list(batch.pred_score)
        labels = _as_list(batch.pred_label)
        maps = batch.anomaly_map

        for i, path in enumerate(paths):

            # ----------------------------------------------------------
            # 이상 점수 / 판정
            # ----------------------------------------------------------

            score = float(
                np.asarray(scores[i]).item()
            )

            label = int(
                np.asarray(labels[i]).item()
            )

            verdict = (
                "불량"
                if label == 1
                else "정상"
            )

            rows.append(
                (
                    Path(path).name,
                    score,
                    verdict
                )
            )

            # 이상 맵이 없으면 결과 그림 생성 생략
            if maps is None:
                continue

            # ----------------------------------------------------------
            # 이상 맵
            # ----------------------------------------------------------

            anomaly_map = np.asarray(
                maps[i]
            ).squeeze()

            # 원본 이미지 읽기
            image = plt.imread(path)

            # ----------------------------------------------------------
            # 결과 시각화
            # ----------------------------------------------------------

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(12, 4)
            )

            # ① 원본
            axes[0].imshow(image)
            axes[0].set_title("INPUT")

            # ② 이상 맵
            axes[1].imshow(
                anomaly_map,
                cmap="jet"
            )
            axes[1].set_title("ANOMALY MAP")

            # ③ Overlay
            axes[2].imshow(image)

            # 이상 맵을 원본 크기로 확대
            resized_map = _resize_map(
                anomaly_map,
                image.shape[:2]
            )

            axes[2].imshow(
                resized_map,
                cmap="jet",
                alpha=0.45
            )

            axes[2].set_title(
                f"{'DEFECT' if verdict == '불량' else 'NORMAL'} "
                f"score={score:.3f}"
            )

            # 축 제거
            for ax in axes:
                ax.axis("off")

            fig.tight_layout()

            # 결과 저장
            result_path = (
                out_dir
                / f"{Path(path).stem}_result.png"
            )

            fig.savefig(
                result_path,
                dpi=120
            )

            plt.close(fig)

    # ------------------------------------------------------------------
    # 결과 정렬
    # ------------------------------------------------------------------

    # 1, 2, 3 ... 10 순서로 정렬
    rows.sort(
        key=lambda r: (
            len(Path(r[0]).stem),
            r[0]
        )
    )

    # ------------------------------------------------------------------
    # 판정 결과 출력
    # ------------------------------------------------------------------

    print(
        f"\n{'파일명':<24}"
        f"{'이상 점수':>12}  판정"
    )

    print("-" * 48)

    for name, score, verdict in rows:

        print(
            f"{name:<24}"
            f"{score:>12.4f}  "
            f"{verdict}"
        )

    # ------------------------------------------------------------------
    # 요약
    # ------------------------------------------------------------------

    n_bad = sum(
        1
        for r in rows
        if r[2] == "불량"
    )

    n_total = len(rows)
    n_normal = n_total - n_bad

    print("-" * 48)

    print(
        f"전체 {n_total}장 / "
        f"정상 {n_normal}장 / "
        f"불량 {n_bad}장"
    )

    print(
        f"결과 이미지 저장 위치: "
        f"{out_dir.resolve()}"
    )


# ----------------------------------------------------------------------
# 결과 보기
# ----------------------------------------------------------------------
def show_results(
    out_dir: Path,
    rows
) -> None:
    """
    저장된 결과 이미지를 판정표 순서대로 한 장씩 띄운다.

    조작
        아무 키      다음 이미지
        q / ESC      보기 종료
        창 닫기(X)   보기 종료
    """

    import cv2

    win = "PatchCore metal_nut result"

    print(
        "\n결과 이미지를 한 장씩 표시한다."
        " 아무 키 = 다음, q/ESC = 종료"
    )

    for i, (
        name,
        score,
        verdict
    ) in enumerate(
        rows,
        start=1
    ):

        path = (
            out_dir
            / f"{Path(name).stem}_result.png"
        )

        if not path.exists():
            continue

        # --------------------------------------------------------------
        # 이미지 읽기
        # --------------------------------------------------------------
        # Windows 한글 경로 문제를 피하기 위해
        # np.fromfile + cv2.imdecode 사용
        image = cv2.imdecode(
            np.fromfile(
                str(path),
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        cv2.imshow(
            win,
            image
        )

        # 창 제목
        cv2.setWindowTitle(
            win,
            f"[{i}/{len(rows)}] "
            f"{name}  "
            f"{'DEFECT' if verdict == '불량' else 'NORMAL'}  "
            f"score={score:.3f}"
        )

        # --------------------------------------------------------------
        # 키 입력 / 창 닫기 확인
        # --------------------------------------------------------------
        while True:

            key = cv2.waitKey(100)

            # 창 X 버튼으로 닫은 경우
            if cv2.getWindowProperty(
                win,
                cv2.WND_PROP_VISIBLE
            ) < 1:

                cv2.destroyAllWindows()
                return

            # 키 입력
            if key != -1:
                break

        # q 또는 ESC
        if key & 0xFF in (
            ord("q"),
            27
        ):
            break

    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# 명령행 인자
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "PatchCore 체크포인트로 "
            "실사 metal_nut 이미지 "
            "정상/불량 판별"
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # 체크포인트
    parser.add_argument(
        "--ckpt",
        default=str(Model_path),
        help="metal_nut 학습 체크포인트 경로"
    )

    # 실사 이미지 폴더
    parser.add_argument(
        "--images",
        default=str(Real_dir),
        help="판별할 metal_nut 이미지 폴더"
    )

    # 결과 저장 폴더
    parser.add_argument(
        "--out",
        default="./metal_nut_real_predict_out",
        help="결과 이미지 저장 폴더"
    )

    # --------------------------------------------------------------
    # 학습 때와 동일하게 유지해야 하는 값
    # --------------------------------------------------------------

    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.1,
        help="학습 때와 같은 coreset 비율"
    )

    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=9,
        help="학습 때와 같은 최근접 이웃 수"
    )

    return parser


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
if __name__ == "__main__":

    args = build_parser().parse_args()

    inspect(args)
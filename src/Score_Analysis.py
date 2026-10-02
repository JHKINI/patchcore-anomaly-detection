"""
06_Score_Analysis.py

MVTec AD metal_nut 테스트 데이터의 PatchCore 이상 점수 분석

목적
1. 정상 / 결함 이미지의 anomaly score 분포 확인
2. threshold 변화에 따른 FP / FN 분석
3. 정상과 결함의 score 분리 정도 확인

주의
- 기존 patchcore_anomaly_detection.py는 수정하지 않는다.
- 학습하지 않는다.
- 기존 metal_nut PatchCore 체크포인트만 사용한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from patchcore_anomaly_detection import (
    Engine,
    PredictDataset,
    _as_list,
    build_model,
)


# ============================================================
# 1. 경로 설정
# ============================================================

script_dir = Path(__file__).resolve().parent

# 기존에 학습 완료한 metal_nut 체크포인트
CKPT = (
    script_dir
    / "results"
    / "Patchcore"
    / "MVTecAD"
    / "metal_nut"
    / "v1"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

# MVTec AD metal_nut 테스트 폴더
TEST_DIR = (
    script_dir
    / ".."
    / "00_Dataset"
    / "metal_nut"
    / "test"
)

# 결과 저장
OUT_DIR = script_dir / "score_analysis_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 입력 확인
# ============================================================

if not CKPT.exists():
    raise FileNotFoundError(
        f"체크포인트를 찾을 수 없습니다.\n{CKPT}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"MVTec 테스트 폴더를 찾을 수 없습니다.\n{TEST_DIR}"
    )


print("=" * 70)
print("PatchCore Score Analysis")
print("=" * 70)
print(f"Checkpoint : {CKPT}")
print(f"Test data  : {TEST_DIR}")
print(f"Output     : {OUT_DIR}")
print()


# ============================================================
# 3. 테스트 이미지 목록 구성
# ============================================================

image_paths = []
true_labels = []

valid_extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}

for defect_dir in sorted(TEST_DIR.iterdir()):

    if not defect_dir.is_dir():
        continue

    # good = 정상
    if defect_dir.name == "good":
        label = 0

    # 나머지 = 결함
    else:
        label = 1

    for image_path in sorted(defect_dir.iterdir()):

        if image_path.suffix.lower() not in valid_extensions:
            continue

        image_paths.append(image_path)
        true_labels.append(label)


print(f"전체 테스트 이미지 : {len(image_paths)}")
print(f"정상 이미지         : {sum(x == 0 for x in true_labels)}")
print(f"결함 이미지         : {sum(x == 1 for x in true_labels)}")
print()


# ============================================================
# 4. PatchCore 추론
# ============================================================

dataset = PredictDataset(
    path=TEST_DIR,
    image_size=(256, 256),
)

# 학습 당시와 동일한 PatchCore 설정
model = build_model(
    coreset_ratio=0.1,
    num_neighbors=9,
)

engine = Engine()

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=str(CKPT),
)


# ============================================================
# 5. anomaly score 수집
# ============================================================

rows = []

for batch in predictions:

    paths = _as_list(batch.image_path)
    scores = _as_list(batch.pred_score)

    for i, path in enumerate(paths):

        score = float(
            np.asarray(scores[i]).item()
        )

        # 폴더 이름으로 실제 정답 결정
        path_obj = Path(path)

        if path_obj.parent.name == "good":
            true_label = 0
        else:
            true_label = 1

        rows.append(
            {
                "filename": path_obj.name,
                "category": path_obj.parent.name,
                "true_label": true_label,
                "true_class": (
                    "normal"
                    if true_label == 0
                    else "defect"
                ),
                "anomaly_score": score,
            }
        )

df = pd.DataFrame(rows)
# ============================================================
# 6. 결과 CSV 저장
# ============================================================

csv_path = OUT_DIR / "score_results.csv"

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"Score 결과 저장 : {csv_path}")
print()


# ============================================================
# 7. 정상 / 결함 Score 통계
# ============================================================

normal_scores = df.loc[
    df["true_label"] == 0,
    "anomaly_score",
].to_numpy()

defect_scores = df.loc[
    df["true_label"] == 1,
    "anomaly_score",
].to_numpy()


print("=" * 70)
print("Anomaly Score Statistics")
print("=" * 70)

print(
    f"정상 : "
    f"min={normal_scores.min():.4f}, "
    f"mean={normal_scores.mean():.4f}, "
    f"max={normal_scores.max():.4f}"
)

print(
    f"결함 : "
    f"min={defect_scores.min():.4f}, "
    f"mean={defect_scores.mean():.4f}, "
    f"max={defect_scores.max():.4f}"
)

print()


# ============================================================
# 8. Score Distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    normal_scores,
    bins=30,
    alpha=0.7,
    label="Normal",
)

plt.hist(
    defect_scores,
    bins=30,
    alpha=0.7,
    label="Defect",
)

plt.xlabel("Anomaly Score")
plt.ylabel("Number of Images")
plt.title("PatchCore Anomaly Score Distribution")
plt.legend()
plt.grid(alpha=0.3)

distribution_path = (
    OUT_DIR / "01_score_distribution.png"
)

plt.tight_layout()
plt.savefig(
    distribution_path,
    dpi=200,
)

plt.close()

print(
    f"Score 분포 그래프 저장 : "
    f"{distribution_path}"
)


# ============================================================
# 9. Threshold 분석
# ============================================================

def calculate_confusion(df, threshold):

    predicted = (
        df["anomaly_score"] >= threshold
    ).astype(int)

    true = df["true_label"]

    tp = int(((true == 1) & (predicted == 1)).sum())
    tn = int(((true == 0) & (predicted == 0)).sum())
    fp = int(((true == 0) & (predicted == 1)).sum())
    fn = int(((true == 1) & (predicted == 0)).sum())

    return tp, tn, fp, fn


thresholds = np.linspace(
    df["anomaly_score"].min(),
    df["anomaly_score"].max(),
    100,
)

threshold_rows = []

for threshold in thresholds:

    tp, tn, fp, fn = calculate_confusion(
        df,
        threshold,
    )

    threshold_rows.append(
        {
            "threshold": threshold,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }
    )


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# 10. Threshold별 FP / FN 그래프
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["FP"],
    label="False Positive",
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["FN"],
    label="False Negative",
)

plt.xlabel("Threshold")
plt.ylabel("Number of Images")
plt.title("False Positive / False Negative by Threshold")
plt.legend()
plt.grid(alpha=0.3)

threshold_path = (
    OUT_DIR / "02_threshold_fp_fn.png"
)

plt.tight_layout()
plt.savefig(
    threshold_path,
    dpi=200,
)

plt.close()

print(
    f"Threshold 분석 그래프 저장 : "
    f"{threshold_path}"
)


# ============================================================
# 11. FP + FN 최소 지점 확인
# ============================================================

threshold_df["FP_FN"] = (
    threshold_df["FP"]
    + threshold_df["FN"]
)

best_row = threshold_df.loc[
    threshold_df["FP_FN"].idxmin()
]

analysis_threshold = float(
    best_row["threshold"]
)

tp, tn, fp, fn = calculate_confusion(
    df,
    analysis_threshold,
)


print()
print("=" * 70)
print("Threshold Analysis Result")
print("=" * 70)

print(
    f"분석용 threshold : "
    f"{analysis_threshold:.4f}"
)

print(f"TP : {tp}")
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")

print()


# ============================================================
# 12. Threshold 결과 CSV
# ============================================================

threshold_csv = (
    OUT_DIR / "threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_csv,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"Threshold 결과 저장 : "
    f"{threshold_csv}"
)


# ============================================================
# 13. 최종 출력
# ============================================================

print()
print("=" * 70)
print("분석 완료")
print("=" * 70)

print(
    "생성 파일:"
)

print(
    "  01_score_distribution.png"
)

print(
    "  02_threshold_fp_fn.png"
)

print(
    "  score_results.csv"
)

print(
    "  threshold_analysis.csv"
)
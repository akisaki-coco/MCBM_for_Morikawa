import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# 例: Concept 0 の結果 (TP=21, FP=22, FN=104, TN=103)
# tp, fp, fn, tn, agree = 146, 144, 4, 6, 0.507
tp, fp, fn, tn, agree = 69, 65, 81, 85, 0.513

cm = np.array([
    [tn, fp],  # 正解が 0 の行: [TN, FP]
    [fn, tp],  # 正解が 1 の行: [FN, TP]
])

fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Absent (0)", "Present (1)"],
)
disp.plot(cmap="Blues", values_format="d", ax=ax)
ax.set_xlabel("LLM Prediction")
ax.set_ylabel("SAE Activation")
ax.set_title(f"Concept 2 (agreement = {agree:.3f})")

plt.tight_layout()
plt.savefig("confusion_matrix_concept2.png", dpi=150)
plt.close(fig)
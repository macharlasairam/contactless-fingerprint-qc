import glob
import cv2
import numpy as np

for path in sorted(glob.glob("test_dataset/glare/*.*")):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(
        f"{path:30s}  max={gray.max():3d}  "
        f"p99={np.percentile(gray, 99):6.1f}  "
        f"p95={np.percentile(gray, 95):6.1f}  "
        f"mean={gray.mean():6.1f}"
    )
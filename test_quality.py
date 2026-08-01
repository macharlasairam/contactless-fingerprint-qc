"""
test_quality.py

Runs quality_gate() over the 20-image test_dataset/ folder (5 each of
good/blurry/dark/glare) and prints + saves a summary table so you can
verify each defect type is actually flagged correctly.
"""

import glob
import os

import pandas as pd

from quality_assessment import quality_gate


def run_batch_tests(dataset_dir: str = "test_dataset") -> pd.DataFrame:
    records = []
    image_paths = sorted(glob.glob(f"{dataset_dir}/*/*.*"))

    if not image_paths:
        print(
            f"No images found under '{dataset_dir}/'. "
            f"Add your 20 test photos into good/, blurry/, dark/, glare/ subfolders first."
        )
        return pd.DataFrame()

    for path in image_paths:
        folder_category = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)

        res = quality_gate(path)

        records.append(
            {
                "File": filename,
                "Expected Category": folder_category,
                "Passed": res["passed"],
                "Composite Score": res["composite_score"],
                "Blur Score": res["blur"]["blur_score"],
                "Brightness": res["brightness"]["brightness"],
                "Glare Fraction": res["glare"]["glare_fraction"],
                "ROI Fraction": res["roi"]["roi_fraction"],
                "Ridge Score": res["ridge"]["ridge_score"],
                "Guidance": res["guidance"],
            }
        )

    df = pd.DataFrame(records)
    print("\n================ QUALITY CONTROL BATCH EVALUATION ================\n")
    print(df.to_string(index=False))
    df.to_csv("test_results.csv", index=False)
    print("\nSaved full results to test_results.csv")
    return df


if __name__ == "__main__":
    run_batch_tests()

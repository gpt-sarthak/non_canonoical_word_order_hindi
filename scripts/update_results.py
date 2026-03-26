"""
update_results.py

Step 6 of the pipeline — run after features.csv is rebuilt.
Updates all results, reports, and Word documents in one shot.

Usage:
    PYTHONPATH=. python scripts/update_results.py

Steps:
    1. Train ranking model     -> prints updated accuracy / coefficients
    2. Compare with paper      -> reports/paper_comparison.txt
    3. Generate HTML report    -> reports/pipeline_analysis.html
    4. Regenerate Word docs    -> reports/pipeline_explanation.docx
                                  reports/pipeline_detailed_explanation.docx
    5. Run all 4 tests         -> tests/output_test/
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def run(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)
    print(f"  [DONE] {label}")


if __name__ == "__main__":
    run("1/5  Train ranking model",
        [PYTHON, "scripts/train_ranking_model.py"])

    run("2/5  Compare with paper",
        [PYTHON, "scripts/compare_with_paper.py"])

    run("3/5  Generate HTML report",
        [PYTHON, "scripts/analyse_results.py"])

    run("4/5  Regenerate Word documents",
        [PYTHON, "scripts/generate_doc.py"])

    run("4/5  Regenerate detailed Word document",
        [PYTHON, "scripts/generate_detailed_doc.py"])

    run("5/5  Run tests",
        [PYTHON, "tests/test_dl.py"])
    run("5/5  Run tests",
        [PYTHON, "tests/test_trigram.py"])
    run("5/5  Run tests",
        [PYTHON, "tests/test_lstm.py"])
    run("5/5  Run tests",
        [PYTHON, "tests/test_ranking_model.py"])

    print(f"\n{'='*60}")
    print("  All results updated successfully.")
    print(f"  reports/paper_comparison.txt")
    print(f"  reports/pipeline_analysis.html")
    print(f"  reports/pipeline_explanation.docx")
    print(f"  reports/pipeline_detailed_explanation.docx")
    print(f"  tests/output_test/")
    print(f"{'='*60}\n")

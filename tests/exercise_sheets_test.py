import subprocess
import tempfile
import matplotlib

class Test_exercise_sheets:

    def test_exercise_sheet_1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise1",
                    "--output-dir",
                    temp_dir,
                    "Notebooks/exercise1.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                ["python3", f"{temp_dir}/exercise1.py"],
                check=True,
            )  # Check that exercise1.ipynb runs without errors.


    def test_exercise_solution_1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise1",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise1_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                ["python3", f"{temp_dir}/exercise1.py"],
                check=True,
            )  # Check that exercise1_sol.ipynb runs without errors.
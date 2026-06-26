import os
import sys
import subprocess
import tempfile

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EXERCISE_SHEET_TESTS") != "1",
    reason="exercise sheet notebook tests are disabled; set RUN_EXERCISE_SHEET_TESTS=1 to enable them",
)


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
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise1.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
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
                    "exercise1_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise1_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise1_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )  # Check that exercise1_sol.ipynb runs without errors.


    def test_exercise_solution_2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise2_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise2_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise2_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )  # Check that exercise2_sol.ipynb runs without errors.


    def test_exercise_solution_3(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise3_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise3_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise3_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )  # Check that exercise3_sol.ipynb runs without errors.

    def test_exercise_solution_4(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise4_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise4_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise4_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise4_sol.ipynb runs without errors.

    def test_exercise_solution_5(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise5_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise5_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise5_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise5_sol.ipynb runs without errors.


    def test_exercise_sheet_6(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise6",
                    "--output-dir",
                    temp_dir,
                    "Notebooks/exercise6.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise6.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise6.ipynb runs without errors.

    def test_exercise_solution_6(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise6_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise6_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise6_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise6_sol.ipynb runs without errors.

    def test_exercise_sheet_7(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise7",
                    "--output-dir",
                    temp_dir,
                    "Notebooks/exercise7.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise7.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise7.ipynb runs without errors.

    def test_exercise_solution_7(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise7_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise7_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise7_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise7_sol.ipynb runs without errors.

    def test_exercise_solution_8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise8_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise8_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise8_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise8_sol.ipynb runs without errors.

    def test_exercise_solution_9(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise9_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise9_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise9_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise9_sol.ipynb runs without errors.
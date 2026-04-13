import subprocess

class Test_exercise_sheets:

    def test_exercise_sheet_1(self):
        subprocess.run(["jupyter", "nbconvert", "--to", "script", "Notebooks/exercise1.ipynb"])
        run_sheet = subprocess.run(["python3", "Notebooks/exercise1.py"])
        try:
            assert run_sheet.returncode == 0 # Check that exercise1.ipynb runs without errors.
        finally:
            subprocess.run(["rm", "Notebooks/exercise1.py"])
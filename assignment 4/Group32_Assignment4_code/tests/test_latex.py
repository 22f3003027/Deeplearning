"""Check LaTeX escaping and self-contained project packaging."""
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from latex_report import tex,Report,project_zip


class LatexTests(unittest.TestCase):
    def test_special_characters_are_escaped(self):
        self.assertEqual(tex("20% & a_b"),r"20\% \& a\_b")
        self.assertEqual(tex("{x}#"),r"\{x\}\#")

    def test_portable_project_excludes_build_files(self):
        # Keep test scratch files inside the workspace for restricted environments.
        parent=Path(__file__).resolve().parents[1]/"tests"/"scratch"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            project=Path(temporary)/"report"
            report=Report(project,4,"Test")
            report.section("A & B",False)
            report.table(["Name","Value"],[["test",r"20\%"]],"Yr")
            source=report.finish()
            (project/"build").mkdir(); (project/"build"/"omit.aux").write_text("temporary")
            archive=Path(temporary)/"project.zip"
            project_zip(project,archive)
            with zipfile.ZipFile(archive) as z:
                self.assertIn(source.name,z.namelist())
                self.assertIn("README.md",z.namelist())
                self.assertNotIn("build/omit.aux",z.namelist())
                self.assertIn(r"\section{A \& B}",z.read(source.name).decode())


if __name__=="__main__": unittest.main()

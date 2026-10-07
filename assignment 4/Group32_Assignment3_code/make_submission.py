"""Create the required Assignment 3 code ZIP, including measured compact outputs."""
from pathlib import Path
import zipfile


def main():
    code=Path(__file__).resolve().parent
    destination=code.parent/"output"/"Group32_Assignment3_code.zip"
    destination.parent.mkdir(parents=True,exist_ok=True)
    source=[p for p in code.rglob("*.py") if "__pycache__" not in p.parts and ".venv" not in p.parts]
    source.extend([code/"README.md",code/"requirements.txt"])
    source.extend((code/"results").glob("*.json")); source.extend((code/"results").glob("*.csv"))
    source.extend((code/"results"/"figures").glob("*.png"))
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(source)):
            relative=path.relative_to(code)
            if relative.parts[0]=="results": relative=Path("reference_results",*relative.parts[1:])
            archive.write(path,Path(code.name)/relative)
        latex=code.parent/"output"/"latex"/"assignment3"
        for path in sorted(latex.rglob("*")):
            if path.is_file() and "build" not in path.relative_to(latex).parts:
                archive.write(path,Path(code.name)/"report_latex"/path.relative_to(latex))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(name.startswith(code.name+"/") for name in archive.namelist())
    print(f"Assignment 3 code ZIP: {destination} ({destination.stat().st_size/1024/1024:.2f} MiB)")


if __name__=="__main__": main()

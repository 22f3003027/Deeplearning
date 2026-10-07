"""Package only submission source and compact results, preserving required names."""
from pathlib import Path
import zipfile


def main():
    code=Path(__file__).resolve().parent
    destination=code.parent/"output"/"Group32_Assignment4_code.zip"
    destination.parent.mkdir(parents=True,exist_ok=True)
    source=[p for p in code.rglob("*.py") if "__pycache__" not in p.parts and ".venv" not in p.parts]
    source.extend([code/"README.md",code/"requirements.txt"])
    for pattern in ("*.json","*.csv","*.txt"):
        source.extend((code/"results").glob(pattern))
    source.extend((code/"results"/"figures").glob("*.png"))
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(set(source)):
            relative=path.relative_to(code)
            # Historical outputs must not block a fresh default experiment run after extraction.
            if relative.parts[0]=="results":
                relative=Path("reference_results",*relative.parts[1:])
            archive.write(path,Path(code.name)/relative)
        a3=code.parent/"Group32_Assignment3_code"/"results"
        for path in sorted([*a3.glob("*.json"),*a3.glob("*.csv"),*(a3/"figures").glob("*.png")]):
            archive.write(path,Path(code.name)/"assignment3_reference"/path.relative_to(a3))
        latex=code.parent/"output"/"latex"/"assignment4"
        for path in sorted(latex.rglob("*")):
            if path.is_file() and "build" not in path.relative_to(latex).parts:
                archive.write(path,Path(code.name)/"report_latex"/path.relative_to(latex))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(n.startswith("Group32_Assignment4_code/") for n in archive.namelist())
    print(f"Code submission: {destination} ({destination.stat().st_size/1024/1024:.2f} MiB)")


if __name__=="__main__":
    main()

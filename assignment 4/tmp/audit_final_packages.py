from pathlib import Path
import ast
import json
import re
import zipfile

root = Path(__file__).resolve().parent.parent
summary = {}
for assignment in (3, 4):
    name = f'Group32_Assignment{assignment}'
    latex = root / f'output/{name}_report_latex.zip'
    with zipfile.ZipFile(latex) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert all(not p.startswith('/') and '..' not in Path(p).parts for p in names)
        assert not any('build/' in p for p in names)
        source = archive.read(f'{name}_report.tex').decode('utf-8')
        images = re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', source)
        assert images and all(f'figures/{p}' in names for p in images)
        assert archive.read(f'{name}_report.tex') == (root / f'output/latex/assignment{assignment}/{name}_report.tex').read_bytes()
    code = root / f'output/{name}_code.zip'
    with zipfile.ZipFile(code) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert all(p.startswith(f'{name}_code/') for p in names)
        assert not any(p.endswith(('.npz', '.pt')) or '.venv/' in p or '/build/' in p for p in names)
        python_files = [p for p in names if p.endswith('.py')]
        for p in python_files:
            ast.parse(archive.read(p).decode('utf-8'), filename=p)
        stored = json.loads(archive.read(f'{name}_code/reference_results/metrics.json'))
        assert stored['status'] == 'complete'
        assert json.loads(archive.read(f'{name}_code/reference_results/verification.json'))['status'] == 'passed'
        assert archive.read(f'{name}_code/report_latex/{name}_report.tex') == (root / f'output/latex/assignment{assignment}/{name}_report.tex').read_bytes()
        if assignment == 4:
            reference = json.loads(archive.read(f'{name}_code/assignment3_reference/assignment3_best.json'))
            assert reference == stored['assignment3']
    summary[str(assignment)] = {'package_integrity':'passed', 'python_sources':len(python_files), 'latex_figures':len(images)}
print(json.dumps(summary,indent=2))
(root/'tmp/pdfs/latex-final/package_audit.json').write_text(json.dumps(summary,indent=2))

from pathlib import Path
import json
import fitz
from PIL import Image, ImageDraw

workspace = Path(__file__).resolve().parent.parent
root = workspace / 'tmp/pdfs/latex-final'
summary = {}
for assignment in (3, 4):
    pdf = workspace / f'output/pdf/Group32_Assignment{assignment}_report.pdf'
    folder = root / f'assignment{assignment}'
    folder.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf)
    thumbs = []
    bounds = []
    for index, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
        filename = folder / f'page-{index+1:02}.png'
        pixmap.save(filename)
        thumb = Image.open(filename).convert('RGB')
        thumb.thumbnail((440, 622))
        thumbs.append(thumb)
        for block in page.get_text('dict')['blocks']:
            if block['type'] == 0:
                box = block['bbox']
                if box[0] < 20 or box[2] > page.rect.width - 20:
                    bounds.append({'page': index+1, 'bbox': box})
    for start in range(0, len(thumbs), 6):
        sheet = Image.new('RGB', (920, 1980), '#dddddd')
        draw = ImageDraw.Draw(sheet)
        for offset, thumb in enumerate(thumbs[start:start+6]):
            x = 10 + (offset % 2) * 460
            y = 30 + (offset // 2) * 660
            sheet.paste(thumb, (x, y))
            draw.text((x, y-20), f'Assignment {assignment}, page {start+offset+1}', fill='black')
        sheet.save(folder / f'contact-{start//6+1}.png')
    summary[str(assignment)] = {'pages':len(document), 'out_of_page_text':bounds}
(root/'qa_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

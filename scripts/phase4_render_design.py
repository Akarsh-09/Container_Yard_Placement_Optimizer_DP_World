"""Development-only reference pagination of docs/design.md (A4, 11pt)."""
import html
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
from pypdf import PdfReader
import pymupdf as fitz
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/pdfs';OUT.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyDesign',fontName='Helvetica',fontSize=11,leading=13.2,spaceAfter=5))
styles.add(ParagraphStyle(name='SectionDesign',fontName='Helvetica-Bold',fontSize=11,leading=13,spaceBefore=7,spaceAfter=4,keepWithNext=True))
styles.add(ParagraphStyle(name='CellDesign',fontName='Helvetica',fontSize=7.2,leading=8.7))
styles.add(ParagraphStyle(name='CodeDesign',fontName='Courier',fontSize=8,leading=10,spaceAfter=5))
def inline(text):
    text=html.escape(text)
    codes=[]
    def keep_code(match):
        codes.append(match.group(1))
        return f"CODETOKEN{len(codes)-1}Z"
    text=re.sub(r'`([^`]+)`',keep_code,text)
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<link href="\2" color="#234e83">\1</link>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'\*([^*]+)\*',r'<i>\1</i>',text)
    for i,code in enumerate(codes):
        text=text.replace(f"CODETOKEN{i}Z",f'<font name="Courier" size="8">{code}</font>')
    return text

lines=(ROOT/'docs/design.md').read_text().splitlines()
flow=[];i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('# '):flow.append(Paragraph(inline(line[2:]),styles['Title']));i+=1
    elif line.startswith('## '):flow.append(Paragraph(inline(line[3:]),styles['SectionDesign']));i+=1
    elif line.startswith('```'):
        code=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
        flow.append(Preformatted('\n'.join(code),styles['CodeDesign']));i+=1
    elif line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            values=[v.strip() for v in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',v) for v in values):rows.append([Paragraph(inline(v),styles['CellDesign']) for v in values])
            i+=1
        width=A4[0]-32*mm
        n=len(rows[0]);first=86 if n==6 else 72
        widths=[first]+[(width-first)/(n-1)]*(n-1)
        table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#72869b')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d1d9e2'))]))
        flow.extend([table,Spacer(1,5)])
    else:
        paragraph=[]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')):paragraph.append(lines[i]);i+=1
        flow.append(Paragraph(inline(' '.join(paragraph)),styles['BodyDesign']))
def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#697889'))
    canvas.drawString(16*mm,12*mm,'Design review - reference Markdown pagination')
    canvas.drawRightString(A4[0]-16*mm,12*mm,str(doc.page))
path=OUT/'design-review.pdf'
SimpleDocTemplate(str(path),pagesize=A4,leftMargin=16*mm,rightMargin=16*mm,topMargin=15*mm,bottomMargin=18*mm).build(flow,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(path)
print('PAGES',len(reader.pages))
pdf=fitz.open(path)
for i,page in enumerate(pdf):
    page.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(OUT/f'design-{i+1}.png')
print('WORD COUNT',len((ROOT/'docs/design.md').read_text().split()))

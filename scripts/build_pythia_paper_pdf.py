"""Build the editable TeX manuscript and a reportlab reader-map PDF edition."""
from __future__ import annotations

import subprocess
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "papers/tex/PYTHIA_METIS_PAPER.tex"
TEMP = ROOT / "tmp/pdfs"
OUTPUT = ROOT / "papers/pdf/pythia_metis_working_paper_2026.pdf"


def _reader_map(path: Path) -> None:
    blue = HexColor("#183246")
    teal = HexColor("#008A88")
    pale = HexColor("#EAF5F4")
    grey = HexColor("#526675")
    width, height = A4
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFillColor(blue)
    pdf.rect(0, height - 15, width, 15, fill=1, stroke=0)
    pdf.setFillColor(teal)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(54, height - 65, "PYTHIA / METIS  |  RESEARCH MAP")
    pdf.setFillColor(blue)
    pdf.setFont("Helvetica-Bold", 29)
    pdf.drawString(54, height - 112, "Where prediction ends")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(54, height - 139, "The floor, the assumptions, and the work still ahead")
    pdf.setStrokeColor(teal)
    pdf.setLineWidth(2)
    pdf.line(54, height - 160, width - 54, height - 160)

    def box(top: float, label: str, lines: tuple[str, ...]) -> None:
        box_height = 129
        pdf.setFillColor(pale)
        pdf.roundRect(54, top - box_height, width - 108, box_height, 10,
                      fill=1, stroke=0)
        pdf.setFillColor(teal)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(72, top - 27, label)
        pdf.setFillColor(blue)
        pdf.setFont("Helvetica", 11)
        for index, line in enumerate(lines):
            pdf.drawString(72, top - 51 - 19 * index, line)

    box(height - 190, "01  THE MATHEMATICAL OBJECT", (
        "A train-free lower witness for finite-design Bayes risk.",
        "MAE, q10/q90 pinball, and W15/W30/W60 miss decisions.",
        "Conditional confidence needs an audited metric and independence.",
    ))
    box(height - 337, "02  THE EVIDENCE", (
        "Ten known-risk designs; 200 fresh draws; five shared losses.",
        "1,000/1,000 lower-bound checks valid, but no universal win.",
        "Mayo 28.85-minute MAE result remains calibrated, retrospective.",
    ))
    box(height - 484, "03  THE FUTURE SYSTEM", (
        "Metis would search representations and witnesses behind a gate.",
        "External predictors belong in Horos; Pythia itself does not train.",
        "Prospective Mayo confirmation and a proof audit remain necessary.",
    ))
    pdf.setFillColor(blue)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(54, 123, "The negative result matters")
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(grey)
    pdf.drawString(54, 104, "Bentkus beats the adaptive envelope on most quantile trials;")
    pdf.drawString(54, 87, "cross-fitted local estimators beat the witness center on point error.")
    pdf.setStrokeColor(grey)
    pdf.line(54, 61, width - 54, 61)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(54, 46, "Working paper, 13 September 2026  |  Full manuscript begins next page")
    pdf.save()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["tectonic", "--outdir", str(TEMP), str(SOURCE)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    if "warning:" in completed.stderr:
        print(completed.stderr)
    article = TEMP / "PYTHIA_METIS_PAPER.pdf"
    reader_map = TEMP / "pythia_reader_map.pdf"
    _reader_map(reader_map)
    writer = PdfWriter()
    writer.append(str(reader_map))
    writer.append(str(article))
    writer.add_metadata({"/Title": "Where prediction ends: Pythia and Metis",
                         "/Author": "Itamar Zernitsky and collaborators"})
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    print(f"{OUTPUT}: {len(PdfReader(str(OUTPUT)).pages)} pages")


if __name__ == "__main__":
    main()

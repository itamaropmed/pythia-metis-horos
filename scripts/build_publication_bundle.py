"""Combine the frozen Pythia paper and supplement into a reader PDF.

The two component PDFs are preserved unchanged. Run from any directory.
"""

from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "papers/pdf/PYTHIA_ICML_COMPLETE_WITH_SUPPLEMENT.pdf"
PARTS = (
    ("Main paper", ROOT / "papers/pdf/PYTHIA_ICML_SUBMISSION.pdf"),
    ("Supplementary material", ROOT / "papers/pdf/PYTHIA_ICML_SUPPLEMENT.pdf"),
)


def main() -> None:
    writer = PdfWriter()
    expected_pages = 0
    for title, path in PARTS:
        if not path.is_file():
            raise FileNotFoundError(path)
        count = len(PdfReader(path).pages)
        if count == 0:
            raise ValueError(f"Empty PDF: {path}")
        writer.append(str(path), outline_item=title)
        expected_pages += count
    writer.add_metadata(
        {
            "/Title": "Pythia: Certified Floors on Decision Risk — Paper and Supplement",
            "/Subject": "Combined research reader copy; not an official conference template",
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as handle:
        writer.write(handle)
    actual_pages = len(PdfReader(OUTPUT).pages)
    if actual_pages != expected_pages:
        raise RuntimeError(f"PDF page mismatch: {actual_pages} != {expected_pages}")
    print(f"{OUTPUT}: {actual_pages} pages")


if __name__ == "__main__":
    main()

from pypdf import PdfReader
from pathlib import Path

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    pages_text = []
    for p in reader.pages:
        pages_text.append(p.extract_text() or "")
    return "\n".join(pages_text)

if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1])
    print(extract_text_from_pdf(p))

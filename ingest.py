"""Step 1 + 2: read the menu PDF and split it into one chunk per dish."""
import re
from dataclasses import dataclass, asdict, field
import pdfplumber

HEADER = re.compile(r"^DESI DASTARKHWAN\b")
FOOTER = re.compile(r"^Prices in PKR")
DISH = re.compile(r"^(?P<name>.+?)(?P<veg>\s+\(V\))?\s+Rs\.\s*(?P<price>[\d,]+)$")


@dataclass
class Chunk:
    id: str
    kind: str                 # "dish" or "glossary"
    name: str
    section: str = ""
    price: int | None = None
    vegetarian: bool = False
    ingredients: str = ""
    page: int = 0
    text: str = field(default="", repr=False)   # what gets embedded and shown to the LLM

    def to_dict(self):
        return asdict(self)


def _clean_lines(page) -> list[str]:
    lines = (page.extract_text() or "").splitlines()
    return [l.strip() for l in lines if l.strip()
            and not HEADER.match(l.strip()) and not FOOTER.match(l.strip())]


def _dish_text(c: Chunk) -> str:
    veg = "vegetarian" if c.vegetarian else "non-vegetarian"
    return (f"{c.name} | Section: {c.section} | {veg} | Price: Rs. {c.price}\n"
            f"Ingredients: {c.ingredients}")


def parse_pdf(path: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            lines = _clean_lines(page)
            if not any(DISH.match(l) for l in lines):
                continue  # cover, welcome page, glossary

            # A page that starts a new section: heading, subtitle, then a dish line.
            if len(lines) >= 3 and not DISH.match(lines[0]) and not DISH.match(lines[1]) \
                    and DISH.match(lines[2]):
                section = lines[0]
                lines = lines[2:]
            cur: Chunk | None = None
            for line in lines:
                m = DISH.match(line)
                if m:
                    if cur:
                        chunks.append(cur)
                    cur = Chunk(id=f"dish-{len(chunks)+1:03d}", kind="dish",
                                name=m["name"].strip(), section=section,
                                price=int(m["price"].replace(",", "")),
                                vegetarian=bool(m["veg"]), page=pno)
                elif cur is not None:
                    part = line.removeprefix("Ingredients:").strip()
                    cur.ingredients = (cur.ingredients + " " + part).strip()
            if cur:
                chunks.append(cur)

        # Glossary page: a two-column table (term | meaning)
        for pno, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            if "Glossary" not in txt:
                continue
            # Each glossary row sits between two thin horizontal rules; terms are bold.
            words = page.extract_words(extra_attrs=["fontname"])
            ys = sorted({round(l["top"], 1) for l in page.lines
                         if abs(l["top"] - l["bottom"]) < 1 and l["x1"] - l["x0"] > 200})
            terms: list[list[str]] = []
            bounds = [min(w["top"] for w in words if "Bold" in w["fontname"] and w["top"] > 100) - 8] + ys
            for lo, hi in zip(bounds, bounds[1:]):
                band = sorted((w for w in words if lo < w["top"] < hi), key=lambda w: (round(w["top"]), w["x0"]))
                term = " ".join(w["text"] for w in band if "Bold" in w["fontname"] and w["x0"] < 150)
                meaning = " ".join(w["text"] for w in band if not ("Bold" in w["fontname"] and w["x0"] < 150))
                if term and meaning:
                    terms.append([term, meaning])
            for term, meaning in terms:
                if meaning.startswith("Prices in PKR"):
                    continue
                chunks.append(Chunk(id=f"gloss-{len(chunks)+1:03d}", kind="glossary",
                                    name=term, section="Glossary", ingredients=meaning,
                                    page=pno, text=f"{term} (glossary): {meaning}"))

    for c in chunks:
        if c.kind == "dish":
            c.text = _dish_text(c)
    return chunks


if __name__ == "__main__":
    import sys
    cs = parse_pdf(sys.argv[1] if len(sys.argv) > 1 else "Desi_Restaurant_Menu.pdf")
    dishes = [c for c in cs if c.kind == "dish"]
    print(len(dishes), "dishes,", len(cs) - len(dishes), "glossary terms")
    for c in cs[:2] + dishes[-2:] + cs[-1:]:
        print("-" * 60); print(c.section, "|", c.text)

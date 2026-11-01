"""usage: stack.py out.jpg label=img.jpg ...  (rows); a,b side by side with '|'"""
import sys
from PIL import Image, ImageDraw
rows = []
for arg in sys.argv[2:]:
    cells = []
    for part in arg.split("|"):
        label, path = part.split("=", 1)
        im = Image.open(path).convert("RGB")
        g = ImageDraw.Draw(im); w = 9 * len(label) + 16
        g.rectangle((8, 8, 8 + w, 34), fill="black"); g.text((16, 15), label, fill="white")
        cells.append(im)
    h = max(c.height for c in cells)
    cells = [c.resize((round(c.width * h / c.height), h)) for c in cells]
    row = Image.new("RGB", (sum(c.width for c in cells) + 6 * (len(cells) - 1), h), "white")
    x = 0
    for c in cells: row.paste(c, (x, 0)); x += c.width + 6
    rows.append(row)
W = max(r.width for r in rows)
rows = [r.resize((W, round(r.height * W / r.width))) for r in rows]
out = Image.new("RGB", (W, sum(r.height for r in rows) + 6 * (len(rows) - 1)), "white")
y = 0
for r in rows: out.paste(r, (0, y)); y += r.height + 6
out.save(sys.argv[1], quality=90)

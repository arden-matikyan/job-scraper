from bs4 import BeautifulSoup
html = open("_leidos.html", encoding="utf-8").read()
soup = BeautifulSoup(html, "html.parser")
import re
links = soup.select('a[href*="/jobs/"]')
links = [a for a in links if re.search(r"/jobs/\d+", a.get("href",""))]
print("job-detail anchors:", len(links))
a = links[0]
print("first anchor href:", a.get("href"))
print("first anchor text:", repr(a.get_text(strip=True)))
# Walk up to find the row container
row = a
for _ in range(6):
    row = row.parent
    if row.name in ("tr","li") or (row.get("class") and any("job" in c.lower() or "row" in c.lower() or "result" in c.lower() for c in row.get("class"))):
        break
print("row tag:", row.name, "classes:", row.get("class"))
print("---- row text (whitespace-collapsed) ----")
print(re.sub(r"\s+"," ", row.get_text(" ", strip=True))[:400])
print("---- row child elements w/ classes ----")
for el in row.find_all(True, recursive=True)[:25]:
    cls = el.get("class")
    txt = el.get_text(" ", strip=True)[:50]
    if cls or el.name in ("a","td","span","div"):
        print(f"  <{el.name} class={cls}> {txt!r}")

import glob, os
pdfs = sorted(glob.glob("reports/NVDA_memo*.pdf"), key=os.path.getmtime, reverse=True)
p = pdfs[0]
data = open(p, "rb").read()
print("Newest report:", p, "size:", len(data), "bytes")
print("Has embedded image (chart):", b"/Image" in data or b"/XObject" in data)
print("Chart present if size > 40KB:", len(data) > 40000)

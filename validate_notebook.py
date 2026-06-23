import json

with open("takemeter_notebook.ipynb") as f:
    nb = json.load(f)

cells = nb["cells"]
code_cells = [c for c in cells if c["cell_type"] == "code"]
md_cells   = [c for c in cells if c["cell_type"] == "markdown"]

print(f"Valid notebook.")
print(f"  Format: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"  Total cells: {len(cells)}")
print(f"  Code cells: {len(code_cells)}")
print(f"  Markdown cells: {len(md_cells)}")

for i, c in enumerate(cells):
    assert c["source"], f"Cell {i} has empty source"
print("All cells have content: OK")

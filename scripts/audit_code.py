import ast
import glob
from pathlib import Path

print("=" * 60)
print("STATIC AUDIT: UNUSED IMPORTS & VARIABLES")
print("=" * 60)

for path_str in sorted(glob.glob("src/*.py")):
    path = Path(path_str)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    tree = ast.parse(content, filename=str(path))
    
    # 1. Unused imports
    imported = set()
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imported.add(n.asname or n.name)
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                imported.add(n.asname or n.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
    unused_imports = imported - used_names
    
    # 2. Unused variables in functions
    func_issues = []
    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        assigned = set()
        loaded = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assigned.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    loaded.add(node.id)
        params = {a.arg for a in func.args.args + func.args.posonlyargs + func.args.kwonlyargs}
        unused_vars = assigned - loaded - params - {'_', 'engine'}
        if unused_vars:
            func_issues.append((func.name, unused_vars))
            
    print(f"\n[FILE] {path}")
    print(f"  Unused Imports   : {unused_imports if unused_imports else 'NONE (Clean)'}")
    print(f"  Unused Variables : {func_issues if func_issues else 'NONE (Clean)'}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)

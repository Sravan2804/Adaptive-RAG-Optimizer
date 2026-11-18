import ast
import tokenize
from pathlib import Path

print("=" * 60)
print("STATIC AUDIT: UNUSED IMPORTS & VARIABLES")
print("=" * 60)

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
src_dir = repo_root / "src"
py_files = sorted(src_dir.glob("*.py"))

if not py_files:
    print(f"[ERROR] No Python source files found in {src_dir}")
else:
    for path in py_files:
        with tokenize.open(path) as f:
            content = f.read()
        tree = ast.parse(content, filename=str(path))
        
        # 1. Module-level __all__ detection: each assignment replaces previously recorded names
        exported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                            exported_names = {
                                elt.value for elt in node.value.elts
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                            }

        # 2. Collect module-level imports, recursively inspecting module control-flow (If, Try, etc.)
        # Stop traversal at function and class definitions
        imported_entries = [] # list of (bound_name, qual_name, diag)
        
        def collect_module_imports(nodes):
            for node in nodes:
                if isinstance(node, ast.Import):
                    for n in node.names:
                        bound = n.asname or n.name.split(".")[0]
                        qual = n.name
                        diag = n.name if not n.asname else f"{n.name} as {n.asname}"
                        imported_entries.append((bound, qual, diag))
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "__future__" and node.level == 0:
                        continue
                    if node.module:
                        mod = ("." * node.level) + node.module
                        for n in node.names:
                            bound = n.asname or n.name
                            qual = f"{mod}.{n.name}"
                            diag = qual if not n.asname else f"{qual} as {n.asname}"
                            imported_entries.append((bound, qual, diag))
                    else:
                        mod = "." * node.level
                        for n in node.names:
                            bound = n.asname or n.name
                            qual = f"{mod}{n.name}"
                            diag = qual if not n.asname else f"{qual} as {n.asname}"
                            imported_entries.append((bound, qual, diag))
                elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    sub_nodes = list(ast.iter_child_nodes(node))
                    # For Try, also descend into ExceptHandler bodies (which are not in iter_child_nodes' flat list)
                    handler_bodies = []
                    if isinstance(node, ast.Try):
                        for h in node.handlers:
                            handler_bodies.extend(h.body)
                    collect_module_imports(
                        [n for n in sub_nodes if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                        + [n for n in handler_bodies if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                    )

        collect_module_imports(tree.body)

        used_module_names = set(exported_names)
        qualified_attribute_reads = set()
        func_issues = []

        def extract_loads_from_expr(expr, target_set):
            if expr is None:
                return
            for n in ast.walk(expr):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    target_set.add(n.id)
                elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                    qualified_attribute_reads.add((n.value.id, n.attr))
                    if isinstance(n.ctx, ast.Load):
                        target_set.add(n.value.id)

        def get_params(fn):
            p = {a.arg for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs}
            if fn.args.vararg:
                p.add(fn.args.vararg.arg)
            if fn.args.kwarg:
                p.add(fn.args.kwarg.arg)
            return p

        def analyze_function_scope(fn, outer_scope):
            # Evaluate decorators, defaults, annotations in defining/outer scope
            for d in fn.decorator_list:
                extract_loads_from_expr(d, outer_scope["loaded"])
            for def_val in fn.args.defaults + fn.args.kw_defaults:
                if def_val:
                    extract_loads_from_expr(def_val, outer_scope["loaded"])
            if fn.returns:
                extract_loads_from_expr(fn.returns, outer_scope["loaded"])
            for arg in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs:
                if arg.annotation:
                    extract_loads_from_expr(arg.annotation, outer_scope["loaded"])

            params = get_params(fn)
            assigned = set()
            loaded = set()
            global_names = set()
            nonlocal_names = set()
            nested_fns = []
            nested_func_names = set()
            classes_in_func = []

            for node in ast.walk(fn):
                if isinstance(node, ast.Global):
                    global_names.update(node.names)
                elif isinstance(node, ast.Nonlocal):
                    nonlocal_names.update(node.names)

            def walk_body(nodes):
                for child in nodes:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        nested_fns.append(child)
                        nested_func_names.add(child.name)
                        assigned.add(child.name)
                        continue
                    elif isinstance(child, ast.ClassDef):
                        classes_in_func.append(child)
                        assigned.add(child.name)
                        continue
                    elif isinstance(child, ast.Import):
                        for n in child.names:
                            bound = n.asname or n.name.split(".")[0]
                            assigned.add(bound)
                    elif isinstance(child, ast.ImportFrom):
                        for n in child.names:
                            bound = n.asname or n.name
                            assigned.add(bound)
                    elif isinstance(child, ast.Name):
                        if isinstance(child.ctx, ast.Store):
                            if child.id in global_names:
                                used_module_names.add(child.id)
                            elif child.id in nonlocal_names:
                                outer_scope["loaded"].add(child.id)
                            else:
                                assigned.add(child.id)
                        elif isinstance(child.ctx, ast.Load):
                            loaded.add(child.id)
                    elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                        qualified_attribute_reads.add((child.value.id, child.attr))
                        if isinstance(child.ctx, ast.Load):
                            loaded.add(child.value.id)
                    elif isinstance(child, ast.ExceptHandler):
                        if child.name:
                            assigned.add(child.name)
                        if child.type:
                            walk_body([child.type])
                        walk_body(child.body)
                        continue
                    elif isinstance(child, ast.AugAssign):
                        if isinstance(child.target, ast.Name):
                            assigned.add(child.target.id)
                            loaded.add(child.target.id)
                        else:
                            extract_loads_from_expr(child.target, loaded)
                        extract_loads_from_expr(child.value, loaded)
                        continue
                    elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                        # Leftmost iterable evaluated in enclosing scope
                        first_gen = child.generators[0]
                        extract_loads_from_expr(first_gen.iter, loaded)
                        
                        comp_assigned = set()
                        for gen in child.generators:
                            for n in ast.walk(gen.target):
                                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                                    comp_assigned.add(n.id)
                            if gen is not first_gen:
                                for n in ast.walk(gen.iter):
                                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in comp_assigned:
                                        loaded.add(n.id)
                            for cond in gen.ifs:
                                for n in ast.walk(cond):
                                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in comp_assigned:
                                        loaded.add(n.id)
                        comp_elts = [child.elt] if hasattr(child, "elt") else [child.key, child.value]
                        for elt in comp_elts:
                            for n in ast.walk(elt):
                                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in comp_assigned:
                                    loaded.add(n.id)
                        continue
                    
                    walk_body(list(ast.iter_child_nodes(child)))

            walk_body(fn.body)

            func_scope_context = {
                "assigned": assigned | params,
                "loaded": loaded
            }

            captured_by_inner = set()
            for nested in nested_fns:
                inner_escaped_reads = analyze_function_scope(nested, func_scope_context)
                for read_name in inner_escaped_reads:
                    if read_name in assigned and read_name not in params:
                        captured_by_inner.add(read_name)
                    elif read_name not in (assigned | params):
                        loaded.add(read_name)

            for cls in classes_in_func:
                process_class(cls, func_scope_context)

            unused = (assigned - loaded - captured_by_inner - params - nested_func_names) - {"_", "self", "cls", "engine"}
            if unused:
                func_issues.append((fn.name, unused))

            escaped_reads = set()
            for name in loaded:
                if name not in (assigned | params):
                    escaped_reads.add(name)
                    outer_scope["loaded"].add(name)

            return escaped_reads

        def process_class(cls_node, defining_scope):
            for b in cls_node.bases:
                extract_loads_from_expr(b, defining_scope["loaded"])
            for kw in cls_node.keywords:
                extract_loads_from_expr(kw.value, defining_scope["loaded"])
            for d in cls_node.decorator_list:
                extract_loads_from_expr(d, defining_scope["loaded"])

            class_assigned = set()
            for item in cls_node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    analyze_function_scope(item, defining_scope)
                elif isinstance(item, ast.ClassDef):
                    process_class(item, defining_scope)
                else:
                    for n in ast.walk(item):
                        if isinstance(n, ast.Name):
                            if isinstance(n.ctx, ast.Store):
                                class_assigned.add(n.id)
                            elif isinstance(n.ctx, ast.Load):
                                if n.id not in class_assigned:
                                    defining_scope["loaded"].add(n.id)
                        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                            qualified_attribute_reads.add((n.value.id, n.attr))
                            if isinstance(n.ctx, ast.Load) and n.value.id not in class_assigned:
                                defining_scope["loaded"].add(n.value.id)

        module_scope = {"loaded": used_module_names, "assigned": set()}

        def dispatch_module_statements(nodes):
            for stmt in nodes:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    analyze_function_scope(stmt, module_scope)
                elif isinstance(stmt, ast.ClassDef):
                    process_class(stmt, module_scope)
                elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    for child in ast.iter_child_nodes(stmt):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            analyze_function_scope(child, module_scope)
                        elif isinstance(child, ast.ClassDef):
                            process_class(child, module_scope)
                        elif isinstance(child, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                            dispatch_module_statements([child])
                        else:
                            extract_loads_from_expr(child, used_module_names)
                else:
                    extract_loads_from_expr(stmt, used_module_names)

        dispatch_module_statements(tree.body)

        unused_imports = set()
        for bound, qual, diag in imported_entries:
            if bound not in used_module_names:
                unused_imports.add(diag)
            elif "." in qual and not diag.endswith(f" as {bound}"):
                parts = qual.split(".")
                if len(parts) >= 2:
                    sub_attr = parts[1]
                    sharing = [e for e in imported_entries if e[0] == bound]
                    if len(sharing) > 1 and (parts[0], sub_attr) not in qualified_attribute_reads:
                        unused_imports.add(diag)

        print(f"\n[FILE] {path}")
        print(f"  Unused Imports   : {unused_imports if unused_imports else 'NONE (Clean)'}")
        print(f"  Unused Variables : {func_issues if func_issues else 'NONE (Clean)'}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)

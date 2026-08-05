"""Static scope check for a notebook: catches NameErrors without executing."""
import ast, json, builtins, sys

def strip_magic(line):
    stripped = line.lstrip()
    if stripped.startswith(('%', '!', '?')) or line.rstrip().endswith('??'):
        return ('pass\n' if line.endswith('\n') else 'pass')
    return line

def bound_names(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.add(n.name)
        elif isinstance(n, ast.arg): out.add(n.arg)
        elif isinstance(n, ast.alias): out.add((n.asname or n.name).split('.')[0])
        elif isinstance(n, ast.ExceptHandler) and n.name: out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)): out.update(n.names)
    return out

def check(path):
    nb = json.load(open(path))
    defined, deferred, problems = set(dir(builtins)), [], 0
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code': continue
        src = ''.join(strip_magic(l) for l in c['source'])
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f"  cell {i:2d}: SYNTAX ERROR line {e.lineno}: {e.msg}"); problems += 1; continue
        local = bound_names(tree)
        inner = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for m in ast.walk(n):
                    if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load): inner.add(m.id)
        top = {n.id for n in ast.walk(tree)
               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - inner
        miss = sorted(top - defined - local)
        if miss:
            print(f"  cell {i:2d}: MODULE-LEVEL MISSING {miss}"); problems += 1
        deferred += [(i, n) for n in inner]
        defined |= local
    still = sorted({n for _, n in deferred} - defined)
    if still:
        print(f"  used inside functions, never defined: {still}"); problems += 1
    n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    print(f"{path}: {'OK' if problems == 0 else str(problems) + ' problem(s)'} "
          f"({n_code} code cells)")
    return problems

if __name__ == '__main__':
    sys.exit(min(1, sum(check(p) for p in sys.argv[1:])))

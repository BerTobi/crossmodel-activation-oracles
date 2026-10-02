"""
Implement TRUNCATION for the adapter-free cross-model path (subject wider than oracle), e.g. Qwen3-8B (4096) read by a
Qwen3-4B oracle (2560): keep the first d_oracle dims. Lossy by design — this is the "truncate" rung of the pad/truncate
adaptation ladder. Replaces the deliberate NotImplementedError in four places, mirrors it in the eval shim, updates the
two tests that asserted the old behaviour, and corrects crossmodel_patch.py so fresh clones match.

Usage: python apply_truncation_patch.py [repo=activation_oracles] [patch_script=crossmodel_patch.py | /dev/null]
Run from the code/ directory (collect_readouts_xm.py and the tests are resolved relative to cwd when present).
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "activation_oracles")
PATCH = Path(sys.argv[2] if len(sys.argv) > 2 else "crossmodel_patch.py")
NL = chr(10)

# ---- A. dataset_utils.py: freshly captured vectors ----
A_OLD = NL.join([
    "            if d_s < d_o:",
    "                vectors = torch.nn.functional.pad(vectors, (0, d_o - d_s))  # zero-pad trailing dims (lossless)",
    "            else:",
    "                raise NotImplementedError(",
    '                    f"Subject hidden_size {d_s} > oracle hidden_size {d_o}: truncation is lossy and deliberately not "',
    '                    "implemented here (see the adaptation-ladder arm of the plan)."',
    "                )",
]) + NL
A_NEW = NL.join([
    "            if d_s < d_o:",
    "                vectors = torch.nn.functional.pad(vectors, (0, d_o - d_s))  # zero-pad trailing dims (lossless)",
    "            else:",
    "                vectors = vectors[:, :d_o].contiguous()  # [truncate] keep the first d_o dims (lossy; adaptation-ladder arm)",
]) + NL

# ---- B. dataset_utils.py: precomputed vectors ([pad_precomputed] block) ----
B_OLD = NL.join([
    "            v = dp.steering_vectors",
    "            if v is not None and v.shape[1] != d_o:",
    "                if v.shape[1] > d_o:",
    "                    raise NotImplementedError(",
    '                        f"Precomputed subject vectors ({v.shape[1]}) wider than oracle ({d_o}): truncation not implemented"',
    "                    )",
    "                dp = dp.model_copy(deep=True)",
    "                dp.steering_vectors = torch.nn.functional.pad(v, (0, d_o - v.shape[1]))",
]) + NL
B_NEW = NL.join([
    "            v = dp.steering_vectors",
    "            if v is not None and v.shape[1] != d_o:",
    "                dp = dp.model_copy(deep=True)",
    "                if v.shape[1] > d_o:",
    "                    dp.steering_vectors = v[:, :d_o].contiguous()  # [truncate] lossy: keep the first d_o dims",
    "                else:",
    "                    dp.steering_vectors = torch.nn.functional.pad(v, (0, d_o - v.shape[1]))",
]) + NL

# ---- C. sft.py: dimension check ----
C_OLD = NL.join([
    "        if d_subject < d_oracle:",
    '            print(f"Cross-model: zero-padding subject activations {d_subject} -> {d_oracle} (lossless)")',
    "        elif d_subject > d_oracle:",
    "            raise NotImplementedError(",
    '                f"Subject hidden_size {d_subject} > oracle {d_oracle}: truncation is lossy and not implemented"',
    "            )",
]) + NL
C_NEW = NL.join([
    "        if d_subject < d_oracle:",
    '            print(f"Cross-model: zero-padding subject activations {d_subject} -> {d_oracle} (lossless)")',
    "        elif d_subject > d_oracle:",
    '            print(f"Cross-model: TRUNCATING subject activations {d_subject} -> {d_oracle} (lossy: keeps the first {d_oracle} dims)")',
]) + NL


def edit(path: Path, pairs, label):
    t = path.read_text(encoding="utf-8")
    done = 0
    for old, new in pairs:
        if old not in t and new in t:
            continue  # already applied
        c = t.count(old)
        assert c == 1, f"[{label}] expected exactly 1 match, found {c}: {old[:70]!r}"
        t = t.replace(old, new)
        done += 1
    path.write_text(t, encoding="utf-8")
    print(f"{label}: {done} edit(s) applied" if done else f"{label}: already up to date")


edit(REPO / "nl_probes" / "utils" / "dataset_utils.py", [(A_OLD, A_NEW), (B_OLD, B_NEW)], "dataset_utils.py")
edit(REPO / "nl_probes" / "sft.py", [(C_OLD, C_NEW)], "sft.py")

# ---- D. eval shim ----
D_OLD1 = '    """Zero-pad every datapoint\'s steering_vectors (K, d) -> (K, d_target). Lossless. Returns new list."""' + NL
D_NEW1 = '    """Zero-pad (d < d_target, lossless) or truncate (d > d_target, lossy) every datapoint\'s steering_vectors to (K, d_target)."""' + NL
D_OLD2 = NL.join([
    '        assert v.shape[1] < d_target, f"subject dim {v.shape[1]} > oracle dim {d_target}: truncation not supported"',
    "        new = dp.model_copy(deep=True)",
    "        new.steering_vectors = torch.nn.functional.pad(v, (0, d_target - v.shape[1]))",
]) + NL
D_NEW2 = NL.join([
    "        new = dp.model_copy(deep=True)",
    "        if v.shape[1] > d_target:",
    "            new.steering_vectors = v[:, :d_target].contiguous()  # truncate (lossy) - mirrors training",
    "        else:",
    "            new.steering_vectors = torch.nn.functional.pad(v, (0, d_target - v.shape[1]))",
]) + NL
D_OLD3 = '            print(f"[B] zero-padded activations {d_subject} -> {d_oracle}")' + NL
D_NEW3 = '            print(f"[B] {\'zero-padded\' if d_oracle > d_subject else \'TRUNCATED\'} activations {d_subject} -> {d_oracle}")' + NL
shim = Path("collect_readouts_xm.py")
if shim.exists():
    edit(shim, [(D_OLD1, D_NEW1), (D_OLD2, D_NEW2), (D_OLD3, D_NEW3)], "collect_readouts_xm.py")

# ---- E. tests that asserted the old NotImplementedError ----
T1 = Path("test_padding.py")
if T1.exists():
    T1_OLD = NL.join([
        "try:",
        "    with torch.no_grad():",
        "        materialize_missing_steering_vectors([dp], tok, oracle, subject_model=subject, subject_tokenizer=tok, oracle_hidden_size=d_s - 64)",
        '    chk(False, "subject wider than oracle raises NotImplementedError")',
        "except NotImplementedError:",
        '    chk(True, "subject wider than oracle raises NotImplementedError")',
    ]) + NL
    T1_NEW = NL.join([
        "with torch.no_grad():",
        "    tr = materialize_missing_steering_vectors([dp], tok, oracle, subject_model=subject, subject_tokenizer=tok, oracle_hidden_size=d_s - 64)",
        'chk(tr[0].steering_vectors.shape[1] == d_s - 64, "subject wider than oracle -> captured vectors truncated to oracle width")',
    ]) + NL
    edit(T1, [(T1_OLD, T1_NEW)], "test_padding.py")
T2 = Path("test_padding_precomputed.py")
if T2.exists():
    T2_OLD = NL.join([
        'print("=== wider precomputed vectors rejected ===")',
        "try:",
        "    with torch.no_grad():",
        "        materialize_missing_steering_vectors([dp_with_vectors(d_o)], None, oracle, subject_model=subject, subject_tokenizer=None, oracle_hidden_size=d_s)",
        '    chk(False, "raises NotImplementedError")',
        "except NotImplementedError:",
        '    chk(True, "raises NotImplementedError")',
    ]) + NL
    T2_NEW = NL.join([
        'print("=== wider precomputed vectors truncated ===")',
        "wide = dp_with_vectors(d_o)",
        "with torch.no_grad():",
        "    tr = materialize_missing_steering_vectors([wide], None, oracle, subject_model=subject, subject_tokenizer=None, oracle_hidden_size=d_s)",
        'chk(tuple(tr[0].steering_vectors.shape) == (3, d_s) and torch.equal(tr[0].steering_vectors, wide.steering_vectors[:, :d_s]), "truncated to oracle width, leading dims preserved")',
        'chk(wide.steering_vectors.shape[1] == d_o, "input untouched (deep copy)")',
    ]) + NL
    edit(T2, [(T2_OLD, T2_NEW)], "test_padding_precomputed.py")

# ---- F. patcher (fragments appear inside repr'd single-quoted literals) ----
if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only")
    sys.exit(0)
s = PATCH.read_text(encoding="utf-8")


def r(x):  # repr body without the enclosing quotes; the patcher's literals were written with !r
    q = repr(x)
    assert q[0] == "'" and q[-1] == "'", "fragment repr not single-quoted"
    return q[1:-1]


pairs = [(r(A_OLD), r(A_NEW)), (r(B_OLD), r(B_NEW)), (r(C_OLD), r(C_NEW))]
done = 0
for old, new in pairs:
    if old not in s and new in s:
        continue
    c = s.count(old)
    assert c == 1, f"[patch] expected exactly 1 match, found {c}: {old[:70]}"
    s = s.replace(old, new)
    done += 1
PATCH.write_text(s, encoding="utf-8")
print(f"crossmodel_patch.py: {done} edit(s) applied" if done else "crossmodel_patch.py: already up to date")

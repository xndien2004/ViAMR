import re
from collections import defaultdict
from underthesea import word_tokenize
from typing import Iterable, Optional
import penman

def remove_single_prop_nodes(amr_str: str) -> str:
    """
    Xóa các thuộc tính dạng :role (var / concept) nếu concept không có thuộc tính con nào khác.
    Ví dụ: ':tense (là)' hoặc ':tense (t / tense)' -> xóa cả cụm.
    """
    pattern = re.compile(r'\s*:[a-zA-Z0-9_-]+\s*\([^():]+(?:\s*/\s*[^():]+)?\)')

    prev = None
    new_str = amr_str
    while prev != new_str:
        prev = new_str
        new_str = pattern.sub('', new_str)

    new_str = re.sub(r'\s+', ' ', new_str).strip()
    return new_str

def has_duplicate_nodes(amr_str: str) -> bool:
    """
    Kiểm tra xem AMR string có biến node nào bị trùng tên hay không.
    Node: ký tự (hoặc chuỗi ký tự) đứng ngay sau '(' và trước '/'.
    """
    pattern = re.compile(r'\(\s*([^\s/()]+)\s*/')
    nodes = pattern.findall(amr_str)

    seen = set()
    for var in nodes:
        if var in seen:
            print(f"Duplicate node found: {var}")
            return True
        seen.add(var)
    return False

def vi_tokenize(text):
   return word_tokenize(text, format="text")

def dedup_and_tidy(amr: str,
                   roles: Optional[Iterable[str]] = (":agent",),
                   trim_tail: bool = True) -> str:
    """
    - Xoá các lần lặp *sau* của mẫu ':role <var>' trên toàn chuỗi (giữ lần đầu).
      Mặc định chỉ áp dụng cho ':agent'; truyền roles=None để áp dụng mọi role.
    - Không đụng dạng ':role (<var> / concept)'.
    - Dọn khoảng trắng và (tuỳ chọn) cắt các dòng cuối chỉ toàn dấu ')'.
    """
    # 1) Dedup ':role var' toàn cục, giữ lần đầu
    role_pat = "|".join(re.escape(r[1:] if r.startswith(":") else r) for r in roles) if roles else r"[-A-Za-z0-9_.]+"
    rx = re.compile(rf'(\s:(?P<role>{role_pat})\s+(?P<var>[A-Za-z][A-Za-z0-9_-]*))(?!\s*/)')
    seen = set()
    def _sub(m: re.Match) -> str:
        key = (m.group("role"), m.group("var"))
        if key in seen:
            return ""              # xoá lần lặp
        seen.add(key)
        return m.group(1)          # giữ nguyên lần đầu
    s = rx.sub(_sub, amr)

    s = re.sub(r'[ \t]+\)', ')', s)        # gom '   )' -> ')'
    lines = [ln for ln in s.splitlines() if ln.strip() != ""]
    if trim_tail:
        while lines and re.fullmatch(r'\s*\)+\s*', lines[-1]):
            lines.pop()
    return "\n".join(lines)

def fix_amr_vars(amr_str: str) -> str:
    """
    Xóa trường hợp node có hai lần '/' liên tiếp như:
    (n3 / núi / hehe ...) => (n3 / núi ...)
    """
    pattern = re.compile(r'(\(\s*\w+\s*/\s*[^()\s]+)\s*/\s*[^()\s]+', re.UNICODE)

    fixed = amr_str
    while True:
        new_fixed = re.sub(pattern, r'\1', fixed)
        if new_fixed == fixed:
            break
        fixed = new_fixed

    return fixed

def balance_parens(amr: str) -> str:
    """
    Thêm đủ dấu ')' vào cuối nếu số ngoặc mở '(' nhiều hơn ngoặc đóng ')'.
    """
    opens = amr.count("(")
    closes = amr.count(")")
    diff = opens - closes
    if diff > 0:
        amr = amr.rstrip() + ")" * diff
    return amr

# def balance_parens(amr: str, trim_trailing_excess: bool = True, ensure_newline: bool = True) -> str:
#     """
#     Cân bằng dấu ngoặc '()' cho chuỗi AMR/PENMAN.
#     - Không đếm ngoặc nằm trong chuỗi trích dẫn (giữa dấu " ").
#     - Nếu thiếu ')', thêm đủ số ')' vào cuối chuỗi.
#     - Nếu thừa ') ' ở *cuối chuỗi* và trim_trailing_excess=True, cắt bớt cho cân bằng.
#     - Giữ nguyên nội dung khác.

#     Params:
#       trim_trailing_excess: cho phép cắt bớt ')' thừa ở đuôi
#       ensure_newline: thêm newline trước khi nối ')' nếu cần

#     Returns:
#       Chuỗi AMR đã cân bằng ngoặc.
#     """
#     opens = closes = 0
#     in_quote = False

#     for i, ch in enumerate(amr):
#         if ch == '"':
#             # đảo trạng thái nếu không phải dấu " được escape
#             if not (i > 0 and amr[i-1] == '\\'):
#                 in_quote = not in_quote
#         elif not in_quote:
#             if ch == '(':
#                 opens += 1
#             elif ch == ')':
#                 closes += 1

#     diff = opens - closes  # >0 thiếu ')', <0 thừa ')'
#     s = amr

#     if diff > 0:
#         s = s.rstrip()
#         if ensure_newline and not s.endswith('\n'):
#             s += '\n'
#         s += ')' * diff
#     elif diff < 0 and trim_trailing_excess:
#         need = -diff
#         # tạm bỏ whitespace cuối để xử lý cụm ')' đuôi
#         trail_ws_len = len(s) - len(s.rstrip())
#         core = s[:len(s) - trail_ws_len]
#         m = re.search(r'\)+$', core)
#         if m:
#             run = m.end() - m.start()
#             cut = min(run, need)
#             core = core[:m.end() - cut]
#             s = core + s[len(core):]  # gắn lại whitespace cũ

#     return s

def find_vars_with_concept(amr):
    return set(re.findall(r'\(\s*(\w+)\s*/\s*[^()\s]+', amr))

def find_single_var_references(amr):
    pattern = re.compile(r'(:[\w\-]+)\s+(\w+)(?!\s*/)')
    return pattern.findall(amr)

def remove_empty_roles(amr_str):
    pattern = re.compile(r':[\w\-]+(?=\s*[\)\n])')
    amr_str = pattern.sub('', amr_str)
    # Dọn dẹp khoảng trắng và tab dư thừa (giữ nguyên newline)
    amr_str = re.sub(r'[ \t]{2,}', ' ', amr_str)
    amr_str = re.sub(r'\(\s*\)', '', amr_str)
    return amr_str.strip()

def process_amr_general(amr):
    vars_with_concept = find_vars_with_concept(amr)
    refs = find_single_var_references(amr)
    vars_in_refs = set([var for _, var in refs])

    vars_declared_and_referenced = vars_with_concept.intersection(vars_in_refs)
    vars_referenced_but_not_declared = vars_in_refs - vars_with_concept

    for v in vars_declared_and_referenced:
        amr = re.sub(r'\(\s*' + re.escape(v) + r'\s*/[^()]*\)', '', amr)
        amr = re.sub(r'(:[\w\-]+)\s+' + re.escape(v), '', amr)

    for v in vars_referenced_but_not_declared:
        amr = re.sub(r'(:[\w\-]+)\s+' + re.escape(v), '', amr)

    # Xóa các quan hệ trống (role không có giá trị node/var)
    amr = remove_empty_roles(amr)

    return amr.strip()

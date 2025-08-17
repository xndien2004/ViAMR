import re

# 1) Chỉ nối concept đa từ sau dấu '/' thành dạng có gạch dưới.
def join_concepts_underscores(text: str) -> str:
    rx = re.compile(r'/\s+([^\n\r():/]+?)(?=\s*[:/)]|\s*$)', re.UNICODE)
    def _repl(m):
        return '/ ' + re.sub(r'\s+', '_', m.group(1).strip())
    return rx.sub(_repl, text)

# 2) Gộp lỗi "hai lần / concept" trong ĐẦU node: (v / A / B ...) -> (v / A ...)
def fix_amr_vars(amr_str: str) -> str:
    pat = re.compile(r'(\(\s*\w+\s*/\s*[^()\s:]+)\s*/\s*[^()\s:]+', re.UNICODE)
    fixed = amr_str
    while True:
        new_fixed = pat.sub(r'\1', fixed)
        if new_fixed == fixed:
            break
        fixed = new_fixed
    return fixed

# 3) Chuẩn hoá khoảng trắng quanh role để tránh dính 'word:role' hay ':role('
def normalize_roles_spacing(s: str) -> str:
    s = re.sub(r'(?<!:)([^\s():]+):([A-Za-z][\w-]*)', r'\1 :\2', s)  # word:role -> word :role
    s = re.sub(r'(:[\w-]+)\(', r'\1 (', s)                           # :role( -> :role (
    s = re.sub(r'(:[\w-]+)_([^\s():]+)', r'\1 \2', s)                # :role_x -> :role x
    return s

# 4) Xoá dấu '/' mồ côi (trước ')', trước ':', trước newline/EOF)
def strip_orphan_slashes(s: str) -> str:
    return re.sub(r'/\s*(?=\)|:[\w-]|\n|$)', '', s)

# 5) Cân ngoặc (không cắt nội dung), thêm ')' nếu thiếu; bỏ ')' lạc lõng
def balance_parens(s: str) -> str:
    out, depth = [], 0
    for ch in s:
        if ch == '(':
            depth += 1; out.append(ch)
        elif ch == ')':
            if depth > 0:
                depth -= 1; out.append(ch)
            # nếu depth==0: bỏ ')' lạc
        else:
            out.append(ch)
    if depth > 0:
        out.append(')' * depth)
    return ''.join(out)

# 6) (TUỲ CHỌN) Bỏ lặp ':role var' RÕ RÀNG cho một vài role (mặc định: TẮT)
def dedup_selected_roles(amr: str, roles=()):
    if not roles:  # tắt mặc định để khỏi mất thông tin
        return amr
    role_pat = "|".join(re.escape(r[1:] if r.startswith(":") else r) for r in roles)
    rx = re.compile(rf'(\s:(?P<role>{role_pat})\s+(?P<var>[A-Za-z][A-Za-z0-9_-]*))(?!\s*/)')
    seen = set()
    def _sub(m):
        key = (m.group("role"), m.group("var"))
        if key in seen: return ""
        seen.add(key);  return m.group(1)
    return rx.sub(_sub, amr)

# 7) HÀM CHÍNH: sanitize tối thiểu, KHÔNG xoá node/role hợp lệ
def penman_safe_minimal(amr: str, roles_to_dedup=()):
    s = amr
    s = normalize_roles_spacing(s)
    s = join_concepts_underscores(s)
    s = fix_amr_vars(s)
    s = strip_orphan_slashes(s)
    s = balance_parens(s)
    # s = dedup_selected_roles(s, roles=roles_to_dedup)  # mặc định không đụng
    s = re.sub(r'[ \t]+', ' ', s).strip()
    return s

def has_duplicate_nodes(amr_str: str) -> bool:
    """ Kiểm tra xem AMR string có biến node nào bị trùng tên hay không.
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

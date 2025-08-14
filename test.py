# amr_to_actions.py
from penman import decode, load
from penman import Graph
from typing import List, Tuple

def amr_to_actions(amr_str: str) -> List[str]:
    """
    Convert an AMR in Penman format to an action sequence:
    SHIFT(concept_id / concept), ARC(role, parent_id, child_id), REDUCE(concept_id)
    """
    graph = decode(amr_str)
    # Get variable names and concepts
    var2concept = {instance.source: instance.target for instance in graph.instances()}
    # Get edges: each relation triple (parent, role, child)
    edges = [(trip.source, trip.role, trip.target) for trip in graph.edges()]
    
    # Heuristic: process nodes in the order they appear in instances()
    actions = []
    # collect nodes ordering
    order = list(var2concept.keys())
    for v in order:
        actions.append(f"SHIFT({v} / {var2concept[v]})")
        # Immediately emit arcs from this node to its children
        for parent, role, child in edges:
            if parent == v:
                actions.append(f"ARC({role}, {parent}, {child})")
    # Finally issue reduces in reverse order
    for v in reversed(order):
        actions.append(f"REDUCE({v})")
    return actions

# Example usage:
example_amr = """
(l1 / làm
    :condition(y / yêu
        :pivot(n / người
            :quant 1)
        :theme(h / hoa
            :classifier(đ / đoá
                :quant 1)
            :mod(d / duy_nhất)
            :mod(s / sao
                :prep(t / trong)
                :classifier(n1 / ngôi)
                :quant(m / multiple
                    :quant 1000000))))
    :cause(n2 / nhìn
        :degree(c / chỉ)
        :agent(s1 / sao
            :classifier(n4 / ngôi)))
    :degree(đ1 / đủ)
    :patient(a / anh
        :mod(t4 / ta))
    :result-arg2(h2 / hạnh_phúc))
"""
acts = amr_to_actions(example_amr)
print("\n".join(acts))

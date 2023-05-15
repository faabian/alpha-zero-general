from typing import List, Dict, Optional, Tuple
import re


def parse_visits(s: str) -> List[Optional[int]]:
    res: List[Optional[int]] = []
    for v in s.splitlines():
        try:
            res.append(int(v[-3:]))
        except ValueError:
            res.append(None)
    return res


def parse_updates(s: str) -> List[Optional[Tuple[int, int, int, int, float]]]:
    updates = s.splitlines()
    res: List[Optional[Tuple[int, int, int, int, float]]] = []
    for u in updates:
        try:
            match = re.match(
                r"update (?P<node>\d\d\d) (?P<action>\d\d) (?P<Ns>\d\d\d) (?P<Nsa>\d\d\d) (?P<Qsa>[+-][01].\d\d\d)",
                u,
            )
            assert match is not None
            d = match.groupdict()
            res.append(
                (
                    int(d["node"]),
                    int(d["action"]),
                    int(d["Ns"]),
                    int(d["Nsa"]),
                    float(d["Qsa"]),
                )
            )
        except (ValueError, AssertionError):
            res.append(None)

    return res


def parse_expand(groupdict: Dict[str, str]) -> Optional[Tuple[int, int, str, float]]:
    try:
        return (
            int(groupdict["node"]),
            int(groupdict["action"]),
            groupdict["board"],
            float(groupdict["v"]),
        )
    except (KeyError, TypeError):
        return None


def parse_mcts_actions(
    s: str,
) -> Tuple[
    List[Optional[int]],
    Optional[Tuple[int, int, str, float]],
    List[Optional[Tuple[int, int, int, int, float]]],
]:
    match = re.match(
        r"^(?P<visits>(visit (\d\d\d)\n)*)(?P<expand>expand (?P<node>\d\d\d) (?P<action>\d\d) (?P<board>([-XO] ){36})(?P<v>[+-][01].\d\d\d)\n)?"
        r"(?P<updates>(update \d\d\d \d\d \d\d\d \d\d\d [+-][01].\d\d\d\n)*)$",
        s,
    )
    assert match is not None, s
    d = match.groupdict()
    visit_ids = parse_visits(d["visits"])
    expand = parse_expand(d)
    updates = parse_updates(d["updates"])
    return visit_ids, expand, updates

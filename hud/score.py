"""Match score from rallies: rally-point scoring to 21, win by 2, capped at 30, best of three.

Who won each rally, first that applies:
  1. point_fixes CSV (rally_id,point_team with us/opp, or 'none' for a let): the human word
  2. rallies.point_team: the shuttle was seen landing and in/out was judged (an estimate:
     line calls carry the calibration error)
  3. otherwise nobody: the score doesn't move and the rally is flagged for review
The rally winner serves next. Start-of-video score comes from config [score], for videos
that begin mid-match.
ponytail: sides are near/far halves, so after the end change between games 'us' would be the
other team. Analyse one game per video until the end change is detected.
"""
import pandas as pd

TEAMS = ("us", "opp")


def game_over(us, opp, s):
    hi, lo = max(us, opp), min(us, opp)
    return hi >= s["cap"] or (hi >= s["target"] and hi - lo >= 2)


def apply(rallies, cfg, fixes=None):
    """Adds point_team_final, point_source, score_us/opp, games_us/opp, game_no, server
    (who serves the next rally), game_won to a copy of rallies."""
    s = cfg["score"]
    over = {}
    if fixes is not None and len(fixes):
        bad = set(fixes.point_team) - set(TEAMS) - {"none"}
        if bad:
            raise SystemExit(f"point fixes: point_team must be us, opp or none, got {sorted(bad)}")
        over = dict(zip(fixes.rally_id.astype(int), fixes.point_team))
    us, opp, g_us, g_opp = s["us"], s["opp"], s["games_us"], s["games_opp"]
    server = s.get("first_server") or None
    rows = []
    for r in rallies.sort_values("rally_id").itertuples():
        if r.rally_id in over:
            team, src = (None if over[r.rally_id] == "none" else over[r.rally_id]), "fixed"
        elif isinstance(r.point_team, str) and r.point_team in TEAMS:
            team, src = r.point_team, "landing"
        elif isinstance(getattr(r, "last_hitter", None), str):
            # landing not seen: guess the side that hit last won it (fix it in the editor if not)
            team, src = ("us" if r.last_hitter in ("P1", "P2") else "opp"), "guess"
        else:
            team, src = None, "unknown"
        won = None
        if team:
            us, opp = us + (team == "us"), opp + (team == "opp")
            server = team
            if game_over(us, opp, s):
                won = "us" if us > opp else "opp"
        rows.append(dict(point_team_final=team, point_source=src, score_us=us, score_opp=opp,
                         games_us=g_us + (won == "us"), games_opp=g_opp + (won == "opp"),
                         game_no=g_us + g_opp + 1, server=server, game_won=won))
        if won:  # next game from 0-0; the winner serves first
            g_us, g_opp, us, opp = g_us + (won == "us"), g_opp + (won == "opp"), 0, 0
    out = rallies.sort_values("rally_id").reset_index(drop=True)
    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def match_over(row, cfg):
    need = cfg["score"]["games_to_win"]
    return row.games_us >= need or row.games_opp >= need

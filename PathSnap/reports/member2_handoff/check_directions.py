"""Run from PathSnap: .venv/bin/python reports/member2_handoff/check_directions.py"""
from pathlib import Path
import numpy as np
import pandas as pd

rows = []
for split in ("train", "student_test"):
    features = pd.read_csv(f"data/member1/{split}_features.csv", usecols=["trip_id","point_id","arc_id"], dtype={"trip_id":str})
    for trip, group in features.groupby("trip_id"):
        base = Path("../../Student folder") / split / trip / trip
        arcs = np.loadtxt(base.with_suffix(".arcs"),dtype=int,ndmin=2)
        route = np.loadtxt(base.with_suffix(".route"),dtype=int,ndmin=1)
        traversed = set(map(tuple,arcs[route]))
        u,v = arcs[group.arc_id].T
        # Collapse parallel arcs with the same ordered endpoints. Self loops have no opposite direction.
        seen = pd.DataFrame(dict(point=group.point_id,lo=np.minimum(u,v),hi=np.maximum(u,v),forward=u<v))
        directions = seen[u!=v].groupby(["point","lo","hi"]).forward.nunique()
        rows.append(dict(split=split,trip_id=trip,represented_fixes=group.point_id.nunique(),
                         fixes_with_both_directions=directions[directions==2].index.get_level_values("point").nunique(),
                         route_uses_both_directions=any((b,a) in traversed for a,b in traversed if a!=b)))
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("directions_by_trip.csv"),index=False)
print(out.groupby("split").agg(trips=("trip_id","size"), fixes=("represented_fixes","sum"),
    fixes_with_both=("fixes_with_both_directions","sum"),trips_traversing_both=("route_uses_both_directions","sum")))

import gpxpy
import numpy as np
from datetime import timezone


def parse_gpx(gpx_file):
    points = []
    with open(gpx_file, "r") as f:
        gpx = gpxpy.parse(f)
        for track in gpx.tracks:
            for seg in track.segments:
                for point in seg.points:
                    t = point.time
                    if t is None:
                        continue
                    t = t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)
                    points.append({"time": t, "lat": point.latitude, "lon": point.longitude})
    points.sort(key=lambda p: p["time"])
    return points


def nearest_gps(gps_points, target_time):
    if not gps_points:
        return None
    deltas = np.array([abs((p["time"] - target_time).total_seconds()) for p in gps_points])
    return gps_points[int(deltas.argmin())]
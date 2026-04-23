from typing import Optional


def seconds_remaining_in_market(end_ts: int, now_ts: Optional[int] = None) -> int:
    if end_ts <= 0:
        return 0
    current_ts = now_ts if now_ts is not None else __import__("time").time()
    return max(int(end_ts - int(current_ts)), 0)


def is_last_minute_of_market(end_ts: int, now_ts: Optional[int] = None) -> bool:
    return seconds_remaining_in_market(end_ts, now_ts=now_ts) <= 60

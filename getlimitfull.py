import asyncio
import websockets
import json
import sys
import requests
import time
import os

def get_live_market_details():
    now_ts = int(time.time())
    current_window = (now_ts // 300) * 300
    slug = f"btc-updown-5m-{current_window}"
    
    gamma_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        resp = requests.get(gamma_url, timeout=2)
        data = resp.json()
        if not data:
            return None, None
        
        market = data[0]['markets'][0]
        raw_ids = market.get('clobTokenIds')
        
        if isinstance(raw_ids, str):
            token_ids = json.loads(raw_ids)
        else:
            token_ids = raw_ids
            
        # Mapping: Index 0 is UP, Index 1 is DOWN
        mapping = {
            str(token_ids[0]): "UP",
            str(token_ids[1]): "DOWN"
        }
        return slug, mapping
    except Exception:
        return None, None

def format_polymarket_dict(obj, id_map):
    if isinstance(obj, list):
        return [format_polymarket_dict(item, id_map) for item in obj]
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Inject the label (UP/DOWN) when we see an asset_id
            if k == 'asset_id' and str(v) in id_map:
                new_obj['outcome_label'] = id_map[str(v)]
                new_obj[k] = v
            # Standard price formatting
            elif k in ['price', 'best_bid', 'best_ask', 'size'] and v is not None:
                try:
                    num_val = float(v)
                    formatted = "{:.2f}".format(num_val).replace("0.", ".") if num_val < 1 else "{:.2f}".format(num_val)
                    new_obj[k] = formatted
                except (ValueError, TypeError):
                    new_obj[k] = v
            else:
                new_obj[k] = format_polymarket_dict(v, id_map)
        return new_obj
    return obj

async def monitor_clob_v2(slug, id_map):
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    records = []
    max_records = 5
    token_ids = list(id_map.keys())
    
    try:
        async with websockets.connect(uri, ping_interval=None) as ws:
            sub_msg = {
                "type": "market",
                "assets_ids": token_ids # Subscribe to both UP and DOWN tokens
            }
            await ws.send(json.dumps(sub_msg))

            while len(records) < max_records:
                response = await ws.recv()
                if not response or response in ["PONG", "[]"]:
                    continue
                
                try:
                    raw_data = json.loads(response)
                    msg_body = raw_data[0] if isinstance(raw_data, list) and raw_data else raw_data
                    
                    if msg_body.get("event_type") == "price_change":
                        msg_body["slug"] = slug
                        # Use the map to label the data correctly
                        records.append(format_polymarket_dict(msg_body, id_map))
                except Exception:
                    continue
            
            print(json.dumps(records, indent=4))
            sys.stdout.flush()
            os._exit(0)

    except Exception:
        os._exit(1)

if __name__ == "__main__":
    slug, id_map = get_live_market_details()
    if id_map:
        asyncio.run(monitor_clob_v2(slug, id_map))
    else:
        sys.exit(1)

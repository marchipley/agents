import asyncio
import websockets
import json
import requests
import time
import sys

def get_live_market_details():
    """Calculates the current active 5m window and fetches the UP token ID."""
    now_ts = int(time.time())
    current_window = (now_ts // 300) * 300
    slug = f"btc-updown-5m-{current_window}"
    
    gamma_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        resp = requests.get(gamma_url)
        data = resp.json()
        if not data:
            return None, None
        
        market = data[0]['markets'][0]
        raw_ids = market.get('clobTokenIds')
        
        if isinstance(raw_ids, str):
            token_ids = json.loads(raw_ids)
        else:
            token_ids = raw_ids
            
        return slug, str(token_ids[0])
    except Exception:
        return None, None

async def get_single_latest_tick(slug, token_id):
    """Fetches exactly one valid price update then exits."""
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    
    try:
        async with websockets.connect(uri, ping_interval=None) as ws:
            sub_msg = {
                "type": "market",
                "assets_ids": [token_id]
            }
            await ws.send(json.dumps(sub_msg))

            while True:
                response = await ws.recv()
                
                if response in ["PONG", "[]", ""] or not response:
                    continue
                
                try:
                    data = json.loads(response)
                    msg_body = data[0] if isinstance(data, list) and data else data
                    
                    if msg_body.get("event_type") == "price_change":
                        buy_ask = "N/A"
                        sell_ask = "N/A"
                        
                        updates = msg_body.get("price_changes", [])
                        for item in updates:
                            side = item.get("side")
                            # Get value as float and format to remove leading zero (0.45 -> .45)
                            raw_val = float(item.get("best_ask", 0))
                            formatted_val = "{:.2f}".format(raw_val).replace("0.", ".")
                            
                            if side == "BUY":
                                buy_ask = formatted_val
                            elif side == "SELL":
                                sell_ask = formatted_val

                        # Output formatted exactly as requested
                        print(f"{slug} | BUY ASK (UP): {buy_ask} | SELL ASK (DOWN): {sell_ask}")
                        break 

                except (json.JSONDecodeError, TypeError):
                    continue

    except Exception:
        pass

if __name__ == "__main__":
    slug, token_id = get_live_market_details()
    
    if token_id:
        try:
            asyncio.run(get_single_latest_tick(slug, token_id))
        except KeyboardInterrupt:
            pass
    else:
        sys.exit(1)

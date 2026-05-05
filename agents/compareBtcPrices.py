import asyncio
import json
import requests
import websockets

# Using the EXACT ID from your working curl
PYTH_BTC_ID = "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

async def get_binance_price():
    try:
        url = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
        async with websockets.connect(url) as websocket:
            data = json.loads(await websocket.recv())
            return float(data['c'])
    except: return "Binance Down"

async def get_coinbase_price():
    try:
        url = "wss://ws-feed.exchange.coinbase.com"
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}))
            while True:
                data = json.loads(await websocket.recv())
                if data.get('type') == 'ticker':
                    return float(data['price'])
    except: return "Coinbase Down"

def get_pyth_price():
    url = f"https://hermes.pyth.network/v2/updates/price/latest?ids[]={PYTH_BTC_ID}&parsed=true"
    try:
        r = requests.get(url, headers=HEADERS, timeout=5)
        data = r.json()
        # Accessing the 'parsed' array directly
        p_obj = data['parsed'][0]['price']
        price = int(p_obj['price'])
        expo = int(p_obj['expo'])
        return price * (10 ** expo)
    except Exception as e:
        return f"Pyth Error: {str(e)[:20]}"

def get_kraken_price():
    """Kraken is a high-authority USD source often used by Oracles."""
    try:
        url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
        r = requests.get(url, headers=HEADERS, timeout=5)
        data = r.json()
        # Kraken uses 'XXBTZUSD' as the key for BTC/USD
        return float(data['result']['XXBTZUSD']['c'][0])
    except: return "Kraken Error"

async def compare_all():
    print(f"\n{'SOURCE':<18} | {'PRICE':<15} | {'DIFF FROM BINANCE'}")
    print("-" * 58)
    
    while True:
        binance = await get_binance_price()
        coinbase = await get_coinbase_price()
        pyth = get_pyth_price()
        kraken = get_kraken_price()

        def get_diff(p):
            if isinstance(p, (int, float)) and isinstance(binance, (int, float)):
                diff = p - binance
                return f"{'+' if diff >= 0 else ''}${diff:>6.2f}"
            return "N/A"

        def fmt(p): return f"${p:,.2f}" if isinstance(p, (int, float)) else p

        print(f"{'Binance (USDT)':<18} | {fmt(binance):<15} | -- Reference --")
        print(f"{'Coinbase (USD)':<18} | {fmt(coinbase):<15} | {get_diff(coinbase)}")
        print(f"{'Kraken (USD)':<18} | {fmt(kraken):<15} | {get_diff(kraken)}")
        print(f"{'Pyth (Oracle)':<18} | {fmt(pyth):<15} | {get_diff(pyth)}")
        print("-" * 58)
        
        await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(compare_all())
    except KeyboardInterrupt:
        print("\nStopped by user.")
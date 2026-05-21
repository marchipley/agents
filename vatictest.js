const asset = 'btc';
const type = '5min';
const ts = 1774990800;

(async () => {
  const url = `https://api.vatic.trading/api/v1/targets/timestamp?asset=${asset}&type=${type}&timestamp=${ts}`;
  const response = await fetch(url);
  const data = await response.json();

  console.log('Vatic API Target:', data.price);
})();

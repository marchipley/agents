Add Features:

1. For each order that is placed, output a file in completed orders with the slug timestamp that shows the status of the active order until completed with data for each tick so we can analyze why each position may have won or lost. 

2. Use a threshold if specified in .env for the buy order for the max price taking into account the amount of time left in the period and how far above/below the strike price the current price is. Probably avoid this within the last minute.

3. Move non critical vars from .env to a config file. We only need sensitive info in .env.

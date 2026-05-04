Add Features:
2. There blocks of times when we see many more wins IE: when we ran testing from 5/6/2026 @ 15:30 PST to 16:20 PST there were only a couple losses. Investigate other factors like news events if not already doing so.

3. Add the order type to the completed_ordres log file (up/down) IE: completed_order_win_up_1777851300.txt to indicate whether it is an up or down orderq. 

3. Move non critical vars from .env to a config file. We only need sensitive info in .env.

4. Add variable to .env for the amount to increase the max_price_to_pay when the confidence is >=.7. Currently it is hard coded at .02

5. Move most of the vars from .env to config.py. Only secrets need to be in .env



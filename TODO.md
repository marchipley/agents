Add Features:

1. Investigate: Execution edge -0.122 <= 0.050 (confidence=0.820; implied_probability=0.760)
Sleeping 5 seconds before next tick..

3. Move non critical vars from .env to a config file. We only need sensitive info in .env.

4. Add variable to .env for the amount to increase the max_price_to_pay when the confidence is >=.7. Currently it is hard coded at .02

5. Move most of the vars from .env to config.py. Only secrets need to be in .env



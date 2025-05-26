"""OpenWeather package."""
__version__ = "0.1.0"

import logging
from logging import NullHandler

# Configure null handler to prevent "No handler found" warnings
logging.getLogger("openweather").addHandler(NullHandler())
logging.getLogger("openweather").info("OpenWeather package loaded.") 
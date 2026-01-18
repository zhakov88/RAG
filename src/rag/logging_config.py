import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": "logs/app.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "level": "DEBUG",
        },
    },

    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",
    },
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)

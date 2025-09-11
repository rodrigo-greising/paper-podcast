import logging


def get_logger(name: str) -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler()
		formatter = logging.Formatter(
			fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
			datefmt="%H:%M:%S",
		)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		logger.setLevel(logging.INFO)
	return logger

"""Quick visual check of extty's colored structured logging.

Shows the recommended way to use the logger from a user's training script.
"""

from extty import logger

# logger.setLevel(logging.DEBUG)

logger.debug("debug — dim cyan level")
logger.info("info — neon cyan level (matches TUI selected run)")
logger.warning("warning — neon yellow (matches TUI running-time)")
logger.error("error — neon magenta (matches TUI selected name)")
logger.critical("critical — bold neon magenta")

print()

logger.info("loading dataset from %s", "/data/train.parquet")
logger.info("starting training: %d epochs, batch_size=%d", 10, 32)
logger.warning("learning rate %.0e looks high — consider warmup", 5e-3)
logger.error("checkpoint upload failed, will retry: %s", "connection reset")

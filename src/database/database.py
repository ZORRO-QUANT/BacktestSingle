from typing import Optional, Dict, Any

import aiomysql
import logging

logger = logging.getLogger(__name__)


class Database:
    """Asynchronous database connection manager with connection pooling."""

    _pool: Optional[aiomysql.Pool] = None

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database manager with configuration.

        Args:
            config: Dictionary containing database configuration
                   Expected format:
                   {
                       "host": str,
                       "user": str,
                       "password": str,
                       "database": str,
                       "port": int,
                   }
        """
        self._config = config
        self.conn: Optional[aiomysql.Connection] = None
        self.cursor: Optional[aiomysql.Cursor] = None

        # Default pool configuration
        self._pool_config = {
            "minsize": 1,
            "maxsize": 10,
            "pool_recycle": 3600,
            "autocommit": True,
            "echo": False,
        }

        # Update pool config if provided in config
        if "pool" in config:
            self._pool_config.update(config["pool"])

    @classmethod
    async def get_pool(
        cls, config: Dict[str, Any], pool_config: Dict[str, Any]
    ) -> aiomysql.Pool:
        """
        Get or create the connection pool.

        Args:
            config: Database configuration dictionary
            pool_config: Pool configuration dictionary

        Returns:
            aiomysql.Pool: The connection pool
        """
        if cls._pool is None:
            cls._pool = await aiomysql.create_pool(
                host=config["host"],
                user=config["user"],
                password=config["password"],
                db=config["database"],
                port=config["port"],
                **pool_config,
            )
            logger.debug("Database connection pool created")
        return cls._pool

    async def initialize(self) -> None:
        """
        Initialize database connection from pool.

        Raises:
            aiomysql.Error: If connection fails
        """
        pool = await self.get_pool(self._config, self._pool_config)
        self.conn = await pool.acquire()
        self.cursor = await self.conn.cursor()
        logger.debug("Database connection and cursor initialized from pool")

    async def close(self) -> None:
        """Safely close cursor and release connection back to pool."""
        if self.cursor:
            try:
                await self.cursor.close()
                logger.debug("Database cursor closed")
            except Exception as e:
                logger.error("Error closing cursor: %s", e)
            finally:
                self.cursor = None

        if self.conn:
            try:
                pool = await self.get_pool(self._config, self._pool_config)
                pool.release(self.conn)
                logger.debug("Database connection released back to pool")
            except Exception as e:
                logger.error("Error releasing connection: %s", e)
            finally:
                self.conn = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        await self.close_pool()

    @classmethod
    async def close_pool(cls) -> None:
        """Close the entire connection pool."""
        if cls._pool:
            cls._pool.close()
            await cls._pool.wait_closed()
            cls._pool = None
            logger.debug("Database connection pool closed")

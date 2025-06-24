from typing import List, Optional, Tuple, Any, Dict, Union
from datetime import datetime

from database.database import Database, logger


class Reader(Database):
    """Database reader for retrieving market data."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reader with database configuration.

        Args:
            config: Database configuration dictionary
        """

        super().__init__(config)

    async def execute_query(
        self,
        query: str,
        params: Optional[Union[tuple, list]] = None,
        fetch: bool = True,
    ) -> Optional[List[Tuple[Any]]]:
        """
        Execute a SQL query efficiently.

        Args:
            query: SQL query string
            params: Query parameters (optional)
            fetch: Whether to fetch results after execution

        Returns:
            List of tuples containing the query results if fetch=True, None otherwise
        """
        try:
            start_time = datetime.now()

            # Execute query
            await self.cursor.execute(query, params or ())

            # Fetch results if requested
            results = await self.cursor.fetchall() if fetch else None

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(
                "Query executed in %.3f seconds%s",
                execution_time,
                f", retrieved {len(results)} rows" if fetch else "",
            )

            return results

        except Exception as e:
            logger.error(
                "Failed to execute query: %s\nParameters: %s\nError: %s",
                query,
                params,
                e,
            )
            raise

    async def get_kline_data(
        self,
        database: str,
        table_name: str,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        columns: Optional[List[str]] = None,
    ) -> List[Tuple[Any]]:
        """
        Retrieve kline data for multiple symbols within a date range.
        Uses composite index (symbol, time) for efficient querying.

        Args:
            table_name: Name of the table to query
            symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            start_time: Start of the time range
            end_time: End of the time range
            columns: List of column names to retrieve. If None, retrieves all columns

        Returns:
            List of tuples containing the query results
        """
        try:
            # Construct column string
            cols = ", ".join(columns) if columns else "*"

            # Create placeholders for symbols IN clause
            symbol_placeholders = ", ".join(["%s"] * len(symbols))

            # Construct query using symbol IN clause and time range
            query = f"""
                SELECT {cols}
                FROM {database}.{table_name}
                WHERE symbol IN ({symbol_placeholders})
                AND openTime BETWEEN %s AND %s
                ORDER BY symbol, openTime ASC
            """

            # Parameters: first all symbols, then start_time and end_time
            params = tuple(symbols) + (start_time, end_time)

            return await self.execute_query(query, params)

        except Exception as e:
            logger.error(
                "Failed to retrieve data from %s for symbols %s: %s",
                table_name,
                symbols,
                e,
            )
            raise

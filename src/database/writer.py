from typing import Dict, Any
from database.database import Database, logger
import pandas as pd


class Writer(Database):
    """Database writer for inserting data."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the writer with database configuration.

        Args:
            config: Database configuration dictionary
        """
        super().__init__(config)

    async def execute_insert(self, table: str, data: pd.DataFrame, kline: bool) -> None:
        """
        Insert data into the specified table, adding columns if they don't exist.

        Args:
            table_name: Name of the table to insert data into
            data: Dictionary of column names and values to insert
        """
        try:
            # drop redundant columns if it's kline data
            if kline:
                data.drop(columns=["closeTime", "ignore"], inplace=True)
            else:
                pass

            # Check and add missing columns
            columns = data.columns
            for column in columns:
                await self.ensure_column_exists(table, column)

            # Create the columns string
            columns_str = ", ".join([f"`{col}`" for col in columns])

            # Create placeholders for values
            placeholders = ", ".join(["%s"] * len(columns))

            # Create UPDATE part for all columns except symbol and openTime (which form the unique key)
            update_parts = [
                f"{col} = VALUES({col})"
                for col in columns
                if col not in ["symbol", "openTime"]
            ]
            update_str = ", ".join(update_parts)

            # Construct query
            insert_query = f"""
                INSERT INTO {self._config["database"]}.{table} ({columns_str}) 
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE
                {update_str}
            """

            # Convert DataFrame to list of tuples for faster insertion
            values = [tuple(x) for x in data.values]

            # Batch size for insertions
            batch_size = 50000

            # Disable autocommit
            self.conn.autocommit = False

            async with self.conn.cursor() as cursor:
                for i in range(0, len(values), batch_size):
                    batch = values[i : i + batch_size]
                    await cursor.executemany(insert_query, batch)
                await self.conn.commit()

            return True

        except Exception as e:
            logger.error("Failed to insert data into %s: %s", table, e)
            if hasattr(self, "conn"):
                await self.conn.rollback()

            return False

    async def ensure_column_exists(self, table_name: str, column_name: str) -> None:
        """
        Ensure that a column exists in the specified table, adding it if necessary.

        Args:
            table_name: Name of the table
            column_name: Name of the column to check/add
        """
        try:
            # Check if the column exists
            query = f"SHOW COLUMNS FROM {table_name} LIKE %s"
            await self.cursor.execute(query, (column_name,))
            result = await self.cursor.fetchone()

            # If the column does not exist, add it
            if not result:
                alter_query = (
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} DOUBLE"
                )
                await self.cursor.execute(alter_query)
                await self.conn.commit()
                logger.info("Added column %s to table %s", column_name, table_name)

        except Exception as e:
            logger.error(
                "Failed to ensure column %s exists in %s: %s",
                column_name,
                table_name,
                e,
            )
            raise

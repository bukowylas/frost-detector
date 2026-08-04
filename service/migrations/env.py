"""Alembic environment: migrate the service schema against DATABASE_URL.

Uses the SQLAlchemy models' metadata as the migration target, and the same
``database_url()`` the app uses, so migrations and the running service always
agree on where the database is.
"""

from alembic import context
from sqlalchemy import engine_from_config, pool

from service.db import Base, database_url

config = context.config
config.set_main_option("sqlalchemy.url", database_url())
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(), target_metadata=target_metadata,
        literal_binds=True, dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.", poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # render_as_batch lets the same migrations run on SQLite (batch mode) as
        # well as Postgres, so dev and production share one migration history.
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

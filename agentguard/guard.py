from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from agentguard.models import SessionRecord
from agentguard.session import Session
from agentguard.storage import Storage


class Guard:
    def __init__(
        self,
        agent_name: str = "default",
        db_path: str | None = None,
    ):
        self.agent_name = agent_name
        self._storage = Storage(db_path=db_path)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        record = SessionRecord(agent_name=self.agent_name)
        session = Session(record=record, storage=self._storage)

        try:
            yield session
            record.finalize("completed")
        except Exception:
            record.finalize("error")
            raise
        finally:
            session._save()

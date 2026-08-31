from __future__ import annotations

import json
from pathlib import Path

from autobrain.auth.models import Provider
from autobrain.auth.service import ConnectionManager
from autobrain.auth.storage import TokenStore
from tests.auth.test_storage_callback import FailedKeyring


def test_malformed_oauth_index_entry_is_fail_closed(tmp_path: Path) -> None:
    store = TokenStore(tmp_path, backend=FailedKeyring())
    store.index.parent.mkdir(parents=True, exist_ok=True)
    store.index.write_text(json.dumps({"slack:T1:U1": None}), encoding="utf-8")

    statuses = store.statuses()
    manager = ConnectionManager(tmp_path.parent, store=store)

    assert statuses == ()
    assert manager.token_for(Provider.SLACK) is None
    assert manager.logout(Provider.SLACK) == 0

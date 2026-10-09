"""
localdb.py — a tiny, zero-dependency, file-backed async store that mimics the
small slice of the (async) PyMongo API that DentalX Camp's backend uses.

Why: the app should run with NO MongoDB install. When MONGODB_URI is empty,
server.py uses this instead. Data persists to a single JSON file so the camp
dashboard/queue survive a restart. Not for production or concurrency at scale —
it is a demo/offline store, which is exactly DentalX's use case.

Implements, per collection:
  find_one(filter, projection=None)
  insert_one(doc)                       -> obj with .inserted_id
  find(filter, projection=None)         -> cursor with .sort().limit().to_list(n)
  find_one_and_update(filter, update, upsert=, return_document=)
  update_one(filter, update, upsert=)
  create_index(...)                     -> records unique indexes for dup checks
Query operators: equality, $or, $regex (+$options 'i').
Update operators: $set, $inc, $setOnInsert.
ObjectId is reused from bson (ships with pymongo).
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from bson import ObjectId

AFTER = "after"  # mirrors pymongo.ReturnDocument.AFTER (compared by identity/truthiness)


def _jsonify(v):
    if isinstance(v, ObjectId):
        return {"$oid": str(v)}
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonify(x) for x in v]
    # datetime -> iso; keep a marker so we can revive it
    if hasattr(v, "isoformat") and not isinstance(v, (str, bytes)):
        return {"$date": v.isoformat()}
    return v


def _revive(v):
    from datetime import datetime
    if isinstance(v, dict):
        if set(v.keys()) == {"$oid"}:
            return ObjectId(v["$oid"])
        if set(v.keys()) == {"$date"}:
            return datetime.fromisoformat(v["$date"])
        return {k: _revive(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_revive(x) for x in v]
    return v


def _match(doc, flt):
    """True if doc matches a (simple) Mongo filter."""
    for key, cond in flt.items():
        if key == "$or":
            if not any(_match(doc, sub) for sub in cond):
                return False
            continue
        val = doc.get(key)
        if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
            for op, operand in cond.items():
                if op == "$regex":
                    flags = re.I if cond.get("$options", "") and "i" in cond["$options"] else 0
                    if val is None or re.search(operand, str(val), flags) is None:
                        return False
                elif op == "$options":
                    continue
                elif op == "$in":
                    if val not in operand:
                        return False
                elif op == "$ne":
                    if val == operand:
                        return False
                else:
                    return False  # unsupported operator -> no match
        else:
            if val != cond:
                return False
    return True


def _project(doc, projection):
    if not projection:
        return dict(doc)
    include = {k: v for k, v in projection.items() if k != "_id"}
    if include and all(v == 1 for v in include.values()):
        out = {k: doc.get(k) for k in include if k in doc}
        if projection.get("_id", 1) != 0 and "_id" in doc:
            out["_id"] = doc["_id"]
        return out
    # exclusion projection {field: 0}
    out = {k: v for k, v in doc.items() if projection.get(k, 1) != 0}
    return out


def _apply_update(doc, update, *, inserting=False):
    if "$set" in update:
        doc.update(update["$set"])
    if "$inc" in update:
        for k, amt in update["$inc"].items():
            doc[k] = (doc.get(k, 0) or 0) + amt
    if "$setOnInsert" in update and inserting:
        for k, v in update["$setOnInsert"].items():
            doc.setdefault(k, v)
    # plain replacement (no operators) not used by server.py
    return doc


class _InsertResult:
    def __init__(self, _id):
        self.inserted_id = _id


class _Cursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, key, direction=-1):
        self._docs.sort(key=lambda d: (d.get(key) is None, d.get(key)), reverse=(direction == -1))
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    async def to_list(self, n=None):
        return self._docs if n is None else self._docs[:n]


class _Collection:
    def __init__(self, store, name):
        self._store = store
        self._name = name
        self._unique = []  # list of unique-index key tuples

    @property
    def _rows(self):
        return self._store._data.setdefault(self._name, [])

    async def create_index(self, keys, unique=False, **_):
        if unique:
            if isinstance(keys, str):
                self._unique.append((keys,))
            elif isinstance(keys, (list, tuple)):
                self._unique.append(tuple(k if isinstance(k, str) else k[0] for k in keys))
        return self._name

    async def find_one(self, flt, projection=None):
        for d in self._rows:
            if _match(d, flt):
                return _project(d, projection)
        return None

    def find(self, flt=None, projection=None):
        flt = flt or {}
        hits = [_project(d, projection) for d in self._rows if _match(d, flt)]
        return _Cursor(hits)

    async def insert_one(self, doc):
        if "_id" not in doc:
            doc["_id"] = ObjectId()
        for keys in self._unique:
            probe = {k: doc.get(k) for k in keys}
            if any(_match(r, probe) for r in self._rows):
                raise DuplicateKeyError(f"duplicate key on {keys}")
        self._rows.append(doc)
        self._store._flush()
        return _InsertResult(doc["_id"])

    async def find_one_and_update(self, flt, update, *, upsert=False, return_document=None, **_):
        for d in self._rows:
            if _match(d, flt):
                _apply_update(d, update)
                self._store._flush()
                return dict(d) if return_document else None
        if upsert:
            newdoc = {k: v for k, v in flt.items() if not (isinstance(v, dict) and any(str(x).startswith("$") for x in v))}
            if "_id" not in newdoc:
                newdoc["_id"] = ObjectId()
            _apply_update(newdoc, update, inserting=True)
            self._rows.append(newdoc)
            self._store._flush()
            return dict(newdoc) if return_document else None
        return None

    async def update_one(self, flt, update, *, upsert=False, **_):
        for d in self._rows:
            if _match(d, flt):
                _apply_update(d, update)
                self._store._flush()
                return
        if upsert:
            newdoc = {k: v for k, v in flt.items() if not (isinstance(v, dict) and any(str(x).startswith("$") for x in v))}
            if "_id" not in newdoc:
                newdoc["_id"] = ObjectId()
            _apply_update(newdoc, update, inserting=True)
            self._rows.append(newdoc)
            self._store._flush()


class DuplicateKeyError(Exception):
    pass


class LocalDB:
    """Mimics a pymongo database handle: db.<collection>.<op>()."""

    def __init__(self, path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._cols = {}
        self._data = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = {k: [_revive(d) for d in v] for k, v in raw.items()}
            except Exception:
                self._data = {}

    def _flush(self):
        with self._lock:
            tmp = self._path.with_suffix(".tmp")
            serial = {k: [_jsonify(d) for d in v] for k, v in self._data.items()}
            tmp.write_text(json.dumps(serial), encoding="utf-8")
            tmp.replace(self._path)

    def __getattr__(self, name):
        # db.doctors, db.patients, ...
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._cols:
            self._cols[name] = _Collection(self, name)
        return self._cols[name]

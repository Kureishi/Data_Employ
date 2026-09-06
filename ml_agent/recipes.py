"""
Reusable pipeline recipes.

A "recipe" captures everything needed to reproduce a training run: the data
source (table + optional relational synthesis), the preprocessing operations,
the target, task type, tuning level, parallelism, and an auto-prepare flag.
It is persisted per-database as a JSON file so a recipe can be reproduced on
new/updated data later with a single call.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional


class RecipeStore:
    """Load, save, list, and delete reproducible pipeline recipes."""

    def __init__(self, store_path: Optional[str] = None):
        self._store_path: Optional[str] = None
        self._recipes: Dict[str, Dict[str, Any]] = {}
        if store_path:
            self.set_store(store_path)

    def set_store(self, path: str) -> None:
        """Point the recipe store at a JSON file backing persistent recipes."""
        self._store_path = path
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._recipes = json.load(f)
            except Exception:
                self._recipes = {}

    def _persist(self) -> None:
        if not self._store_path:
            return
        try:
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(self._recipes, f, indent=2)
        except Exception:
            pass

    def save(self, recipe: Dict[str, Any]) -> str:
        """Save (or overwrite) a recipe under its 'name' key."""
        name = recipe.get("name") or time.strftime("recipe_%Y%m%d_%H%M%S")
        entry = dict(recipe)
        entry["name"] = name
        entry.setdefault("created", time.time())
        self._recipes[name] = entry
        self._persist()
        return name

    def list(self) -> List[Dict[str, Any]]:
        return list(self._recipes.values())

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._recipes.get(name)

    def delete(self, name: str) -> bool:
        if name in self._recipes:
            del self._recipes[name]
            self._persist()
            return True
        return False
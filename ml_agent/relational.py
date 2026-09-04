"""Relational deep-feature synthesis.

Walks the FK graph and aggregates child tables (count, sum, avg, min, max)
grouped by the foreign key, LEFT JOINed back onto the base table.
"""
from typing import List


class RelationalFeatureSynthesizer:
    def __init__(self, db) -> None:
        self.db = db

    def _numeric_columns(self, table):
        out = []
        for c in self.db.get_schema(table):
            t = str(c["type"]).lower()
            if any(k in t for k in ("int","float","double","decimal","numeric","real")):
                out.append(c["name"])
        return out

    def _find_children(self, base_table):
        out = []
        for t in self.db.get_tables():
            for fk in self.db.get_foreign_keys(t):
                if fk.get("referred_table") != base_table: continue
                refs = fk.get("referred_columns") or []; fks = fk.get("constrained_columns") or []
                if not refs or not fks: continue
                out.append({"table": t, "fk_col": fks[0], "base_key": refs[0]})
        return out

    def _agg_selects(self, child, cc, ca):
        sel, meta = [], []
        if cc:
            a = child + "__count"; sel.append("COUNT(*) AS \""+a+"\""); meta.append({"name":a,"child":child,"op":"count","column":None})
        if ca:
            for col in self._numeric_columns(child):
                for op in ("sum","avg","min","max"):
                    a = child + "__"+col+"__"+op; sel.append(op+"(\""+col+"\") AS \""+a+"\""); meta.append({"name":a,"child":child,"op":op,"column":col})
        return sel, meta

    def synthesize(self, base_table, max_rows=None, include_counts=True, include_aggregates=True):
        pk = self.db.get_primary_keys(base_table); base_key = pk[0] if pk else None
        children = self._find_children(base_table)
        if base_key is None or not children:
            return {"data": self.db.load_table(base_table, limit=max_rows), "features": [], "joins": []}
        features, joins, tsql, aliases = [], [], [], []
        select_refs = []
        for ch in children:
            sel, meta = self._agg_selects(ch["table"], include_counts, include_aggregates)
            if not sel: continue
            al = "agg_"+str(len(aliases)); aliases.append(al); features.extend(meta)
            for m in meta:
                select_refs.append(al + '."' + m["name"] + '"')
            joins.append({k: ch[k] for k in ("table","fk_col","base_key")})
            sub = "SELECT " + ch["fk_col"] + " AS __k, " + ", ".join(sel) + " FROM \"" + ch["table"] + "\" GROUP BY " + ch["fk_col"]
            tsql.append("LEFT JOIN (" + sub + ") " + al + " ON main.\"" + ch["base_key"] + "\" = " + al + ".__k")
        base_cols = [c["name"] for c in self.db.get_schema(base_table)]
        col_list = ", ".join("\""+c+"\"" for c in base_cols)
        if select_refs:
            col_list += ", " + ", ".join(select_refs)
        sql = "SELECT " + col_list + " FROM \"" + base_table + "\" main " + " ".join(tsql)
        if max_rows: sql += " LIMIT " + str(max_rows)
        return {"data": self.db.execute_query(sql), "features": features, "joins": joins}

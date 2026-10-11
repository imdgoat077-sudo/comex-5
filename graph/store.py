"""
AgentGraphStore implementation for Comex Agent Graph Memory using embedded Kuzu DB.
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
import kuzu

from .entity_resolution import canonical_key, extract_domain
from .schema import run_migrations

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PartialWriteRollbackError(RuntimeError):
    """Raised when a partial artifact write occurred and compensating rollback failed."""
    def __init__(self, artifact_id: str, message: str) -> None:
        super().__init__(message)
        self.artifact_id = artifact_id


class AgentGraphStore:
    def __init__(self, db_path: str):
        db_dir = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = db_path
        # Configure bounded buffer pool, mmap virtual memory space, and thread count
        # Default 8TB max_db_size triggers macOS ARM64 GPU carveout KERN_PROTECTION_FAILURE (SIGBUS 10).
        self.db = kuzu.Database(
            db_path,
            buffer_pool_size=256 * 1024 * 1024,
            max_db_size=4 * 1024 * 1024 * 1024,
            max_num_threads=1
        )
        self.conn = kuzu.Connection(self.db, num_threads=1)
        self._lock = threading.Lock()
        with self._lock:
            run_migrations(self.conn)

    def create_agent(self, agent_id: str, task: str, intent: str,
                     parent_agent_id: str | None = None) -> None:
        with self._lock:
            now = _utcnow()
            query = """
            CREATE (a:Agent {
                id: $id,
                task: $task,
                intent: $intent,
                status: 'running',
                created_at: $created_at,
                step_count: 0,
                parent_agent_id: $parent_agent_id
            })
            """
            params = {
                "id": agent_id,
                "task": task or "",
                "intent": intent or "UNKNOWN",
                "created_at": now,
                "parent_agent_id": parent_agent_id or ""
            }
            self.conn.execute(query, params)

            if parent_agent_id:
                rel_query = """
                MATCH (parent:Agent {id: $parent_id}), (child:Agent {id: $child_id})
                MERGE (child)-[:DEPENDS_ON]->(parent)
                """
                try:
                    self.conn.execute(rel_query, {"parent_id": parent_agent_id, "child_id": agent_id})
                except Exception:
                    pass

    def complete_agent(self, agent_id: str, status: str, step_count: int) -> None:
        with self._lock:
            now = _utcnow()
            query = """
            MATCH (a:Agent {id: $id})
            SET a.status = $status,
                a.completed_at = $completed_at,
                a.step_count = $step_count
            """
            params = {
                "id": agent_id,
                "status": status or "completed",
                "completed_at": now,
                "step_count": int(step_count)
            }
            self.conn.execute(query, params)

    def create_episode(self, agent_id: str, outcome_status: str,
                       self_reported: str) -> str:
        """Persist one Episode for a completed agent run.
        Returns the episode_id on success.
        constraint_check is always 'not_checkable' in this phase.
        """
        with self._lock:
            now = _utcnow()
            episode_id = f"ep_{uuid.uuid4().hex[:16]}"
            query = """
            CREATE (e:Episode {
                id: $id,
                agent_id: $agent_id,
                outcome_status: $outcome_status,
                self_reported: $self_reported,
                constraint_check: 'not_checkable',
                created_at: $created_at
            })
            """
            params = {
                "id": episode_id,
                "agent_id": agent_id,
                "outcome_status": outcome_status,
                "self_reported": self_reported or "",
                "created_at": now,
            }
            self.conn.execute(query, params)

            # Wire Episode -[FOR]-> Agent
            rel_query = """
            MATCH (e:Episode {id: $ep_id}), (a:Agent {id: $ag_id})
            CREATE (e)-[:FOR]->(a)
            """
            try:
                self.conn.execute(rel_query, {"ep_id": episode_id, "ag_id": agent_id})
            except Exception as re:
                logger.warning(f"Graph: failed to create Episode-FOR->Agent rel: {re}")

            return episode_id

    def record_tab(self, tab_id: str, is_hidden: bool, cdp_ws_url: str,
                   spawned_by_agent_id: str) -> None:
        with self._lock:
            now = _utcnow()
            query = """
            MERGE (t:Tab {id: $tab_id})
            ON CREATE SET t.is_hidden = $is_hidden,
                          t.cdp_ws_url = $cdp_ws_url,
                          t.created_at = $now
            ON MATCH SET t.is_hidden = $is_hidden,
                         t.cdp_ws_url = $cdp_ws_url
            """
            params = {
                "tab_id": tab_id,
                "is_hidden": bool(is_hidden),
                "cdp_ws_url": cdp_ws_url or "",
                "now": now
            }
            self.conn.execute(query, params)

            if spawned_by_agent_id:
                rel_query = """
                MATCH (a:Agent {id: $agent_id}), (t:Tab {id: $tab_id})
                MERGE (a)-[:SPAWNED]->(t)
                """
                self.conn.execute(rel_query, {"agent_id": spawned_by_agent_id, "tab_id": tab_id})

    def record_tool_call(self, agent_id: str, tool_name: str, args_json: str,
                         result_summary: str, step_index: int, success: bool) -> str:
        with self._lock:
            now = _utcnow()
            tool_call_id = str(uuid.uuid4())
            query = """
            CREATE (tc:ToolCall {
                id: $id,
                tool_name: $tool_name,
                args_json: $args_json,
                result_summary: $result_summary,
                step_index: $step_index,
                timestamp: $timestamp,
                success: $success
            })
            """
            params = {
                "id": tool_call_id,
                "tool_name": tool_name or "",
                "args_json": args_json or "",
                "result_summary": result_summary or "",
                "step_index": int(step_index),
                "timestamp": now,
                "success": bool(success)
            }
            self.conn.execute(query, params)

            if agent_id:
                rel_query = """
                MATCH (a:Agent {id: $agent_id}), (tc:ToolCall {id: $tool_call_id})
                MERGE (a)-[:EXECUTED]->(tc)
                """
                self.conn.execute(rel_query, {"agent_id": agent_id, "tool_call_id": tool_call_id})

            return tool_call_id

    def record_navigation(self, tool_call_id: str, url: str, title: str) -> str:
        with self._lock:
            now = _utcnow()
            domain = extract_domain(url)
            page_id = f"page_{hash(url) & 0xFFFFFFFF:08x}"
            
            page_query = """
            MERGE (p:Page {id: $id})
            ON CREATE SET p.url = $url,
                          p.domain = $domain,
                          p.title = $title,
                          p.visited_at = $visited_at,
                          p.screenshot_hash = ''
            ON MATCH SET p.title = $title,
                         p.visited_at = $visited_at
            """
            params = {
                "id": page_id,
                "url": url or "",
                "domain": domain,
                "title": title or "",
                "visited_at": now
            }
            self.conn.execute(page_query, params)

            if tool_call_id:
                rel_query = """
                MATCH (tc:ToolCall {id: $tool_call_id}), (p:Page {id: $page_id})
                MERGE (tc)-[:NAVIGATED_TO]->(p)
                """
                self.conn.execute(rel_query, {"tool_call_id": tool_call_id, "page_id": page_id})

            return page_id

    def upsert_entity(self, tool_call_id: str, page_id: str, etype: str, name: str,
                      price: float | None, currency: str, attrs: dict) -> str:
        with self._lock:
            now = _utcnow()
            domain = ""
            if page_id:
                res = self.conn.execute("MATCH (p:Page {id: $id}) RETURN p.domain", {"id": page_id})
                if res.has_next():
                    domain = res.get_next()[0]

            product_id = attrs.get("product_id") if isinstance(attrs, dict) else None
            canon_key = canonical_key(domain, product_id, name)
            entity_id = f"ent_{canon_key[:16]}"
            price_val = float(price) if (price is not None and not (isinstance(price, float) and (price != price))) else -1.0
            attrs_str = json.dumps(attrs, default=str) if attrs else "{}"

            query = """
            MERGE (e:Entity {id: $id})
            ON CREATE SET e.type = $etype,
                          e.name = $name,
                          e.price = $price,
                          e.currency = $currency,
                          e.attrs = $attrs,
                          e.canonical_key = $canonical_key,
                          e.first_seen_at = $now,
                          e.last_seen_at = $now
            ON MATCH SET e.price = $price,
                         e.attrs = $attrs,
                         e.last_seen_at = $now
            """
            params = {
                "id": entity_id,
                "etype": etype or "unknown",
                "name": name or "",
                "price": price_val,
                "currency": currency or "INR",
                "attrs": attrs_str,
                "canonical_key": canon_key,
                "now": now
            }
            self.conn.execute(query, params)

            if tool_call_id:
                rel1 = """
                MATCH (tc:ToolCall {id: $tool_call_id}), (e:Entity {id: $entity_id})
                MERGE (tc)-[:EXTRACTED]->(e)
                """
                self.conn.execute(rel1, {"tool_call_id": tool_call_id, "entity_id": entity_id})

            if page_id:
                rel2 = """
                MATCH (e:Entity {id: $entity_id}), (p:Page {id: $page_id})
                MERGE (e)-[:FOUND_ON]->(p)
                """
                self.conn.execute(rel2, {"entity_id": entity_id, "page_id": page_id})

            return entity_id


    def delete_artifact(self, artifact_id: str) -> bool:
        """Explicit store-level deletion for artifact compensation and cleanup."""
        if not artifact_id or not isinstance(artifact_id, str) or not artifact_id.strip():
            return False
        with self._lock:
            try:
                query = "MATCH (a:Artifact {id: $id}) DETACH DELETE a"
                self.conn.execute(query, {"id": artifact_id.strip()})
                return True
            except Exception as e:
                logger.error(f"Failed to delete artifact '{artifact_id}': {e}")
                return False

    def save_artifact(
        self,
        agent_id: str,
        title: str,
        artifact_type: str,
        content: str,
        source_url: str = "",
        source_run_id: str = "",
        source_task_id: str = "",
    ) -> str:
        with self._lock:
            now = _utcnow()
            artifact_id = f"art_{uuid.uuid4().hex}"

            # Validate agent existence before storing as graph-Agent provenance
            valid_agent_id = ""
            if agent_id and isinstance(agent_id, str) and agent_id.strip():
                clean_aid = agent_id.strip()
                try:
                    check_q = "MATCH (ag:Agent {id: $aid}) RETURN count(ag)"
                    check_res = self.conn.execute(check_q, {"aid": clean_aid})
                    if check_res.has_next() and check_res.get_next()[0] > 0:
                        valid_agent_id = clean_aid
                    else:
                        logger.warning(
                            f"save_artifact: agent_id '{clean_aid}' does not exist in Agent table. "
                            f"Storing empty source_agent_id and omitting CREATED_ARTIFACT relationship."
                        )
                except Exception as check_err:
                    logger.warning(f"save_artifact: failed to verify agent existence: {check_err}")

            query = """
            MERGE (a:Artifact {id: $id})
            ON CREATE SET a.artifact_type = $type,
                          a.title = $title,
                          a.content = $content,
                          a.created_at = $now,
                          a.source_agent_id = $agent_id,
                          a.source_url = $source_url,
                          a.source_run_id = $source_run_id,
                          a.source_task_id = $source_task_id
            """
            params = {
                "id": artifact_id,
                "type": artifact_type or "document",
                "title": title or "Untitled Artifact",
                "content": content or "",
                "now": now,
                "agent_id": valid_agent_id,
                "source_url": source_url or "",
                "source_run_id": source_run_id or "",
                "source_task_id": source_task_id or "",
            }
            node_created = False
            try:
                self.conn.execute(query, params)
                node_created = True
                if valid_agent_id:
                    rel_q = """
                    MATCH (ag:Agent {id: $agent_id}), (a:Artifact {id: $id})
                    MERGE (ag)-[:CREATED_ARTIFACT]->(a)
                    """
                    self.conn.execute(rel_q, {"agent_id": valid_agent_id, "id": artifact_id})
                return artifact_id
            except Exception as e:
                logger.warning(f"Failed to save artifact: {e}")
                if node_created:
                    # Partial write occurred: Artifact node was created, but subsequent step failed.
                    # Perform best-effort rollback so no orphaned unlinked artifact node remains in Kùzu.
                    try:
                        cleanup_q = "MATCH (a:Artifact {id: $id}) DETACH DELETE a"
                        self.conn.execute(cleanup_q, {"id": artifact_id})
                        logger.info(f"Successfully rolled back partial artifact node {artifact_id}.")
                    except Exception as cleanup_err:
                        logger.error(f"CRITICAL: Failed to clean up partial artifact node {artifact_id}: {cleanup_err}")
                        raise PartialWriteRollbackError(
                            artifact_id,
                            f"Partial artifact write failed and compensating rollback failed for '{artifact_id}': {cleanup_err}"
                        ) from e
                return ""

    def get_artifact(self, artifact_id: str) -> dict:
        with self._lock:
            query = (
                "MATCH (a:Artifact {id: $id}) "
                "RETURN a.title, a.artifact_type, a.content, a.created_at, "
                "a.source_agent_id, a.source_url, a.source_run_id, a.source_task_id"
            )
            try:
                res = self.conn.execute(query, {"id": artifact_id})
                if res.has_next():
                    row = res.get_next()
                    return {
                        "id": artifact_id,
                        "title": row[0],
                        "artifact_type": row[1],
                        "content": row[2],
                        "created_at": str(row[3]) if row[3] is not None else "",
                        "source_agent_id": row[4] if len(row) > 4 and row[4] is not None else "",
                        "source_url": row[5] if len(row) > 5 and row[5] is not None else "",
                        "source_run_id": row[6] if len(row) > 6 and row[6] is not None else "",
                        "source_task_id": row[7] if len(row) > 7 and row[7] is not None else "",
                    }
            except Exception as e:
                logger.warning(f"Failed to get artifact {artifact_id}: {e}")
            return {}

    def find_recent_entities(self, etype: str, max_price: float | None = None,
                             since_minutes: int = 30, include_stale: bool = False) -> list[dict]:
        with self._lock:
            cutoff = _utcnow() - timedelta(minutes=since_minutes)
            query = "MATCH (e:Entity {type: $etype}) "
            params = {"etype": etype}

            where_clauses = []
            if not include_stale:
                where_clauses.append("e.last_seen_at >= $cutoff")
                params["cutoff"] = cutoff

            if max_price is not None:
                where_clauses.append("e.price > 0.0 AND e.price <= $max_price")
                params["max_price"] = float(max_price)

            if where_clauses:
                query += "WHERE " + " AND ".join(where_clauses) + " "

            query += "RETURN e.id, e.type, e.name, e.price, e.currency, e.attrs, e.last_seen_at"

            t0 = time.perf_counter()
            res = self.conn.execute(query, params)
            results = []
            while res.has_next():
                row = res.get_next()
                try:
                    parsed_attrs = json.loads(row[5]) if row[5] else {}
                except Exception:
                    parsed_attrs = {}

                price_out = row[3] if row[3] >= 0 else None
                results.append({
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "price": price_out,
                    "currency": row[4],
                    "attrs": parsed_attrs,
                    "last_seen_at": str(row[6])
                })
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(f"Graph read find_recent_entities ({len(results)} items) completed in {elapsed_ms:.2f}ms")
            return results

    def get_agent_provenance(self, agent_id: str) -> dict:
        with self._lock:
            t0 = time.perf_counter()
            res = self.conn.execute(
                "MATCH (a:Agent {id: $id}) RETURN a.id, a.task, a.status, a.created_at, a.completed_at",
                {"id": agent_id}
            )
            if not res.has_next():
                return {"error": "Agent not found"}

            row = res.get_next()
            agent_data = {
                "id": row[0],
                "task": row[1],
                "status": row[2],
                "created_at": str(row[3]),
                "completed_at": str(row[4]) if row[4] else None,
                "tool_calls": []
            }

            tc_res = self.conn.execute(
                "MATCH (a:Agent {id: $id})-[:EXECUTED]->(tc:ToolCall) "
                "RETURN tc.id, tc.tool_name, tc.args_json, tc.result_summary, tc.step_index, tc.timestamp, tc.success "
                "ORDER BY tc.step_index",
                {"id": agent_id}
            )
            while tc_res.has_next():
                tc_row = tc_res.get_next()
                tc_id = tc_row[0]

                # Find pages
                pages = []
                p_res = self.conn.execute(
                    "MATCH (tc:ToolCall {id: $tc_id})-[:NAVIGATED_TO]->(p:Page) RETURN p.url, p.title",
                    {"tc_id": tc_id}
                )
                while p_res.has_next():
                    p_row = p_res.get_next()
                    pages.append({"url": p_row[0], "title": p_row[1]})

                # Find entities
                entities = []
                e_res = self.conn.execute(
                    "MATCH (tc:ToolCall {id: $tc_id})-[:EXTRACTED]->(e:Entity) RETURN e.name, e.price, e.type",
                    {"tc_id": tc_id}
                )
                while e_res.has_next():
                    e_row = e_res.get_next()
                    entities.append({"name": e_row[0], "price": e_row[1], "type": e_row[2]})

                agent_data["tool_calls"].append({
                    "id": tc_id,
                    "tool_name": tc_row[1],
                    "args_json": tc_row[2],
                    "result_summary": tc_row[3],
                    "step_index": tc_row[4],
                    "timestamp": str(tc_row[5]),
                    "success": tc_row[6],
                    "pages": pages,
                    "entities": entities
                })

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(f"Graph read get_agent_provenance ({agent_id}) completed in {elapsed_ms:.2f}ms")
            return agent_data

    def get_stats(self) -> dict:
        with self._lock:
            t0 = time.perf_counter()
            stats = {}
            for table in ["Agent", "Tab", "Page", "Entity", "ToolCall", "Artifact", "UserMemory", "Skill", "Reflection"]:
                res = self.conn.execute(f"MATCH (n:{table}) RETURN count(n)")
                stats[table.lower() + "_count"] = res.get_next()[0] if res.has_next() else 0
            
            db_file_size = 0
            if os.path.exists(self.db_path):
                db_file_size = os.path.getsize(self.db_path)
            elif os.path.exists(os.path.dirname(self.db_path)):
                for root, dirs, files in os.walk(os.path.dirname(self.db_path)):
                    for f in files:
                        db_file_size += os.path.getsize(os.path.join(root, f))
            stats["db_file_bytes"] = db_file_size
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(f"Graph read get_stats completed in {elapsed_ms:.2f}ms")
            return stats

    def get_query_focused_excerpt(self, text: str, query: str, max_len: int = 500) -> str:
        if len(text) <= max_len:
            return text
            
        import re
        query_words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 3]
        if not query_words:
            return text[:max_len-3] + "..."
            
        lower_text = text.lower()
        best_idx = 0
        
        for word in query_words:
            idx = lower_text.find(word)
            if idx != -1:
                best_idx = idx
                break
                
        start = max(0, best_idx - 100)
        end = min(len(text), start + max_len - 3)
        
        if end == len(text):
            start = max(0, end - max_len + 3)
            
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet[3:]
        if end < len(text):
            snippet = snippet + "..."
            
        return snippet

    def search_memory(self, query: str, domain: str = "ALL", limit: int = 10) -> list[dict]:
        with self._lock:
            t0 = time.perf_counter()
            
            norm_domain = (domain or "ALL").strip().upper() if isinstance(domain, str) else "ALL"
            domains_to_search = []
            if norm_domain == "ENTITIES":
                domains_to_search = [("Entity", "Entity_FTS")]
            elif norm_domain == "HISTORY":
                domains_to_search = [("Agent", "Agent_FTS"), ("ToolCall", "ToolCall_FTS")]
            elif norm_domain == "ARTIFACTS":
                domains_to_search = [("Artifact", "Artifact_FTS")]
            elif norm_domain == "USER":
                domains_to_search = [("UserMemory", "UserMemory_FTS")]
            elif norm_domain == "SKILLS":
                domains_to_search = [("Skill", "Skill_FTS")]
            elif norm_domain == "ALL":
                domains_to_search = [
                    ("Entity", "Entity_FTS"),
                    ("Agent", "Agent_FTS"),
                    ("ToolCall", "ToolCall_FTS"),
                    ("Artifact", "Artifact_FTS"),
                    ("UserMemory", "UserMemory_FTS"),
                    ("Skill", "Skill_FTS")
                ]
            else:
                logger.warning(f"Unknown search domain '{domain}', returning empty results.")
            all_results = []
            domains_attempted = len(domains_to_search)
            failed_domains = []
            
            for table_name, index_name in domains_to_search:
                q = f"""
                CALL QUERY_FTS_INDEX('{table_name}', '{index_name}', $search_query)
                YIELD node, score
                RETURN node, score
                """
                try:
                    res = self.conn.execute(q, {"search_query": query})
                    while res.has_next():
                        node, score = res.get_next()
                        
                        result_dict = {
                            "relevance_score": score,
                            "source_id": node.get("id"),
                            "memory_type": table_name,
                            "timestamp": "",
                            "content_excerpt": "",
                            "provenance": ""
                        }
                        
                        if table_name == "Entity":
                            first_seen = str(node.get("first_seen_at")) if node.get("first_seen_at") is not None else ""
                            last_seen = str(node.get("last_seen_at")) if node.get("last_seen_at") is not None else first_seen
                            result_dict["timestamp"] = first_seen
                            result_dict["first_seen_at"] = first_seen
                            result_dict["last_seen_at"] = last_seen
                            attrs = node.get("attrs")
                            parsed_attrs = {}
                            if attrs:
                                try:
                                    parsed_attrs = json.loads(attrs) if isinstance(attrs, str) else dict(attrs)
                                except Exception as json_err:
                                    logger.warning(f"Malformed attrs JSON for entity {node.get('id')}: {json_err}")
                                    parsed_attrs = {"raw": str(attrs)}
                            type_ = node.get("type")
                            name = node.get("name")
                            price = node.get("price")
                            currency = node.get("currency")
                            content = f"Entity Type: {type_}\nName: {name}\nPrice: {price} {currency}\nAttributes: {json.dumps(parsed_attrs)}"
                            result_dict["content_excerpt"] = content
                            
                            prov_q = "MATCH (e:Entity {id: $id})<-[:EXTRACTED]-(tc:ToolCall) RETURN tc.id LIMIT 1"
                            prov_res = self.conn.execute(prov_q, {"id": node.get("id")})
                            if prov_res.has_next():
                                result_dict["provenance"] = prov_res.get_next()[0]
                                
                        elif table_name == "Agent":
                            ts = node.get("created_at")
                            result_dict["timestamp"] = str(ts) if ts is not None else ""
                            result_dict["content_excerpt"] = f"Task: {node.get('task')}\nStatus: {node.get('status')}"
                            result_dict["provenance"] = node.get("id")
                            
                        elif table_name == "ToolCall":
                            ts = node.get("timestamp")
                            result_dict["timestamp"] = str(ts) if ts is not None else ""
                            tool_name = node.get("tool_name")
                            args_json = node.get("args_json")
                            result_summary = node.get("result_summary")
                            result_dict["content_excerpt"] = f"Tool: {tool_name}\nArgs: {args_json}\nResult: {result_summary}"
                            
                            prov_q = "MATCH (tc:ToolCall {id: $id})<-[:EXECUTED]-(a:Agent) RETURN a.id LIMIT 1"
                            prov_res = self.conn.execute(prov_q, {"id": node.get("id")})
                            if prov_res.has_next():
                                result_dict["provenance"] = prov_res.get_next()[0]
                                
                        elif table_name == "Artifact":
                            ts = node.get("created_at")
                            result_dict["timestamp"] = str(ts) if ts is not None else ""
                            title = node.get("title")
                            content = node.get("content")
                            atype = node.get("artifact_type")
                            source_url = node.get("source_url") or ""
                            source_run_id = node.get("source_run_id") or ""
                            source_task_id = node.get("source_task_id") or ""
                            if source_url:
                                result_dict["content_excerpt"] = f"Artifact [{atype}]: {title} (URL: {source_url})\n{content}"
                                result_dict["source_url"] = str(source_url)
                            else:
                                result_dict["content_excerpt"] = f"Artifact [{atype}]: {title}\n{content}"

                            if source_run_id:
                                result_dict["source_run_id"] = str(source_run_id)
                            if source_task_id:
                                result_dict["source_task_id"] = str(source_task_id)

                            prov_q = "MATCH (a:Artifact {id: $id})<-[:CREATED_ARTIFACT]-(ag:Agent) RETURN ag.id LIMIT 1"
                            prov_res = self.conn.execute(prov_q, {"id": node.get("id")})
                            if prov_res.has_next():
                                result_dict["provenance"] = prov_res.get_next()[0]
                            elif node.get("source_agent_id"):
                                result_dict["provenance"] = str(node.get("source_agent_id"))
                            # If no Agent node or source_agent_id, keep provenance explicitly empty

                        elif table_name == "UserMemory":
                            ts = node.get("updated_at") or node.get("created_at")
                            result_dict["timestamp"] = str(ts) if ts is not None else ""
                            cat = node.get("category")
                            k = node.get("key")
                            v = node.get("value")
                            result_dict["content_excerpt"] = f"User Memory [{cat}]: {k} = {v}"
                            prov_q = "MATCH (u:UserMemory {id: $id})-[:LEARNED_FROM]->(ag:Agent) RETURN ag.id LIMIT 1"
                            prov_res = self.conn.execute(prov_q, {"id": node.get("id")})
                            if prov_res.has_next():
                                result_dict["provenance"] = prov_res.get_next()[0]
                            elif node.get("source_agent_id"):
                                result_dict["provenance"] = str(node.get("source_agent_id"))

                        elif table_name == "Skill":
                            ts = node.get("updated_at") or node.get("created_at")
                            result_dict["timestamp"] = str(ts) if ts is not None else ""
                            sname = node.get("name")
                            sdomain = node.get("domain")
                            sdesc = node.get("description")
                            scode = node.get("code_snippet")
                            result_dict["content_excerpt"] = f"Skill: {sname} ({sdomain})\nDescription: {sdesc}\nCode:\n{scode}"
                            prov_q = "MATCH (sk:Skill {id: $id})-[:DISTILLED_FROM]->(ag:Agent) RETURN ag.id LIMIT 1"
                            prov_res = self.conn.execute(prov_q, {"id": node.get("id")})
                            if prov_res.has_next():
                                result_dict["provenance"] = prov_res.get_next()[0]
                            elif node.get("source_agent_id"):
                                result_dict["provenance"] = str(node.get("source_agent_id"))

                        all_results.append(result_dict)
                except Exception as e:
                    logger.warning(f"FTS search failed for {table_name}: {e}")
                    failed_domains.append(table_name)
                    
            if domains_attempted > 0 and len(failed_domains) == domains_attempted:
                raise RuntimeError(
                    f"FTS memory search failed across all queried domains ({', '.join(failed_domains)})."
                )

            # Sort by score descending
            all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Truncate to limit
            all_results = all_results[:limit]
            
            # Apply truncation to text post-retrieval
            for r in all_results:
                r["content_excerpt"] = self.get_query_focused_excerpt(r["content_excerpt"], query)
                    
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(f"Graph read search_memory completed in {elapsed_ms:.2f}ms")
            
            if failed_domains:
                class MemorySearchResults(list):
                    def __init__(self, items, failed_domains=None):
                        super().__init__(items)
                        self.failed_domains = failed_domains or []
                return MemorySearchResults(all_results, failed_domains=failed_domains)

            return all_results


    def search_procedures(self, abstract_query: str, limit: int = 2, threshold: float = 0.5) -> list[dict]:
        with self._lock:
            t0 = time.perf_counter()
            q = f"""
            CALL QUERY_FTS_INDEX('Artifact', 'Artifact_FTS', $search_query)
            YIELD node, score
            WITH node, score
            WHERE node.artifact_type = 'procedure' AND score >= $threshold
            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
            """
            results = []
            try:
                res = self.conn.execute(q, {"search_query": abstract_query, "threshold": threshold, "limit": limit})
                while res.has_next():
                    node, score = res.get_next()
                    results.append({
                        "id": node.get("id"),
                        "title": node.get("title"),
                        "content": node.get("content"),
                        "score": score
                    })
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Graph: search_procedures failed: {e}")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            import logging
            logging.getLogger(__name__).debug(f"Graph read search_procedures ({abstract_query}) completed in {elapsed_ms:.2f}ms")
            return results

    def set_user_memory(self, key: str, value: str, category: str = "preference", confidence: float = 1.0, agent_id: str | None = None) -> str:
        """Upsert a piece of UserMemory (preference, profile fact, constraint)."""
        with self._lock:
            now = _utcnow()
            mem_id = f"usr_{hash(f'{category}:{key}') & 0xFFFFFFFF:08x}"
            query = """
            MERGE (u:UserMemory {id: $id})
            ON CREATE SET u.category = $category,
                          u.key = $key,
                          u.value = $value,
                          u.confidence = $confidence,
                          u.source_agent_id = $agent_id,
                          u.created_at = $now,
                          u.updated_at = $now
            ON MATCH SET u.value = $value,
                         u.confidence = $confidence,
                         u.source_agent_id = $agent_id,
                         u.updated_at = $now
            """
            params = {
                "id": mem_id,
                "category": category,
                "key": key,
                "value": value,
                "confidence": float(confidence),
                "agent_id": agent_id or "",
                "now": now
            }
            self.conn.execute(query, params)

            if agent_id:
                rel_query = """
                MATCH (u:UserMemory {id: $mem_id}), (ag:Agent {id: $agent_id})
                MERGE (u)-[:LEARNED_FROM]->(ag)
                """
                try:
                    self.conn.execute(rel_query, {"mem_id": mem_id, "agent_id": agent_id})
                except Exception:
                    pass

            return mem_id

    def get_all_user_memory(self, category: str | None = None) -> list[dict]:
        """Retrieve all user memories, optionally filtered by category."""
        with self._lock:
            if category:
                query = "MATCH (u:UserMemory) WHERE u.category = $category RETURN u ORDER BY u.updated_at DESC"
                params = {"category": category}
            else:
                query = "MATCH (u:UserMemory) RETURN u ORDER BY u.updated_at DESC"
                params = {}
            res = self.conn.execute(query, params)
            items = []
            while res.has_next():
                row = res.get_next()
                u = row[0]
                items.append({
                    "id": u.get("id"),
                    "category": u.get("category"),
                    "key": u.get("key"),
                    "value": u.get("value"),
                    "confidence": u.get("confidence"),
                    "updated_at": str(u.get("updated_at"))
                })
            return items

    @staticmethod
    def _extract_skill_tokens(text: str) -> set[str]:
        """Extract normalized semantic tokens from a skill name or description."""
        import re
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        stop_words = {"the", "a", "an", "in", "on", "for", "to", "via", "and", "or", "of", "with", "from", "at", "by"}
        synonyms = {
            "repository": "repo",
            "repositories": "repo",
            "repos": "repo",
            "searching": "search",
            "searches": "search",
            "filtering": "filter",
            "filters": "filter",
        }
        extracted = set()
        for t in tokens:
            if len(t) > 1 and t not in stop_words:
                extracted.add(synonyms.get(t, t))
        return extracted

    def save_skill(self, name: str, domain: str, description: str, code_snippet: str, agent_id: str | None = None, skill_id: str | None = None) -> str:
        """Upsert a distilled procedural skill with semantic deduplication."""
        with self._lock:
            now = _utcnow()
            domain = domain or "*"
            target_id = skill_id

            # Semantic deduplication if target_id not provided
            if not target_id:
                # 1. Exact match hash
                exact_hash_id = f"skl_{hash(f'{domain}:{name}') & 0xFFFFFFFF:08x}"
                check_q = "MATCH (s:Skill {id: $id}) RETURN s.id"
                res_exact = self.conn.execute(check_q, {"id": exact_hash_id})
                if res_exact.has_next():
                    target_id = exact_hash_id
                else:
                    # 2. Token overlap similarity match against existing skills in this domain
                    find_q = "MATCH (s:Skill) WHERE s.domain = $domain OR s.domain = '*' RETURN s"
                    res = self.conn.execute(find_q, {"domain": domain})
                    new_tokens = self._extract_skill_tokens(name)
                    best_match_id = None
                    best_jaccard = 0.0

                    while res.has_next():
                        existing_skill = res.get_next()[0]
                        existing_name = existing_skill.get("name") or ""
                        existing_tokens = self._extract_skill_tokens(existing_name)
                        if new_tokens and existing_tokens:
                            intersection = new_tokens & existing_tokens
                            union = new_tokens | existing_tokens
                            jaccard = len(intersection) / len(union)
                            # If high overlap or one token set is subset of another
                            is_subset = new_tokens.issubset(existing_tokens) or existing_tokens.issubset(new_tokens)
                            if jaccard > 0.45 or (is_subset and len(intersection) >= 2):
                                if jaccard > best_jaccard:
                                    best_jaccard = jaccard
                                    best_match_id = existing_skill.get("id")

                    if best_match_id:
                        target_id = best_match_id
                    else:
                        target_id = exact_hash_id

            query = """
            MERGE (s:Skill {id: $id})
            ON CREATE SET s.name = $name,
                          s.domain = $domain,
                          s.description = $description,
                          s.code_snippet = $code_snippet,
                          s.usage_count = 1,
                          s.success_count = 1,
                          s.created_at = $now,
                          s.updated_at = $now,
                          s.source_agent_id = $agent_id
            ON MATCH SET s.description = CASE WHEN size($description) > size(s.description) THEN $description ELSE s.description END,
                         s.code_snippet = CASE WHEN size($code_snippet) > 0 THEN $code_snippet ELSE s.code_snippet END,
                         s.usage_count = s.usage_count + 1,
                         s.success_count = s.success_count + 1,
                         s.updated_at = $now,
                         s.source_agent_id = $agent_id
            """
            params = {
                "id": target_id,
                "name": name,
                "domain": domain,
                "description": description or "",
                "code_snippet": code_snippet or "",
                "agent_id": agent_id or "",
                "now": now
            }
            self.conn.execute(query, params)

            if agent_id:
                rel_query = """
                MATCH (s:Skill {id: $skill_id}), (ag:Agent {id: $agent_id})
                MERGE (s)-[:DISTILLED_FROM]->(ag)
                """
                try:
                    self.conn.execute(rel_query, {"skill_id": target_id, "agent_id": agent_id})
                except Exception:
                    pass

            return target_id

    def consolidate_skills(self, domain: str | None = None) -> list[dict]:
        """Scan skills and merge duplicate/near-identical skills on the same domain."""
        with self._lock:
            now = _utcnow()
            if domain:
                q = "MATCH (s:Skill) WHERE s.domain = $domain RETURN s"
                params = {"domain": domain}
            else:
                q = "MATCH (s:Skill) RETURN s"
                params = {}

            res = self.conn.execute(q, params)
            skills = []
            while res.has_next():
                s = res.get_next()[0]
                skills.append({
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "domain": s.get("domain") or "*",
                    "description": s.get("description") or "",
                    "code_snippet": s.get("code_snippet") or "",
                    "usage_count": s.get("usage_count") or 0,
                    "success_count": s.get("success_count") or 0,
                })

            # Group by domain
            by_domain: dict[str, list[dict]] = {}
            for sk in skills:
                by_domain.setdefault(sk["domain"], []).append(sk)

            consolidated = []
            for dom, dom_skills in by_domain.items():
                if len(dom_skills) < 2:
                    continue

                visited = set()
                for i in range(len(dom_skills)):
                    if dom_skills[i]["id"] in visited:
                        continue
                    canonical = dom_skills[i]
                    tokens_a = self._extract_skill_tokens(canonical["name"])
                    duplicates = []

                    for j in range(i + 1, len(dom_skills)):
                        other = dom_skills[j]
                        if other["id"] in visited:
                            continue
                        tokens_b = self._extract_skill_tokens(other["name"])
                        if tokens_a and tokens_b:
                            intersection = tokens_a & tokens_b
                            union = tokens_a | tokens_b
                            jaccard = len(intersection) / len(union)
                            is_subset = tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a)
                            if jaccard > 0.45 or (is_subset and len(intersection) >= 2):
                                duplicates.append(other)
                                visited.add(other["id"])

                    if duplicates:
                        visited.add(canonical["id"])
                        # Pick best description and snippet
                        total_usage = canonical["usage_count"] + sum(d["usage_count"] for d in duplicates)
                        total_success = canonical["success_count"] + sum(d["success_count"] for d in duplicates)
                        best_desc = canonical["description"]
                        best_code = canonical["code_snippet"]
                        for d in duplicates:
                            if len(d["description"]) > len(best_desc):
                                best_desc = d["description"]
                            if len(d["code_snippet"]) > len(best_code):
                                best_code = d["code_snippet"]

                        # Update canonical
                        upd_q = """
                        MATCH (s:Skill {id: $id})
                        SET s.description = $description,
                            s.code_snippet = $code_snippet,
                            s.usage_count = $usage_count,
                            s.success_count = $success_count,
                            s.updated_at = $now
                        """
                        self.conn.execute(upd_q, {
                            "id": canonical["id"],
                            "description": best_desc,
                            "code_snippet": best_code,
                            "usage_count": total_usage,
                            "success_count": total_success,
                            "now": now
                        })

                        # Delete duplicate nodes
                        for d in duplicates:
                            del_q = "MATCH (s:Skill {id: $id}) DETACH DELETE s"
                            self.conn.execute(del_q, {"id": d["id"]})

                        consolidated.append({
                            "canonical_id": canonical["id"],
                            "canonical_name": canonical["name"],
                            "domain": dom,
                            "merged_ids": [d["id"] for d in duplicates],
                            "merged_names": [d["name"] for d in duplicates]
                        })

            return consolidated

    def find_skills_for_domain(self, domain: str, limit: int = 5) -> list[dict]:
        """Find relevant skills for a given web domain."""
        with self._lock:
            query = """
            MATCH (s:Skill)
            WHERE s.domain = $domain OR s.domain = '*'
            RETURN s
            ORDER BY s.success_count DESC, s.usage_count DESC
            LIMIT $limit
            """
            res = self.conn.execute(query, {"domain": domain, "limit": limit})
            skills = []
            while res.has_next():
                s = res.get_next()[0]
                skills.append({
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "domain": s.get("domain"),
                    "description": s.get("description"),
                    "code_snippet": s.get("code_snippet"),
                    "usage_count": s.get("usage_count"),
                    "success_count": s.get("success_count")
                })
            return skills

    def record_reflection(self, agent_id: str, is_novel: bool, novelty_reason: str,
                          critique: str, lesson_learned: str, suggested_skill_name: str = "") -> str:
        """Record post-task self-evaluation and meta-cognitive reflection."""
        with self._lock:
            now = _utcnow()
            ref_id = f"ref_{uuid.uuid4().hex[:16]}"
            query = """
            CREATE (r:Reflection {
                id: $id,
                agent_id: $agent_id,
                is_novel: $is_novel,
                novelty_reason: $novelty_reason,
                critique: $critique,
                lesson_learned: $lesson_learned,
                suggested_skill_name: $suggested_skill_name,
                created_at: $now
            })
            """
            params = {
                "id": ref_id,
                "agent_id": agent_id,
                "is_novel": bool(is_novel),
                "novelty_reason": novelty_reason or "",
                "critique": critique or "",
                "lesson_learned": lesson_learned or "",
                "suggested_skill_name": suggested_skill_name or "",
                "now": now
            }
            self.conn.execute(query, params)

            rel_query = """
            MATCH (r:Reflection {id: $ref_id}), (ag:Agent {id: $agent_id})
            MERGE (r)-[:REFLECTED_ON]->(ag)
            """
            try:
                self.conn.execute(rel_query, {"ref_id": ref_id, "agent_id": agent_id})
            except Exception:
                pass

            return ref_id

    def get_recent_reflections(self, limit: int = 10) -> list[dict]:
        """Fetch the most recent post-task reflections."""
        with self._lock:
            query = """
            MATCH (r:Reflection)
            RETURN r
            ORDER BY r.created_at DESC
            LIMIT $limit
            """
            res = self.conn.execute(query, {"limit": limit})
            reflections = []
            while res.has_next():
                r = res.get_next()[0]
                reflections.append({
                    "id": r.get("id"),
                    "agent_id": r.get("agent_id"),
                    "is_novel": r.get("is_novel"),
                    "novelty_reason": r.get("novelty_reason"),
                    "critique": r.get("critique"),
                    "lesson_learned": r.get("lesson_learned"),
                    "suggested_skill_name": r.get("suggested_skill_name"),
                    "created_at": str(r.get("created_at"))
                })
            return reflections

    def delete_user_memory(self, key: str) -> bool:
        """Delete a user memory entry by key."""
        with self._lock:
            query = "MATCH (u:UserMemory {key: $key}) DETACH DELETE u"
            self.conn.execute(query, {"key": key})
            return True

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill entry by id."""
        with self._lock:
            query = "MATCH (s:Skill {id: $id}) DETACH DELETE s"
            self.conn.execute(query, {"id": skill_id})
            return True

    def save_site_adapter(self, name: str, domain: str, action_intent: str,
                          code_template: str, parameter_schema: dict | None = None,
                          dom_fingerprint: dict | None = None,
                          adapter_id: str | None = None,
                          episode_id: str | None = None) -> str:
        """Persist or update a compiled Fast-Path SiteAdapter."""
        with self._lock:
            now = _utcnow()
            aid = adapter_id or f"adp_{uuid.uuid4().hex[:12]}"
            param_schema_str = json.dumps(parameter_schema or {})
            dom_fp_str = json.dumps(dom_fingerprint or {})

            query = """
            MERGE (a:SiteAdapter {id: $id})
            ON CREATE SET a.name = $name,
                          a.domain = $domain,
                          a.action_intent = $action_intent,
                          a.code_template = $code_template,
                          a.parameter_schema = $param_schema,
                          a.dom_fingerprint = $dom_fp,
                          a.success_count = 1,
                          a.failure_count = 0,
                          a.last_success_at = $now,
                          a.status = 'active',
                          a.created_at = $now,
                          a.updated_at = $now
            ON MATCH SET a.name = $name,
                         a.action_intent = $action_intent,
                         a.code_template = $code_template,
                         a.parameter_schema = $param_schema,
                         a.dom_fingerprint = $dom_fp,
                         a.status = 'active',
                         a.updated_at = $now
            """
            params = {
                "id": aid,
                "name": name,
                "domain": domain.lower(),
                "action_intent": action_intent,
                "code_template": code_template,
                "param_schema": param_schema_str,
                "dom_fp": dom_fp_str,
                "now": now
            }
            self.conn.execute(query, params)

            if episode_id:
                try:
                    rel_query = """
                    MATCH (a:SiteAdapter {id: $aid}), (ep:Episode {id: $ep_id})
                    MERGE (a)-[:COMPILED_FROM]->(ep)
                    """
                    self.conn.execute(rel_query, {"aid": aid, "ep_id": episode_id})
                except Exception as re:
                    logger.warning(f"Graph: failed to link COMPILED_FROM: {re}")

            return aid

    def find_matching_adapter(self, domain: str, intent_query: str, limit: int = 1) -> list[dict]:
        """Find active compiled adapters for a domain matching an intent."""
        with self._lock:
            dom = domain.lower().strip()
            query = """
            MATCH (a:SiteAdapter)
            WHERE (a.domain = $domain OR a.domain = '*') AND a.status <> 'broken'
            RETURN a
            ORDER BY a.success_count DESC, a.updated_at DESC
            LIMIT 10
            """
            res = self.conn.execute(query, {"domain": dom})
            matches = []
            words = set(intent_query.lower().split())
            while res.has_next():
                item = res.get_next()[0]
                adapter_dict = {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "domain": item.get("domain"),
                    "action_intent": item.get("action_intent"),
                    "code_template": item.get("code_template"),
                    "parameter_schema": json.loads(item.get("parameter_schema") or "{}"),
                    "dom_fingerprint": json.loads(item.get("dom_fingerprint") or "{}"),
                    "success_count": item.get("success_count", 0),
                    "failure_count": item.get("failure_count", 0),
                    "status": item.get("status", "active"),
                }
                target_intent = str(item.get("action_intent", "")).lower()
                target_words = set(target_intent.split())
                overlap = len(words & target_words)
                adapter_dict["match_score"] = overlap
                matches.append(adapter_dict)

            matches.sort(key=lambda x: (x["match_score"], x["success_count"]), reverse=True)
            return matches[:limit]

    def record_adapter_outcome(self, adapter_id: str, success: bool) -> None:
        """Update success/failure statistics and auto-degrade broken adapters."""
        with self._lock:
            now = _utcnow()
            if success:
                query = """
                MATCH (a:SiteAdapter {id: $id})
                SET a.success_count = a.success_count + 1,
                    a.last_success_at = $now,
                    a.status = 'active',
                    a.updated_at = $now
                """
            else:
                query = """
                MATCH (a:SiteAdapter {id: $id})
                SET a.failure_count = a.failure_count + 1,
                    a.status = CASE WHEN a.failure_count >= 3 THEN 'broken' ELSE 'degraded' END,
                    a.updated_at = $now
                """
            try:
                self.conn.execute(query, {"id": adapter_id, "now": now})
            except Exception as e:
                logger.warning(f"Graph: failed to record adapter outcome: {e}")

    def get_all_adapters(self, domain: str | None = None) -> list[dict]:
        """Return all stored site adapters, optionally filtered by domain."""
        with self._lock:
            if domain:
                query = "MATCH (a:SiteAdapter) WHERE a.domain = $domain RETURN a ORDER BY a.updated_at DESC"
                res = self.conn.execute(query, {"domain": domain.lower()})
            else:
                query = "MATCH (a:SiteAdapter) RETURN a ORDER BY a.updated_at DESC"
                res = self.conn.execute(query)
            adapters = []
            while res.has_next():
                a = res.get_next()[0]
                adapters.append({
                    "id": a.get("id"),
                    "name": a.get("name"),
                    "domain": a.get("domain"),
                    "action_intent": a.get("action_intent"),
                    "code_template": a.get("code_template"),
                    "parameter_schema": json.loads(a.get("parameter_schema") or "{}"),
                    "dom_fingerprint": json.loads(a.get("dom_fingerprint") or "{}"),
                    "success_count": a.get("success_count", 0),
                    "failure_count": a.get("failure_count", 0),
                    "status": a.get("status", "active"),
                    "created_at": str(a.get("created_at")),
                    "updated_at": str(a.get("updated_at"))
                })
            return adapters

    def delete_adapter(self, adapter_id: str) -> bool:
        """Delete an adapter by ID."""
        with self._lock:
            query = "MATCH (a:SiteAdapter {id: $id}) DETACH DELETE a"
            self.conn.execute(query, {"id": adapter_id})
            return True

    def record_antipattern(self, domain: str, failed_selector: str, trap_type: str, mitigation_hint: str) -> str:
        """Record or increment a known anti-pattern for a domain."""
        with self._lock:
            now = _utcnow()
            ap_id = f"ap_{hash(f'{domain}:{failed_selector}') & 0xFFFFFFFF:08x}"
            query = """
            MERGE (ap:AntiPattern {id: $id})
            ON CREATE SET ap.domain = $domain,
                          ap.failed_selector = $failed_selector,
                          ap.trap_type = $trap_type,
                          ap.mitigation_hint = $mitigation_hint,
                          ap.occurrence_count = 1,
                          ap.last_seen_at = $now,
                          ap.created_at = $now
            ON MATCH SET ap.occurrence_count = ap.occurrence_count + 1,
                         ap.mitigation_hint = $mitigation_hint,
                         ap.last_seen_at = $now
            """
            params = {
                "id": ap_id,
                "domain": domain.lower(),
                "failed_selector": failed_selector,
                "trap_type": trap_type,
                "mitigation_hint": mitigation_hint,
                "now": now
            }
            self.conn.execute(query, params)
            return ap_id

    def get_domain_antipatterns(self, domain: str, limit: int = 5) -> list[dict]:
        """Fetch known failure patterns for a domain."""
        with self._lock:
            query = """
            MATCH (ap:AntiPattern)
            WHERE ap.domain = $domain OR ap.domain = '*'
            RETURN ap
            ORDER BY ap.occurrence_count DESC, ap.last_seen_at DESC
            LIMIT $limit
            """
            res = self.conn.execute(query, {"domain": domain.lower(), "limit": limit})
            antipatterns = []
            while res.has_next():
                row = res.get_next()[0]
                antipatterns.append({
                    "id": row.get("id"),
                    "domain": row.get("domain"),
                    "failed_selector": row.get("failed_selector"),
                    "trap_type": row.get("trap_type"),
                    "mitigation_hint": row.get("mitigation_hint"),
                    "occurrence_count": row.get("occurrence_count", 1)
                })
            return antipatterns

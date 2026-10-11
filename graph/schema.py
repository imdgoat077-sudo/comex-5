"""
DDL definitions and migration runner for the Comex Agent Graph Memory.
Uses Kuzu's Cypher-like DDL.
"""

SCHEMA_VERSION = "4"

NODE_TABLES = [
    """
    CREATE NODE TABLE IF NOT EXISTS Agent(
        id STRING,
        task STRING,
        intent STRING,
        status STRING,
        created_at TIMESTAMP,
        completed_at TIMESTAMP,
        step_count INT64,
        parent_agent_id STRING,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Tab(
        id STRING,
        is_hidden BOOLEAN,
        cdp_ws_url STRING,
        created_at TIMESTAMP,
        closed_at TIMESTAMP,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Page(
        id STRING,
        url STRING,
        domain STRING,
        title STRING,
        visited_at TIMESTAMP,
        screenshot_hash STRING,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Entity(
        id STRING,
        type STRING,
        name STRING,
        price DOUBLE,
        currency STRING,
        attrs STRING,
        canonical_key STRING,
        first_seen_at TIMESTAMP,
        last_seen_at TIMESTAMP,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS ToolCall(
        id STRING,
        tool_name STRING,
        args_json STRING,
        result_summary STRING,
        step_index INT64,
        timestamp TIMESTAMP,
        success BOOLEAN,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Artifact(
        id STRING,
        artifact_type STRING,
        title STRING,
        content STRING,
        created_at TIMESTAMP,
        source_agent_id STRING,
        source_url STRING,
        source_run_id STRING,
        source_task_id STRING,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Episode(
        id              STRING,
        agent_id        STRING,
        outcome_status  STRING,
        self_reported   STRING,
        constraint_check STRING,
        created_at      TIMESTAMP,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS UserMemory(
        id              STRING,
        category        STRING,
        key             STRING,
        value           STRING,
        confidence      DOUBLE,
        source_agent_id STRING,
        created_at      TIMESTAMP,
        updated_at      TIMESTAMP,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Skill(
        id              STRING,
        name            STRING,
        domain          STRING,
        description     STRING,
        code_snippet    STRING,
        usage_count     INT64,
        success_count   INT64,
        created_at      TIMESTAMP,
        updated_at      TIMESTAMP,
        source_agent_id STRING,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Reflection(
        id              STRING,
        agent_id        STRING,
        is_novel        BOOLEAN,
        novelty_reason  STRING,
        critique        STRING,
        lesson_learned  STRING,
        suggested_skill_name STRING,
        created_at      TIMESTAMP,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS SchemaMeta(
        key STRING,
        value STRING,
        PRIMARY KEY(key)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS SiteAdapter(
        id STRING,
        name STRING,
        domain STRING,
        action_intent STRING,
        code_template STRING,
        parameter_schema STRING,
        dom_fingerprint STRING,
        success_count INT64,
        failure_count INT64,
        last_success_at TIMESTAMP,
        status STRING,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS DOMFingerprint(
        id STRING,
        domain STRING,
        url_pattern STRING,
        structural_hash STRING,
        primary_selectors STRING,
        created_at TIMESTAMP,
        PRIMARY KEY(id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS AntiPattern(
        id STRING,
        domain STRING,
        failed_selector STRING,
        trap_type STRING,
        mitigation_hint STRING,
        occurrence_count INT64,
        last_seen_at TIMESTAMP,
        created_at TIMESTAMP,
        PRIMARY KEY(id)
    )
    """,
]

REL_TABLES = [
    "CREATE REL TABLE IF NOT EXISTS SPAWNED(FROM Agent TO Tab)",
    "CREATE REL TABLE IF NOT EXISTS EXECUTED(FROM Agent TO ToolCall)",
    "CREATE REL TABLE IF NOT EXISTS NAVIGATED_TO(FROM ToolCall TO Page)",
    "CREATE REL TABLE IF NOT EXISTS EXTRACTED(FROM ToolCall TO Entity)",
    "CREATE REL TABLE IF NOT EXISTS FOUND_ON(FROM Entity TO Page)",
    "CREATE REL TABLE IF NOT EXISTS DEPENDS_ON(FROM Agent TO Agent)",
    "CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(FROM Entity TO Entity)",
    "CREATE REL TABLE IF NOT EXISTS CREATED_ARTIFACT(FROM Agent TO Artifact)",
    "CREATE REL TABLE IF NOT EXISTS FOR(FROM Episode TO Agent)",
    "CREATE REL TABLE IF NOT EXISTS REFLECTED_ON(FROM Reflection TO Agent)",
    "CREATE REL TABLE IF NOT EXISTS DISTILLED_FROM(FROM Skill TO Agent)",
    "CREATE REL TABLE IF NOT EXISTS LEARNED_FROM(FROM UserMemory TO Agent)",
    "CREATE REL TABLE IF NOT EXISTS COMPILED_FROM(FROM SiteAdapter TO Episode)",
    "CREATE REL TABLE IF NOT EXISTS MATCHES_DOM(FROM SiteAdapter TO DOMFingerprint)",
    "CREATE REL TABLE IF NOT EXISTS HAS_ANTIPATTERN(FROM Page TO AntiPattern)",
]


def _safe_add_column(conn, table: str, col: str, col_type: str, default_val: str = "''") -> None:
    """Safely adds a column to an existing table, ignoring only already-existing property errors."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD {col} {col_type} DEFAULT {default_val}")
    except RuntimeError as re:
        err_msg = str(re).lower()
        if "already has property" in err_msg or "already exists" in err_msg:
            return
        raise


def run_migrations(conn) -> None:
    """Execute DDL and set schema version. Idempotent."""
    for ddl in NODE_TABLES:
        conn.execute(ddl)
    for ddl in REL_TABLES:
        conn.execute(ddl)

    # Safe additive column migrations for existing databases
    _safe_add_column(conn, "Artifact", "source_url", "STRING", "''")
    _safe_add_column(conn, "Artifact", "source_run_id", "STRING", "''")
    _safe_add_column(conn, "Artifact", "source_task_id", "STRING", "''")

    # Upsert schema version
    conn.execute(
        "MERGE (s:SchemaMeta {key: $key}) "
        "ON MATCH SET s.value = $val "
        "ON CREATE SET s.value = $val",
        {"key": "schema_version", "val": SCHEMA_VERSION}
    )

    # Create FTS indices (catch exception if they already exist)
    fts_indices = [
        ("Agent", "Agent_FTS", "['task']"),
        ("ToolCall", "ToolCall_FTS", "['tool_name', 'args_json', 'result_summary']"),
        ("Entity", "Entity_FTS", "['type', 'name', 'attrs']"),
        ("Artifact", "Artifact_FTS", "['title', 'content', 'artifact_type']"),
        ("UserMemory", "UserMemory_FTS", "['category', 'key', 'value']"),
        ("Skill", "Skill_FTS", "['name', 'domain', 'description', 'code_snippet']"),
        ("Reflection", "Reflection_FTS", "['novelty_reason', 'critique', 'lesson_learned']"),
        ("SiteAdapter", "SiteAdapter_FTS", "['name', 'domain', 'action_intent', 'code_template']"),
        ("AntiPattern", "AntiPattern_FTS", "['domain', 'failed_selector', 'trap_type', 'mitigation_hint']")
    ]
    for table_name, index_name, properties in fts_indices:
        try:
            conn.execute(f"CALL CREATE_FTS_INDEX('{table_name}', '{index_name}', {properties})")
        except Exception:
            pass  # Index already exists or FTS not supported on this Kuzu version

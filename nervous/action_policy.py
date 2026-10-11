from __future__ import annotations

import ast
import ipaddress
import logging
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("comex.nervous.action_policy")

PROHIBITED_FOREGROUND_ALIASES: Set[str] = {
    "active_tab",
    "foreground",
    "active",
    "current",
    "activetab",
    "default",
    "default_tab",
}

# Strict finite lease constraints
DEFAULT_LEASE_TTL: float = 300.0   # 5 minutes
MAX_LEASE_TTL: float = 1800.0      # 30 minutes

# Ports considered sensitive internal infrastructure
SENSITIVE_INTERNAL_PORTS: Set[int] = {
    9222,   # Chrome DevTools Protocol remote debugging
    8000,   # Comex Agent Service REST API
    20128,  # 9Router local LLM proxy
}

# Prohibited Playwright / CDP mutation methods in harness scripts
FORBIDDEN_MUTATION_METHODS: Set[str] = {
    "click",
    "dblclick",
    "tripleclick",
    "press",
    "down",
    "up",
    "type",
    "fill",
    "check",
    "uncheck",
    "select_option",
    "drag_to",
    "drop",
    "hover",
    "dispatchmouseevent",
    "dispatchkeyevent",
    "inserttext",
}

# Prohibited builtins, reflection, and dangerous execution functions
FORBIDDEN_META_FUNCTIONS: Set[str] = {
    "exec",
    "eval",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "open",
    "input",
    "importlib",
    "__builtins__",
}

# Dangerous libraries for unisolated script execution
FORBIDDEN_MODULES: Set[str] = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "pathlib",
    "importlib",
    "ctypes",
}

# Prohibited in-page JavaScript mutation tokens in evaluate() calls
FORBIDDEN_JS_MUTATION_SNIPPETS = [
    r"\.click\s*\(",
    r"\[\s*['\"]click['\"]\s*\]\s*\(",
    r"\.submit\s*\(",
    r"\[\s*['\"]submit['\"]\s*\]\s*\(",
    r"\.dispatchevent\s*\(",
    r"\[\s*['\"]dispatchevent['\"]\s*\]\s*\(",
    r"\.remove\s*\(",
    r"\.removechild\s*\(",
    r"innerhtml\s*=",
    r"outerhtml\s*=",
    r"value\s*=",
]


@dataclass
class TabLease:
    worker_id: str
    tab_id: str
    canonical_target_id: Optional[str] = None
    cdp_ws_url: Optional[str] = None
    created_at_mono: float = field(default_factory=time.monotonic)
    expires_at_mono: float = field(default_factory=lambda: time.monotonic() + DEFAULT_LEASE_TTL)
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + DEFAULT_LEASE_TTL)
    is_active: bool = True
    lease_token: str = field(default_factory=lambda: secrets.token_hex(16))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at_mono


class ServerTabLeaseRegistry:
    """
    Authoritative server-owned tab-lease and worker execution context registry.
    Ensures that caller-provided workerId and tabId cannot forge authority.
    Leases must be explicitly granted by an authenticated orchestrator/issuer,
    enforce finite TTLs, and guarantee exclusive ownership of target tabs.
    """

    def __init__(self) -> None:
        self._leases: Dict[str, TabLease] = {}

    def acquire_lease(
        self,
        worker_id: str,
        tab_id: str,
        cdp_ws_url: Optional[str] = None,
        canonical_target_id: Optional[str] = None,
        ttl_seconds: Optional[float] = DEFAULT_LEASE_TTL,
        trusted_issuer: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TabLease:
        """
        Explicitly issue a bounded tab lease to a worker.
        Only trusted orchestrators/issuers can grant leases.
        Enforces exclusive tab ownership and strict TTL bounding.
        """
        if not trusted_issuer:
            raise PermissionError(
                "Unauthorized: tab leases can only be issued by an authenticated trusted orchestrator/issuer."
            )
        if not worker_id or not isinstance(worker_id, str):
            raise ValueError("worker_id must be a non-empty string.")
        if not tab_id or not isinstance(tab_id, str):
            raise ValueError("tab_id must be a non-empty string.")

        worker_clean = worker_id.strip()
        tab_clean = tab_id.strip()
        norm_tab = tab_clean.lower()

        if norm_tab in PROHIBITED_FOREGROUND_ALIASES:
            raise PermissionError(
                f"Cannot lease foreground/active tab alias '{tab_id}' to worker '{worker_clean}'. "
                "Workers must be assigned dedicated, isolated tabs."
            )

        # 1. Enforce exclusive tab ownership: no two active workers can hold the same tab
        now_mono = time.monotonic()
        now_wall = time.time()
        for other_w, other_lease in list(self._leases.items()):
            if other_lease.is_active and not other_lease.is_expired():
                if other_lease.tab_id == tab_clean and other_w != worker_clean:
                    raise PermissionError(
                        f"Tab '{tab_clean}' is already exclusively leased to worker '{other_w}' "
                        f"until {other_lease.expires_at:.1f}. Concurrent tab leasing is prohibited."
                    )
            elif other_lease.is_expired() or not other_lease.is_active:
                # Clean up stale lease
                self._leases.pop(other_w, None)

        # 2. Enforce worker single-lease constraint: a worker cannot hold conflicting leases
        existing_worker_lease = self._leases.get(worker_clean)
        if existing_worker_lease and existing_worker_lease.is_active and not existing_worker_lease.is_expired():
            if existing_worker_lease.tab_id != tab_clean:
                raise PermissionError(
                    f"Worker '{worker_clean}' already holds an active lease for tab '{existing_worker_lease.tab_id}'. "
                    "A worker cannot hold multiple conflicting tab leases simultaneously."
                )

        # 3. Calculate and clamp finite TTL
        raw_ttl = DEFAULT_LEASE_TTL if ttl_seconds is None else float(ttl_seconds)
        clamped_ttl = min(max(raw_ttl, 0.001), MAX_LEASE_TTL)
        expires_mono = now_mono + clamped_ttl
        expires_wall = now_wall + clamped_ttl

        token = secrets.token_hex(16)
        lease = TabLease(
            worker_id=worker_clean,
            tab_id=tab_clean,
            canonical_target_id=canonical_target_id.strip() if canonical_target_id else None,
            cdp_ws_url=cdp_ws_url.strip() if cdp_ws_url else None,
            created_at_mono=now_mono,
            expires_at_mono=expires_mono,
            created_at=now_wall,
            expires_at=expires_wall,
            is_active=True,
            lease_token=token,
            metadata=metadata or {},
        )
        self._leases[worker_clean] = lease
        logger.info(
            f"Granted server lease: worker={worker_clean} -> tab={tab_clean} "
            f"(canonical={canonical_target_id}, ttl={clamped_ttl:.1f}s)"
        )
        return lease

    def revoke_lease(
        self,
        worker_id: str,
        tab_id: Optional[str] = None,
        trusted_caller: bool = False,
    ) -> bool:
        """
        Explicitly revoke an active lease. Requires trusted caller authorization.
        Once revoked, the lease cannot be reused or resurrected.
        """
        if not trusted_caller:
            raise PermissionError("Unauthorized: lease revocation requires trusted caller authorization.")

        worker_key = worker_id.strip() if worker_id else ""
        lease = self._leases.get(worker_key)
        if lease:
            if tab_id and lease.tab_id != tab_id.strip():
                return False
            lease.is_active = False
            self._leases.pop(worker_key, None)
            logger.info(f"Revoked tab lease for worker={worker_id} (tab={lease.tab_id})")
            return True
        return False

    def renew_lease(
        self,
        worker_id: str,
        ttl_seconds: Optional[float] = DEFAULT_LEASE_TTL,
        trusted_issuer: bool = False,
    ) -> TabLease:
        """
        Explicitly renew an active lease. Requires trusted issuer authority.
        Fails closed on missing or expired lease.
        """
        if not trusted_issuer:
            raise PermissionError("Unauthorized: lease renewal requires trusted issuer authority.")

        lease = self.get_active_lease(worker_id)
        if not lease:
            raise PermissionError(f"Cannot renew lease: no active lease found for worker '{worker_id}'.")

        clamped_ttl = min(max(ttl_seconds or DEFAULT_LEASE_TTL, 5.0), MAX_LEASE_TTL)
        now_mono = time.monotonic()
        now_wall = time.time()
        lease.expires_at_mono = now_mono + clamped_ttl
        lease.expires_at = now_wall + clamped_ttl
        logger.info(f"Renewed lease for worker '{worker_id}' (tab={lease.tab_id}, ttl={clamped_ttl:.1f}s)")
        return lease

    def get_active_lease(self, worker_id: str) -> Optional[TabLease]:
        """Retrieve an active, non-expired lease for a worker. Expired leases fail closed."""
        if not worker_id:
            return None
        worker_key = worker_id.strip()
        lease = self._leases.get(worker_key)
        if not lease or not lease.is_active:
            return None
        if lease.is_expired():
            lease.is_active = False
            self._leases.pop(worker_key, None)
            logger.warning(f"Lease for worker '{worker_key}' expired and has been purged.")
            return None
        return lease

    def validate_worker_access(
        self,
        worker_id: Optional[str],
        target_tab: str,
        trusted_app_context: bool = False,
        caller_ws_url: Optional[str] = None,
        expected_target_id: Optional[str] = None,
        worker_token: Optional[str] = None,
    ) -> TabLease:
        """
        Authoritatively validate that the given worker is leased the requested target tab.
        Fails closed on missing worker context, missing lease, expired lease, or mismatched tab.
        Enforces caller authorization: caller must possess trusted app context or the worker's unforgeable lease token.
        NEVER creates, refreshes, or auto-grants leases.
        """
        norm_target = (target_tab or "").strip().lower()

        # 1. Prohibited foreground aliases for workers
        if not norm_target or norm_target in PROHIBITED_FOREGROUND_ALIASES:
            if worker_id:
                raise PermissionError(
                    f"Worker '{worker_id}' is prohibited from accessing foreground/active tab alias '{target_tab}'. "
                    "Workers must operate strictly on their explicitly leased tab."
                )
            if not trusted_app_context:
                raise PermissionError(
                    f"Untrusted access to foreground tab '{target_tab}' is prohibited. "
                    "Trusted application context required."
                )
            # Trusted internal single-agent execution on foreground tab
            return TabLease(
                worker_id="__trusted_single_agent__",
                tab_id="active_tab",
                cdp_ws_url=caller_ws_url,
                created_at_mono=time.monotonic(),
                expires_at_mono=time.monotonic() + DEFAULT_LEASE_TTL,
                created_at=time.time(),
                expires_at=time.time() + DEFAULT_LEASE_TTL,
                is_active=True,
            )

        # 2. Requests without worker ID targeting a specific tab
        if not worker_id:
            if not trusted_app_context:
                raise PermissionError(
                    f"Missing worker authorization context: unauthenticated requests cannot access tab '{target_tab}' "
                    "without trusted application context."
                )
            return TabLease(
                worker_id="__trusted_single_agent__",
                tab_id=target_tab.strip(),
                cdp_ws_url=caller_ws_url,
                created_at_mono=time.monotonic(),
                expires_at_mono=time.monotonic() + DEFAULT_LEASE_TTL,
                created_at=time.time(),
                expires_at=time.time() + DEFAULT_LEASE_TTL,
                is_active=True,
            )

        # 3. Worker requests: must have an active server-registered lease.
        # Authorization ONLY validates existing leases; it NEVER auto-creates them.
        lease = self.get_active_lease(worker_id)
        if not lease:
            raise PermissionError(
                f"No active server lease found for worker '{worker_id}'. "
                "Workers must be issued a valid server-owned tab lease by the trusted orchestrator."
            )

        # Worker caller must be authorized: possession of the unforgeable lease token is MANDATORY.
        # Trusted application context alone does NOT permit impersonating an existing worker.
        if not worker_token or worker_token.strip() != lease.lease_token.strip():
            raise PermissionError(
                f"Unauthorized caller cannot act as worker '{worker_id}'. "
                "Valid unforgeable worker lease token required."
            )

        if lease.tab_id != target_tab.strip():
            raise PermissionError(
                f"Worker authorization mismatch: worker '{worker_id}' is leased tab '{lease.tab_id}', "
                f"not target tab '{target_tab}'."
            )

        # 4. Strict canonical target identity verification if provided
        if expected_target_id and lease.canonical_target_id:
            if expected_target_id.strip() != lease.canonical_target_id.strip():
                raise PermissionError(
                    f"Canonical target mismatch: tab '{lease.tab_id}' is bound to Chromium target "
                    f"'{lease.canonical_target_id}', but resolved target is '{expected_target_id}'."
                )

        # 5. Strict WebSocket target verification if caller supplied one
        if caller_ws_url and lease.cdp_ws_url:
            if caller_ws_url.strip() != lease.cdp_ws_url.strip():
                raise PermissionError(
                    f"Target WebSocket mismatch: requested WebSocket URL does not match "
                    f"server-verified CDP endpoint for leased tab '{lease.tab_id}'."
                )

        return lease

    def reset(self) -> None:
        """Clear all active leases (for test isolation)."""
        self._leases.clear()


class ServerTabTargetRegistry:
    """
    Server-owned mapping from application tab ID to Chromium canonical target ID and WebSocket URL.
    Eliminates fragile URL substring matching and ensures operations bind strictly to verified targets.
    """

    def __init__(self) -> None:
        self._targets: Dict[str, Tuple[str, str]] = {}  # tab_id -> (canonical_target_id, ws_url)

    def register_tab_target(self, tab_id: str, canonical_target_id: str, ws_url: str) -> None:
        if not tab_id or not canonical_target_id or not ws_url:
            return
        self._targets[tab_id.strip()] = (canonical_target_id.strip(), ws_url.strip())
        logger.info(f"Registered tab target: {tab_id} -> {canonical_target_id} ({ws_url})")

    def lookup_tab_target(self, tab_id: str) -> Optional[Tuple[str, str]]:
        if not tab_id:
            return None
        return self._targets.get(tab_id.strip())

    def unregister_tab_target(self, tab_id: str) -> None:
        if tab_id:
            self._targets.pop(tab_id.strip(), None)

    def reset(self) -> None:
        self._targets.clear()


GLOBAL_LEASE_REGISTRY = ServerTabLeaseRegistry()
GLOBAL_TARGET_REGISTRY = ServerTabTargetRegistry()


def get_lease_registry() -> ServerTabLeaseRegistry:
    return GLOBAL_LEASE_REGISTRY


def get_target_registry() -> ServerTabTargetRegistry:
    return GLOBAL_TARGET_REGISTRY


def validate_worker_tab_authorization(
    worker_id: Optional[str],
    target_tab: str,
    trusted_app_context: bool = False,
    caller_ws_url: Optional[str] = None,
    expected_target_id: Optional[str] = None,
    assigned_tab: Optional[str] = None,  # Retained for signature compatibility; NEVER auto-creates lease
    worker_token: Optional[str] = None,
) -> TabLease:
    """
    Authoritatively enforce server-level worker-to-tab authorization.
    Fail closed on missing context, missing lease, expired lease, or mismatched tab.
    Never permits worker fallback to foreground aliases ('active_tab', 'foreground').
    Preserves single-agent foreground access only via explicit trusted_app_context.
    DOES NOT automatically grant, create, or renew leases under any circumstances.
    """
    # If caller provided assigned_tab and worker_id, verify they match before checking registry
    if worker_id and assigned_tab and target_tab.strip() != assigned_tab.strip():
        raise PermissionError(
            f"Worker authorization mismatch: worker '{worker_id}' assigned '{assigned_tab}', "
            f"cannot access '{target_tab}'."
        )

    return GLOBAL_LEASE_REGISTRY.validate_worker_access(
        worker_id=worker_id,
        target_tab=target_tab,
        trusted_app_context=trusted_app_context,
        caller_ws_url=caller_ws_url,
        expected_target_id=expected_target_id,
        worker_token=worker_token,
    )


def validate_navigation_url(
    url: str,
    allow_localhost: bool = True,
    allow_private_network: bool = True,
) -> str:
    """
    Validate that navigation targets a permitted web scheme (http/https)
    with a valid, non-empty network location and hostname.
    Rejects:
    - Non-HTTP(S) schemes (javascript:, file:, data:, about:, chrome:)
    - Missing-hostname URLs (e.g. 'http:///example.com')
    - Embedded user credentials (e.g. 'http://user:pass@example.com')
    - Malformed IP literals (e.g. 'http://999.999.999.999')
    - Navigation to Comex internal infrastructure ports (9222 CDP, 8000 API, 20128 LLM proxy)
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    cleaned = url.strip()
    if not cleaned:
        raise ValueError("URL must be a non-empty string.")

    try:
        parsed = urlparse(cleaned)
    except Exception as e:
        raise ValueError(f"Malformed URL '{cleaned}': {e}")

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Disallowed navigation scheme '{scheme}'. Only http:// and https:// URLs are permitted."
        )

    # 1. Prohibit embedded user credentials (user:pass@host)
    if parsed.username or parsed.password:
        raise ValueError(f"URLs containing user credentials are prohibited by security policy: '{cleaned}'.")

    # 2. Validate non-empty network location
    netloc = parsed.netloc
    if not netloc:
        raise ValueError(f"Malformed URL '{cleaned}': Missing network location (hostname).")

    if netloc.startswith("/") or "/" in netloc.split(":")[0]:
        raise ValueError(f"Malformed URL '{cleaned}': Invalid hostname format.")

    try:
        hostname = parsed.hostname
    except ValueError as he:
        raise ValueError(f"Malformed hostname in URL '{cleaned}': {he}")

    if not hostname:
        raise ValueError(f"Malformed URL '{cleaned}': Missing valid hostname.")

    if any(c in hostname for c in " \t\r\n\\;<>\"'@"):
        raise ValueError(f"Malformed hostname in URL '{cleaned}': contains illegal characters.")

    # 3. Determine port
    port = parsed.port or (80 if scheme == "http" else 443)

    # 4. Check if hostname is an IP literal or IP-like string
    is_ip_literal = False
    clean_host = hostname.strip("[]")
    # Check if host contains only digits, dots, colons (IPv4/IPv6 candidates)
    if re.match(r"^[\d\.:a-fA-F]+$", clean_host) and ("." in clean_host or ":" in clean_host):
        is_ip_literal = True
        try:
            ip_obj = ipaddress.ip_address(clean_host)
        except ValueError:
            raise ValueError(f"Malformed IP literal in URL '{cleaned}': '{hostname}' is not a valid IP address.")

        # Block internal infrastructure ports on loopback or private network
        if (ip_obj.is_loopback or ip_obj.is_private) and port in SENSITIVE_INTERNAL_PORTS:
            raise PermissionError(
                f"Navigation to Comex internal infrastructure port {port} on {hostname} is strictly prohibited."
            )

        if ip_obj.is_loopback and not allow_localhost:
            raise PermissionError(f"Loopback navigation to '{hostname}' is prohibited by policy.")

        if ip_obj.is_private and not allow_private_network:
            raise PermissionError(f"Private network navigation to '{hostname}' is prohibited by policy.")

    # 5. Check named loopback hosts (localhost, *.localhost)
    is_named_loopback = hostname in ("localhost", "127.0.0.1", "::1") or hostname.endswith(".localhost")
    if is_named_loopback:
        if port in SENSITIVE_INTERNAL_PORTS:
            raise PermissionError(
                f"Navigation to Comex internal infrastructure port {port} on {hostname} is strictly prohibited."
            )
        if not allow_localhost:
            raise PermissionError(f"Loopback/localhost navigation is prohibited by policy: '{hostname}'.")

    return cleaned


def validate_harness_script_safety(code: str) -> None:
    """
    Inspects Python code intended for browser-harness via AST analysis.
    Raises PermissionError if the script attempts direct mutations, dynamic code execution,
    reflection bypasses (exec, eval, getattr, __builtins__), or unmediated DOM mutations.
    """
    if not code or not isinstance(code, str):
        return

    try:
        tree = ast.parse(code)
    except SyntaxError as se:
        raise ValueError(f"Syntax error in harness script: {se}")

    for node in ast.walk(tree):
        # 1. Prohibit dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0].lower()
                if root_mod in FORBIDDEN_MODULES:
                    raise PermissionError(
                        f"Import of unisolated module '{alias.name}' is prohibited by action policy."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0].lower()
                if root_mod in FORBIDDEN_MODULES:
                    raise PermissionError(
                        f"Import from unisolated module '{node.module}' is prohibited by action policy."
                    )

        # 2. Prohibit access to __builtins__
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            raise PermissionError("Direct access to '__builtins__' is prohibited by Comex action policy.")

        # 3. Prohibit meta-programming / dynamic execution builtins
        if isinstance(node, ast.Call):
            func = node.func
            eval_target = False
            if isinstance(func, ast.Name):
                func_name = func.id.lower()
                if func_name in FORBIDDEN_META_FUNCTIONS:
                    raise PermissionError(
                        f"Dynamic execution builtin '{func.id}' is prohibited by Comex action policy."
                    )
                if func_name == "click_at_xy":
                    raise PermissionError(
                        "Unmediated coordinate clicks (click_at_xy) are prohibited by Comex action policy. "
                        "Use the authoritative browser_click tool instead."
                    )
                if func_name in ("evaluate", "evaluate_handle", "js", "eval_js"):
                    eval_target = True
            elif isinstance(func, ast.Attribute):
                attr_name = func.attr.lower()
                if attr_name == "click_element":
                    continue
                if attr_name in FORBIDDEN_MUTATION_METHODS:
                    raise PermissionError(
                        f"Unmediated browser mutation '{func.attr}' is prohibited in harness scripts. "
                        "All mutations must be executed through approved action tools "
                        "(browser_click, browser_type, browser_navigate)."
                    )
                if attr_name in FORBIDDEN_META_FUNCTIONS:
                    raise PermissionError(
                        f"Dynamic attribute operation '{func.attr}' is prohibited in harness scripts."
                    )
                if attr_name in ("evaluate", "evaluate_handle", "js", "eval_js"):
                    eval_target = True

            if eval_target:
                for arg in node.args:
                    if isinstance(arg, ast.BinOp):
                        raise PermissionError(
                            "Dynamic string expressions/concatenation in evaluate() are prohibited by action policy."
                        )
                    if isinstance(arg, ast.JoinedStr):
                        raise PermissionError(
                            "Formatted f-strings in evaluate() are prohibited by action policy."
                        )
                    if isinstance(arg, ast.Call):
                        raise PermissionError(
                            "Function calls inside evaluate() arguments are prohibited by action policy."
                        )
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        js_code = arg.value.lower()
                        for pattern in FORBIDDEN_JS_MUTATION_SNIPPETS:
                            if re.search(pattern, js_code, re.IGNORECASE):
                                raise PermissionError(
                                    f"Unmediated JavaScript DOM mutation matching pattern '{pattern}' "
                                    "is prohibited in evaluate() scripts. "
                                    "Use authoritative action tools (browser_click, browser_type) instead."
                                )

        # 4. Prohibit subscript access on builtins or reflection dicts (e.g. __builtins__["exec"])
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in ("__builtins__", "globals", "locals"):
                raise PermissionError(
                    f"Subscript access on '{node.value.id}' is prohibited by Comex action policy."
                )

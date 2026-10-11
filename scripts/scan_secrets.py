#!/usr/bin/env python3
"""
Comex Secret Scanner
Scans working tree candidate files and Git commit history for credentials,
API tokens, private keys, auth headers, and high-entropy secrets.

Strict security invariant: Never prints raw secret values in output.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

# Known false positive / benign placeholder patterns
BENIGN_PLACEHOLDERS = [
    re.compile(r"your-.*-here", re.I),
    re.compile(r"your-randomly-generated-secret-token-here", re.I),
    re.compile(r"your-9router-api-key-here", re.I),
    re.compile(r"test_token", re.I),
    re.compile(r"dummy_token", re.I),
    re.compile(r"mock_token", re.I),
    re.compile(r"mock-bearer-token", re.I),
    re.compile(r"example\.com", re.I),
    re.compile(r"Bearer\s+test", re.I),
    re.compile(r"Bearer\s+\${", re.I),
    re.compile(r"Authorization:\s*Bearer\s+<token>", re.I),
    re.compile(r"<placeholder>", re.I),
    re.compile(r"gho_\*{20,}", re.I),  # Already redacted in docs
]

SECRET_RULES = [
    ("OpenAI API Key", re.compile(r"\b(sk-[a-zA-Z0-9_\-]{20,})\b")),
    ("Anthropic API Key", re.compile(r"\b(sk-ant-[a-zA-Z0-9_\-]{20,})\b")),
    ("Google AI / GCP API Key", re.compile(r"\b(AIza[0-9A-Za-z\-_]{35})\b")),
    ("GitHub Personal Access Token", re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{36,})\b")),
    ("GitHub Fine-Grained PAT", re.compile(r"\b(github_pat_[A-Za-z0-9_]{82})\b")),
    ("AWS Access Key ID", re.compile(r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b")),
    ("Private Key Block", re.compile(r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----")),
    ("Slack Token", re.compile(r"\b(xox[baprs]-[0-9a-zA-Z]{10,})\b")),
    ("Generic Auth Token Assignment", re.compile(r"""(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)\s*[:=]\s*['"]([a-zA-Z0-9_\-]{16,})['"]""", re.I)),
    ("Hardcoded Bearer Token Header", re.compile(r"""['"]?(?:Authorization|proxy-authorization)['"]?\s*:\s*['"]Bearer\s+([a-zA-Z0-9_\-\.]{16,})['"]""", re.I)),
    ("Internal App Token Assignment", re.compile(r"""COMEX_INTERNAL_APP_TOKEN\s*=\s*['"]?([a-zA-Z0-9_\-]{16,})['"]?""", re.I)),
]

def redact(val: str) -> str:
    if len(val) <= 6:
        return "***"
    return val[:3] + "*" * (len(val) - 6) + val[-3:]

def is_benign(match_str: str) -> bool:
    for bp in BENIGN_PLACEHOLDERS:
        if bp.search(match_str):
            return True
    return False

def scan_text(text: str, filename: str) -> list:
    findings = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, 1):
        for rule_name, pattern in SECRET_RULES:
            for match in pattern.finditer(line):
                matched_str = match.group(0)
                secret_part = match.group(1) if match.groups() else matched_str
                if is_benign(secret_part) or is_benign(matched_str):
                    continue
                # Specific check: exclude .env.example placeholders
                if "your-" in secret_part.lower():
                    continue
                findings.append({
                    "file": filename,
                    "line": idx,
                    "rule": rule_name,
                    "redacted": redact(secret_part),
                })
    return findings

def get_candidate_files() -> list:
    """Get all tracked + modified + untracked files (excluding gitignored)."""
    cmd = ["git", "status", "--porcelain", "-uall"]
    out = subprocess.check_output(cmd, text=True)
    files = set()
    for line in out.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[1].strip()
        files.add(path)
    
    # Also add all tracked files
    tracked = subprocess.check_output(["git", "ls-files"], text=True)
    for path in tracked.splitlines():
        if path.strip():
            files.add(path.strip())
            
    # Filter out directories or non-existent files
    valid_files = []
    for f in sorted(files):
        p = Path(f)
        if p.is_file():
            # Skip binary extensions
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pptx", ".pdf", ".pyc"]:
                continue
            valid_files.append(f)
    return valid_files

def scan_git_history() -> list:
    """Scans all commit diffs in git history."""
    findings = []
    cmd = ["git", "log", "-p", "--all"]
    try:
        out = subprocess.check_output(cmd, text=True)
        # Scan log output
        current_commit = "unknown"
        current_file = "unknown"
        for line in out.splitlines():
            if line.startswith("commit "):
                current_commit = line.split()[1][:8]
            elif line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3].lstrip("b/")
            elif line.startswith("+") and not line.startswith("+++"):
                content = line[1:]
                for rule_name, pattern in SECRET_RULES:
                    for match in pattern.finditer(content):
                        matched_str = match.group(0)
                        secret_part = match.group(1) if match.groups() else matched_str
                        if is_benign(secret_part) or is_benign(matched_str):
                            continue
                        if "your-" in secret_part.lower():
                            continue
                        findings.append({
                            "commit": current_commit,
                            "file": current_file,
                            "rule": rule_name,
                            "redacted": redact(secret_part),
                        })
    except Exception as e:
        print(f"Error reading git log: {e}", file=sys.stderr)
    return findings

def main():
    root = Path.cwd()
    print(f"Starting Secret Scan in: {root}")
    
    candidate_files = get_candidate_files()
    print(f"Candidate files to scan: {len(candidate_files)}")
    
    tree_findings = []
    for f in candidate_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            f_findings = scan_text(text, f)
            tree_findings.extend(f_findings)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}", file=sys.stderr)
            
    print(f"Scanning Git commit history...")
    history_findings = scan_git_history()
    
    total = len(tree_findings) + len(history_findings)
    print("\n" + "="*50)
    print("SECRET SCAN RESULTS SUMMARY")
    print("="*50)
    
    if tree_findings:
        print(f"\n[!] Detected {len(tree_findings)} potential secret(s) in working tree files:")
        for finding in tree_findings:
            print(f"  - File: {finding['file']}:{finding['line']}")
            print(f"    Type: {finding['rule']}")
            print(f"    Match (redacted): {finding['redacted']}")
    else:
        print("\n[✓] Working tree candidate files: ZERO live secrets detected.")
        
    if history_findings:
        print(f"\n[!] Detected {len(history_findings)} potential secret(s) in Git history:")
        for finding in history_findings:
            print(f"  - Commit: {finding['commit']} | File: {finding['file']}")
            print(f"    Type: {finding['rule']}")
            print(f"    Match (redacted): {finding['redacted']}")
    else:
        print("[✓] Existing Git history: ZERO live secrets detected.")
        
    print("="*50)
    if total > 0:
        print(f"Total potential secrets flagged: {total}")
        sys.exit(1)
    else:
        print("RESULT: SCAN CLEAN. Ready for safe Git versioning.")
        sys.exit(0)

if __name__ == "__main__":
    main()

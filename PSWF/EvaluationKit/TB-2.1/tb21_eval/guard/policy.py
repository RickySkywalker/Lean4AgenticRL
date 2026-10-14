#!/usr/bin/env python3
"""Shared domain, address, and command policy for the evaluation guard.

Domain denials take precedence over the allowlist. DNS answers must be public
and must pass configured address banlists. Command checks cover Tor, proxy
harvesting, external scanning, and access to protected host services. Pattern
matching is combined with limited Python constant analysis; it is a policy
filter, not a general shell or Python interpreter. Policy digests identify the
actual rule files, including their current bytes."""

from __future__ import annotations

import ast
import hashlib
import ipaddress
import os
import re
import shlex
import socket
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

def proc_starttime_ticks(pid: int | None = None) -> int:
    """Read the Linux process start time used to detect PID reuse."""
    target = os.getpid() if pid is None else pid
    if target <= 1:
        raise ValueError(f"unsafe PID {target}")
    raw = Path(f"/proc/{target}/stat").read_text(encoding="utf-8")
    close = raw.rfind(") ")
    fields = raw[close + 2 :].split() if close >= 0 else []
    if len(fields) < 20:
        raise RuntimeError(f"malformed /proc stat for PID {target}")
    value = int(fields[19])
    if value <= 0:
        raise RuntimeError(f"invalid process starttime for PID {target}")
    return value

def write_pid_identity(path: Path, pid: int | None = None) -> None:
    """Atomically write a process ID and its start time."""
    target = os.getpid() if pid is None else pid
    starttime = proc_starttime_ticks(target)
    temporary = path.with_suffix(path.suffix + f".tmp.{target}")
    temporary.write_text(f"{target}\n{starttime}\n", encoding="utf-8")
    temporary.replace(path)

def normalize_host(raw: str) -> str:

    """Canonicalize a DNS name or IP literal and reject invalid host syntax."""
    host = raw.strip().rstrip(".").lower()
    if not host or len(host) > 253 or "\x00" in host:
        raise ValueError("empty or invalid hostname")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:

        if "%" in host:
            raise ValueError("scoped address is forbidden")
        ipaddress.ip_address(host)
        return host
    except ValueError as exc:
        if "forbidden" in str(exc):
            raise
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("invalid IDNA hostname") from exc
    labels = host.split(".")
    if len(labels) < 2 or any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or not re.fullmatch(r"[a-z0-9-]+", label)
        for label in labels
    ):
        raise ValueError("invalid DNS hostname")
    return host

def is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False

def _load_patterns(path: Path) -> tuple[str, ...]:
    """Load exact hostnames and wildcard subdomains, excluding comments."""
    if not path.is_file():
        raise FileNotFoundError(f"policy file missing: {path}")
    out: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.split("#", 1)[0].strip().lower().rstrip(".")
        if not value:
            continue
        suffix = value[2:] if value.startswith("*.") else value
        if is_ip_literal(suffix):
            raise ValueError(f"{path}:{number}: IP literals are not policy domains")
        normalize_host(suffix)
        out.append(value)
    if not out:
        raise ValueError(f"empty policy file: {path}")
    return tuple(sorted(set(out)))

def _matches(host: str, pattern: str) -> bool:
    """Match exact hosts; a wildcard excludes the suffix hostname itself."""
    if pattern.startswith("*."):
        suffix = pattern[2:]
        return host != suffix and host.endswith("." + suffix)
    return host == pattern

@dataclass(frozen=True)
class DomainPolicy:
    """Immutable domain rules and the digest of the files that define them."""
    allowed: tuple[str, ...]
    denied: tuple[str, ...]
    digest: str

    @classmethod
    def load(cls, allowed_path: Path, denied_path: Path) -> "DomainPolicy":
        allowed = _load_patterns(allowed_path)
        denied = _load_patterns(denied_path)
        payload = allowed_path.read_bytes() + b"\0" + denied_path.read_bytes()
        return cls(allowed, denied, hashlib.sha256(payload).hexdigest())

    def decision(self, raw_host: str) -> tuple[bool, str, str]:
        """Apply syntax validation, IP-literal rejection, denial, then allowance."""
        try:
            host = normalize_host(raw_host)
        except ValueError as exc:
            return False, "invalid_host", str(exc)
        if is_ip_literal(host):
            return False, "ip_literal", host
        if any(_matches(host, pattern) for pattern in self.denied):
            return False, "explicit_deny", host
        if not any(_matches(host, pattern) for pattern in self.allowed):
            return False, "not_allowlisted", host
        return True, "allowed", host

_RESOLVE_CACHE: dict[tuple[str, int], tuple[float, list[tuple[int, tuple]]]] = {}
_RESOLVE_CACHE_TTL = float(os.environ.get("Q38_RESOLVE_CACHE_TTL_SEC", "30"))
_RESOLVE_CACHE_MAX = 512

_IP_BAN_CACHE: dict[str, tuple[tuple[int, int, int], frozenset[str]]] = {}
_IP_BAN_CACHE_LOCK = threading.Lock()

def _reject_banned_addresses(addresses: list[tuple[int, tuple]]) -> None:
    """Reject configured proxy or Tor addresses and fail closed on invalid banlists."""
    state = os.environ.get("Q38_PROXY_LIST_STATE", "")
    if not state and os.environ.get("Q38_GUARD_STATE"):
        state = str(Path(os.environ["Q38_GUARD_STATE"]) / "proxy-banset")
    sources: list[tuple[str, Path, int]] = []
    if state:
        sources.append(("proxy-list", Path(state) / "union.txt", 100))
    tor_file = os.environ.get("Q38_TOR_BAN_FILE", "")
    if tor_file:
        sources.append(("Tor relay", Path(tor_file), 5000))
    for label, path, minimum in sources:
        try:
            stat = path.stat()
            identity = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            with _IP_BAN_CACHE_LOCK:
                cached = _IP_BAN_CACHE.get(str(path))
                if cached is None or cached[0] != identity:
                    banned = frozenset(str(ipaddress.ip_address(line.strip()))
                                       for line in path.read_text().splitlines() if line.strip())
                    if len(banned) < minimum:
                        raise ValueError(f"hollow {label} IP ban file")
                    cached = (identity, banned)
                    _IP_BAN_CACHE[str(path)] = cached
            if any(str(ipaddress.ip_address(sockaddr[0])) in cached[1] for _, sockaddr in addresses):
                raise PermissionError(f"resolved destination is in the {label} ban union")
        except (OSError, ValueError) as exc:
            raise PermissionError(f"{label} IP ban check refused: {exc}") from exc

def public_addresses(host: str, port: int) -> list[tuple[int, tuple]]:

    """Resolve public TCP addresses, rechecking banlists even for cached DNS answers."""
    cache_key = (host, port)
    if _RESOLVE_CACHE_TTL > 0:
        hit = _RESOLVE_CACHE.get(cache_key)
        if hit is not None:
            expires, addresses = hit
            if expires > time.monotonic():
                _reject_banned_addresses(addresses)
                return list(addresses)

            _RESOLVE_CACHE.pop(cache_key, None)

    infos = socket.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    unique: list[tuple[int, tuple]] = []
    seen: set[tuple[int, str, int]] = set()
    unsafe: list[str] = []
    for family, socktype, proto, _canon, sockaddr in infos:
        del socktype, proto
        addr = sockaddr[0]
        ip = ipaddress.ip_address(addr)
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            unsafe.append(str(ip))
            continue
        key = (family, str(ip), port)
        if key not in seen:
            seen.add(key)
            unique.append((family, sockaddr))
    if unsafe:
        raise PermissionError(
            "non-public DNS answer(s): " + ",".join(sorted(set(unsafe)))
        )
    if not unique:
        raise OSError("no public A/AAAA answers")

    ordered = sorted(unique, key=lambda item: (item[0] != socket.AF_INET, item[1][0]))
    _reject_banned_addresses(ordered)

    if _RESOLVE_CACHE_TTL > 0:
        if len(_RESOLVE_CACHE) >= _RESOLVE_CACHE_MAX:
            _RESOLVE_CACHE.clear()
        _RESOLVE_CACHE[cache_key] = (time.monotonic() + _RESOLVE_CACHE_TTL, list(ordered))
    return ordered

class ClientLimiter:

    """Track per-client concurrency, connection bursts, and recent domain counts."""
    def __init__(self, max_active: int, max_new_10s: int, max_domains: int):
        self.max_active = max_active
        self.max_new_10s = max_new_10s
        self.max_domains = max_domains
        self._active: dict[str, int] = defaultdict(int)
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._domains: dict[str, dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()

    def acquire(self, client: str, host: str) -> tuple[bool, str]:
        now = time.monotonic()
        with self._lock:
            recent = self._recent[client]
            while recent and recent[0] < now - 10:
                recent.popleft()
            domains = self._domains[client]
            for old_host, last_seen in list(domains.items()):
                if last_seen < now - 3600:
                    domains.pop(old_host, None)
            if self._active[client] >= self.max_active:
                return False, "active_connection_limit"
            if len(recent) >= self.max_new_10s:
                return False, "new_connection_rate_limit"
            if host not in domains and len(domains) >= self.max_domains:
                return False, "destination_diversity_limit"
            self._active[client] += 1
            recent.append(now)
            domains[host] = now
            return True, "allowed"

    def rate_retry_after(self, client: str) -> float:

        """Return the remaining wait before another connection fits the burst limit."""
        now = time.monotonic()
        with self._lock:
            recent = self._recent[client]
            while recent and recent[0] < now - 10:
                recent.popleft()
            if len(recent) < self.max_new_10s:
                return 0.0
            return max(0.01, recent[0] + 10 - now)

    def release(self, client: str) -> None:
        with self._lock:
            self._active[client] = max(0, self._active[client] - 1)

# Command boundaries include shell separators and common execution wrappers.
_CMD_POS = (
    r"(?:^|\x00|[\n;`\"']|&&|\|\||\||\$\(|\bdo\s+|\bthen\s+|\belse\s+"
    r"|\bxargs\s+(?:\S+\s+){0,4}|\bsudo\s+(?:-\S+\s+)*|\bexec\s+|\bnohup\s+|\bsetsid\s+"
    r"|\bnice\s+(?:-n\s*-?\d+\s+|-\S+\s+)?|\bionice\s+(?:-\S+\s+){0,2}|\bstdbuf\s+(?:-\S+\s+){0,2}"
    r"|\benv\s+(?:[A-Za-z_]\w*=\S*\s+){0,4}|\btimeout\s+[\d.]+[smhd]?\s+)\s*"
)
_PATH_PREFIX = r"(?:\.{0,2}/[\w./+-]{0,80}/|\./)?"

_ACT_CMD_POS = (
    r"(?:^|\x00|[\n;`]|&&|\|\||\||\$\(|\bdo\s+|\bthen\s+|\belse\s+"
    r"|\bsudo\s+(?:-\S+\s+)*|\bexec\s+|\bnohup\s+|\bsetsid\s+"
    r"|\bxargs\s+(?:\S+\s+){0,4}|\btimeout\s+[\d.]+[smhd]?\s+)\s*"
)

_HARVESTER = (
    r"(?:proxybroker2?|free-proxy|proxyscrape|requests-ip-rotator|scrapy-rotating-proxies"
    r"|rotating-free-proxies|proxy-rotator|proxy-list-scraper|mubeng)"
)

_PUBLIC_IP_ONLY = (
    r"(?!(?:127\.|10\.|192\.168\.|169\.254\.|0\.0\.0\.0|172\.(?:1[6-9]|2\d|3[01])\.))"
)

_PUBLIC_IP6_TARGET = (
    r"(?<![\w:.])(?!(?:::|fe80:|f[cd][0-9a-f]{2}:|::ffff:))"
    r"(?:(?:[0-9a-f]{1,4}:){1,6}:(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){0,5})?|(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4})"
    r"(?:/\d{1,3})?(?![\w:])"
)

_PUBLIC_IP6_TEMPLATE = (
    r"(?<![\w:.])[23][0-9a-f]{3}:(?:[0-9a-f]{1,4}:){1,6}:?"
    r"(?:\$\{?\w|\{\d+\.\.\d+\}|%0?\d*[dsx]|\{\}|\{\w{1,16}\})"
)

def _win(n: int, extra: int = 3) -> str:
    """Build a bounded, optionally multiline regex window."""
    return r"[^\n]{0,%d}?(?:\n[^\n]{0,%d}?){0,%d}" % (n, n, extra)

_L_TOR_OPT = (
    r"(?:--(?:SocksPort|ControlPort|ControlSocket|DataDirectory|RunAsDaemon|verify-config"
    r"|defaults-torrc|ClientTransportPlugin|UseBridges|Bridge|HashedControlPassword"
    r"|CookieAuthentication|ExitNodes|EntryNodes|ExcludeNodes|StrictNodes|HTTPTunnelPort"
    r"|DNSPort|TransPort|NATDPort|HiddenServiceDir|HiddenServicePort|GeoIPFile|PidFile"
    r"|SafeLogging|MaxCircuitDirtiness|NewCircuitPeriod|CircuitBuildTimeout)(?![\w-])"
    r"|-f\s+[\w./+~-]{0,80}torrc(?![\w.+-]))"
)

_L_ARGTOK = r"(?:-{1,2}[\w.-]+(?:=[^\s\"']{0,60})?|\d[\w.:%-]*|[/~][\w./+~-]{1,80})"
_L_BARE_TOR = r"(?<![\w.+-])tor(?![\w.+-])"

_L_INIT_VERB = r"(?:invoke-rc\.d|update-rc\.d|initctl|s6-svc|s6-rc|rcctl|svcadm|systemd-run)"
_L_INIT_ACT = r"(?:start|stop|restart|reload|enable|disable|defaults|remove|once|up|down|-u|-d|-r)"
_L_SVC_PATH = r"(?:/(?:run|etc|var)/(?:service|sv)/)?"

_L_SPAWN = (
    r"(?:Popen|execvp?e?|execlp?e?|posix_spawnp?|spawnv?p?e?|spawnSync|execFile(?:Sync)?|exec\.Command"
    r"|Command::new|CreateProcess|ProcessBuilder|\.(?:run|call|system|check_output|check_call|popen))"
)

_WRAPPED_ACTION_POSITION = (
    r"(?:^|\x00|[\n;`]|&&|\|\||\||\$\(|\bdo\s+|\bthen\s+|\belse\s+"
    r"|\bsudo\s+(?:-\S+\s+)*|\bexec\s+|\bnohup\s+|\bsetsid\s+"
    r"|\bxargs\s+(?:\S+\s+){0,4}|\btimeout\s+[\d.]+[smhd]?\s+"
    r"|\b(?:ba|z|da|k)?sh\s+(?:-{1,2}[\w-]+\s+){0,3}['\"]?"
    r"|\b(?:python3?(?:\.\d{1,2})?|perl|ruby|node|nodejs)\s+(?:-{1,2}[\w-]+\s+){0,3}['\"]?"
    r"|[:,]\s*\\?\")\s*"
)

_GIT_CLONE_COMMAND = (
    r"git(?:[ \t]+(?:-C[ \t]+\S{1,60}|-c[ \t]+[\w.]{1,40}=\S{0,40}"
    r"|--(?:git-dir|work-tree|namespace|exec-path)=\S{1,60}"
    r"|--no-pager|--bare|--literal-pathspecs|--no-replace-objects|-p)){0,4}[ \t]+clone"
)

_DEPENDENCY_MANIFEST = (
    r"(?<![\w-])(?:requirements(?:[-_.][\w.]{1,20})?\.(?:txt|in)|requirements/[\w.-]{1,30}\.(?:txt|in)"
    r"|constraints(?:[-_.][\w.]{1,20})?\.txt|Pipfile(?:\.lock)?|pyproject\.toml|setup\.cfg|setup\.py"
    r"|environment\.ya?ml)(?![\w.-])"
)

_IPV4_CIDR_TARGET = (r"(?<![\d.])(?:\d{1,3}(?:\.\d{1,3}){3}/(?:3[0-2]|[12]\d|[89])"
              r"|\d{1,3}(?:\.\d{1,3}){2}/(?:2[0-4]|1[7-9])"
              r"|\d{1,3}\.\d{1,3}/(?:1[0-6]|[89]))(?![\d.])")

_TORPROJ_ACT = (
    _WRAPPED_ACTION_POSITION
    + r"(?:curl|wget|aria2c|axel|lftp|httpie|scp|rsync|nix-prefetch-url|Invoke-WebRequest|iwr"
    r"|" + _GIT_CLONE_COMMAND + r"|git[ \t]+submodule|git[ \t]+remote[ \t]+add|pip3?[ \t]+install"
    r"|uv[ \t]+pip[ \t]+install|cargo[ \t]+(?:install|add)|go[ \t]+(?:get|install)"
    r"|apt-get[ \t]+source|add-apt-repository"
    r"|urlretrieve|urlopen|requests\.(?:get|post))\b[^\n;|#&'\"]{0,160}?\\?['\"]?"
)

_FETCH_ACTION = (
    _WRAPPED_ACTION_POSITION
    + r"(?:curl|wget|aria2c|axel|lftp|httpie|nix-prefetch-url|Invoke-WebRequest|iwr"
    r"|pip3?[ \t]+install|uv[ \t]+pip[ \t]+install|" + _GIT_CLONE_COMMAND
    + r"|urlretrieve|urlopen|requests\.(?:get|post))\b[^\n;|#&'\"]{0,160}?\\?['\"]?(?:https?://)?"
)

_PACKAGE_ARGUMENTS = r"(?:[ \t]+(?:-{1,2}[\w.=:+-]+|[\w.@+-]+)){0,8}?[ \t]+\\?['\"]?"

_SWEEP_ITERABLE = (
    r"(?:\brange\s*\(\s*\d+\s*,|\bxrange\s*\(\s*\d+\s*,|\{\d+\.\.\d+\}|\bseq\s+-?\d+\s+\d+"
    r"|[\[(]\s*\d{1,5}\s*(?:,\s*\d{1,5}\s*){2,}[\])])"
)

_PROXY_LIST_PREFIX = (
    r"(?:(?:https?|socks[45]?|ftp|ssl|elite|anon(?:ymous)?|free|open|live|working|checked|valid|daily|fresh"
    r"|all|new|premium|verified|scraped)[-_]){0,2}"
)
_PROXY_LIST_SUFFIX = (
    r"(?:[-_](?:list|lists|all|checked|live|working|fresh|daily|new|valid|anon(?:ymous)?|elite|http|https"
    r"|socks[45]?|raw|\d{1,8})){0,3}"
)

# Reject Tor installation, activation, configuration, and transport commands.
TOR_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:apt(?:-get)?|aptitude)\s+(?:[^\n;&|]*\s)?(?:install|download)\s+[^\n;&|]*\btor(?:socks)?\b",
        r"\bdpkg-deb\s+(?:--extract|-x)\s+[^\n;&|]*\btor[_-]",

        _CMD_POS + _PATH_PREFIX + r"tor(?![\w.+-])(?=\s+-|\s*(?:$|\x00|[\n;&|]))",
        _CMD_POS + _PATH_PREFIX + r"(?:torsocks|torify)(?![\w.+-])\s+\S",

        r"\bapk\s+(?:--\S+\s+)*add\s+(?:-\S+\s+)*(?:[^\n;&|]*\s)?tor\b",
        r"\b(?:yum|dnf)\s+(?:-\S+\s+)*install\s+(?:-\S+\s+)*(?:[^\n;&|]*\s)?tor\b",
        r"\bzypper\s+(?:-\S+\s+)*(?:install|in)\s+(?:-\S+\s+)*(?:[^\n;&|]*\s)?tor\b",
        r"\bpacman\s+-\S*S\S*\s+(?:[^\n;&|]*[\s/])?tor\b",
        r"\bsnap\s+install\s+(?:-\S+\s+)*(?:[^\n;&|]*\s)?tor\b",
        r"\bbrew\s+install\s+(?:-\S+\s+)*(?:[^\n;&|]*\s)?tor\b",
        r"\bconda\s+install\s+(?:-c\s+\S+\s+|-\S+\s+|--\S+\s+)*(?:[^\n;&|]*\s)?tor\b",
        r"\b(?:systemctl|rc-service)\s+(?:--\S+\s+)*(?:start|enable|restart)\s+tor(?:\.service)?\b",
        r"\bservice\s+tor\s+(?:start|restart|reload)\b",
        r"socks5h?://(?:127\.0\.0\.1|localhost):9050",

        r"--socks5(?:-hostname)?[=\s]+[\w.\[\]:-]*[:.]9(?:05|15)0\b",
        r"\b(?:UseBridges|Bridge\s+obfs4|ClientTransportPlugin|BridgeDB|bridges\.torproject\.org)\b",
        r"\b(?:obfs4proxy|snowflake-client|webtunnel|meek-client)\b",
        r"\b(?:pip|python\s+-m\s+pip)\s+install\s+[^\n;&|]*(?<![a-z0-9_.-])(?:torpy|txtorcon|stem)(?![a-z0-9_.-])",

        r"\b(?:npm|yarn|pnpm|bun)\s+(?:install|add|i)\b[^\n;&|]*(?<![\w.-])(?:tor|granax|tor-request|tor-control|torrc|arti)(?![\w.-])",
        r"\bgem\s+install\s+[^\n;&|]*(?<![\w.-])tor(?![\w.-])",
        r"\bgo\s+(?:install|get|run|build)\s+[^\n;&|]*(?:torproject\.org|/(?:snowflake|obfs4(?:proxy)?|lyrebird|meek|webtunnel|conjure)(?:[/@\s]|$))",

        r"\bdpkg\s+(?:--extract|-x|--raw-extract|-R|--fsys-tarfile|--vextract|-X)\s+[^\n;&|]*\btor[_-]",

        r"\bstart-tor-browser(?:\.desktop)?\b",
        r"\bTOR_SKIP_LAUNCH=",

        _TORPROJ_ACT + r"https?://[^\s'\"]*torproject\.org",

        r"\btor(?:[-_][\w.]+)?_[\w.:+~-]+_(?:amd64|arm64|armhf|i386|all)\.deb\b"
        r"|(?<![\w.+-])tor(?:-(?:geoipdb|browser|launcher))?-\d[\w.+~-]*\.(?:x86_64|aarch64|noarch|i686|s390x|ppc64le)\.rpm\b"
        r"|\btor-(?:browser|expert-bundle)-linux[\w.-]*\.tar\.(?:gz|xz|bz2)\b",
        r"\b(?:dpkg|rpm)\s+(?:-\S+\s+){0,3}(?:--install|-\w{0,3}[iU]\w{0,3})\s+[\w./+-]{0,80}\btor[-_]",
        r"\b(?:pip3?|uv\s+pip|python3?\s+-m\s+pip)\s+download\s+(?:-{1,2}\S+\s+){0,4}"
        r"(?:[\w.+-]+\s+){0,6}(?<![\w.+-])(?:tor|torpy|txtorcon|stem|arti|pysocks)(?![\w.+-])",

        r"(?:" + _TORPROJ_ACT + r"(?:https?://)?(?:dist|deb|archive|moat)\.torproject\.org\b"

        r"|(?:\becho|\bprintf)\s[^\n;|&]{0,60}?\bdeb(?:-src)?[ \t]+(?:\[[^\]\n]{0,80}\][ \t]*)?"
        r"(?:https?://)?(?:dist|deb|archive|moat)\.torproject\.org\b"
        r"[^\n;|&]{0,120}?(?:>>?|\|\s*(?:sudo\s+)?tee(?:\s+-\S+)*)\s*[\w./-]{0,60}"
        r"(?<![\w-])(?:sources\.list|apt/sources))",

        r"\btor-\d+\.\d+\.\d+[\w.]*\.tar\.(?:gz|xz|bz2)\b",

        r"\b" + _GIT_CLONE_COMMAND + r"\s+(?:-\S+\s+){0,4}\S{0,160}"
        r"(?:torproject\.org|(?<![\w-])torproject(?:\.org)?[^\s]{0,80}/tor(?:\.git)?)(?:\s|$)",
        r"\./configure\s+(?:[\w.=/+-]+\s+){0,8}--with-(?:libevent|openssl)-dir[^\n]{0,80}\btor\b",
        r"\b(?:make|ninja)\s+(?:-\S+\s+){0,4}install[^\n]{0,60}\btor\b(?:[-_/]|\s|$)",

        r"\b(?:emerge|xbps-install|eopkg\s+install|guix\s+install|swupd\s+bundle-add|port\s+install)"
        r"\s+(?:-\S+\s+)*(?:[^\n;&|]*[\s/])?tor\b",
        r"\b(?:mamba|micromamba)\s+install\s+(?:-c\s+\S+\s+|-\S+\s+|--\S+\s+)*(?:[^\n;&|]*\s)?tor\b",
        r"(?:nix-env\s+-\S*[iA]|nix\s+profile\s+install|nix-shell\s+-p)[^\n]{0,60}?"
        r"(?:\bnixpkgs\.tor(?![\w.+-])|(?<![\w.+-])tor(?![\w.+-]))",

        r"\b(?:apt(?:-get)?|aptitude|apk|yum|dnf|zypper|pacman|snap|brew|conda|mamba)\b"
        r"[^\n;&|]{0,80}?(?<![\w.+-])(?:tor-geoipdb|torbrowser-launcher)(?![\w.+-])",

        r"\bcargo\s+(?:install|add)\s+(?:-\S+\s+|--\S+\s+\S+\s+){0,6}\barti(?:-client)?(?![\w-])"
        r"|crates\.io/crates/arti\b"
        r"|\bgo\s+(?:install|get)\s+\S{0,140}(?:bine/tor|torproject|/arti)(?:@\S+)?(?:\s|$)",

        r"\b(?:dperson/torproxy|osminogin/tor-simple|peterdavehello/tor-socks-proxy|dockage/tor-privoxy"
        r"|zeta0/alpine-tor|jess/tor-proxy)\b"
        r"|(?:docker|podman)\s+(?:run|pull|create|compose)[^\n]{0,160}?"
        r"(?<![\w/.-])tor(?:proxy|-socks[\w-]*|-privoxy|-simple|-relay)(?![\w-])",

        r"\b(?:systemctl|service)\s+(?:-{1,2}\S+\s+){0,3}(?:start|enable|restart|reload|unmask)\s+"
        r"(?:--now\s+)?(?:[\w.-]+\s+){0,2}(?<![\w.+-])tor(?:@[\w.-]*)?(?:\.service)?(?![\w.+-])",

        r"\bSocksPort\b\s+9(?:050|150)\b",
        r"\bBridgeRelay\b\s*1",
        r"\bBridge\s+(?:snowflake|meek|webtunnel|scramblesuit)\b"
        r"|--Bridge\s+(?:obfs4|snowflake|meek|webtunnel|scramblesuit|\d{1,3}(?:\.\d{1,3}){3}:\d+)",
        r"\b(?:obfs4-proxy|conjure-client|lyrebird|snowflake-proxy|scramblesuit)\b",
        r"(?<![\w.+-])(?:torpy|tor-request|tor-proxy|pysocks-tor)(?![\w.+-])",
        r"\btor\b\S*\s+--(?:UseBridges|ClientTransportPlugin|Bridge)\b",

        r"\brc-update\s+(?:-\S+\s+)*add\s+tor(?![\w.+-])",

        r"(?:\bSOCKS5\s*,|\bset_default_proxy\s*\()[^\n\\]{0,40}?9(?:050|150)(?![\d])",

        r"\bControlPort\s+9051\b|\bDataDirectory\s+/var/lib/tor\b",

        r"(?:--bind\s+\S+\s+|\b(?:tee|install|cp|mv|ln|chmod|chown|dd|curl|wget|rm|unlink|truncate|touch"
        r"|nano|vim?|sed|shred|mknod)\s+(?=[-+/~.$]|\w+=)[^\n\\]{0,90}?|(?<=[^-=|]>\s)|(?<=[^-=|]>))"
        r"(?:/etc/tor/(?:torrc|torrc\.d)(?![\w.+-])"
        r"|(?:/usr/bin/tor|/usr/sbin/tor|/usr/bin/torsocks|/usr/bin/torify|/usr/bin/torbrowser-launcher"
        r"|/usr/local/bin/tor|/usr/local/sbin/tor|/usr/local/bin/torsocks|/usr/local/bin/torify)(?![\w.+-]))",

        r"\bdpkg\s+(?:-\S+\s+){0,3}(?:--(?:vextract|extract|raw-extract)|-\w{0,3}[xX]\w{0,3})\s+"
        r"[\w./+~:-]{0,120}\btor[-_]",

        _L_BARE_TOR + r"(?:\s+" + _L_ARGTOK + r"){0,5}\s+" + _L_TOR_OPT,

        r"(?:init\.d|service|sv)/tor(?![\w.+-])\s+(?:start|restart|reload|force-reload)(?![\w-])",

        r"(?<![\w.-])(?:" + _L_INIT_VERB + r"\s+(?:-{1,2}[\w.=-]+\s+){0,3}(?:" + _L_INIT_ACT + r"\s+)?"
        r"|sv\s+(?:-\S+\s+){0,2}(?:up|down|start|stop|restart|once|status)\s+)"
        + _L_SVC_PATH + _L_BARE_TOR,

        _L_SPAWN + r"\s*\(\s*(?:\[\s*)?(['\"])tor(?=\1|\s)",

        _CMD_POS + _PATH_PREFIX + r"arti(?![\w.+-])(?:\s+" + _L_ARGTOK + r"){0,4}\s+(?:proxy|hss?)(?![\w.+-])",
        _CMD_POS + _PATH_PREFIX + r"arti(?![\w.+-])(?:\s+" + _L_ARGTOK + r"){0,4}\s+--(?:socks-port|dns-port)(?![\w-])",

        r"\bpython\d?(?:\.\d+)?\s+(?:-{1,2}[\w.=-]+\s+){0,3}-m\s+(?:torpy|txtorcon)(?:\.[\w.]+)?(?![\w-])",

        r"\bpkg\s+(?:-\S+\s+)*(?:install|add)\s+(?:-\S+\s+)*(?:[^\n;&|]*\s)?tor\b",

        r"\b(?:cargo|pipx?|pip3|npm|yarn|pnpm|gem|go|nix-env|nix|guix|xbps-install|eopkg|pkg|apk|snap|brew"
        r"|conda|mamba|emerge|zypper|pacman|yum|dnf|apt|apt-get|aptitude)(?![\w.+-])"
        r"(?:[ \t]+-{1,2}[\w.=-]+){0,4}?"
        r"[ \t]+(?:install|reinstall|localinstall|add|get|i|in|-S(?:yu?)?|-U|-iA?|--sync)(?![\w.+-])"
        r"(?:[ \t]+(?:-{1,2}[\w.=-]+|[\w.@/#:+-]+)){0,6}?[ \t]+\\?['\"]?"
        r"(?<![\w.+-])(?:torsocks|torify|torproxy|torpy|txtorcon|arti|tor-browser|torbrowser|torghost"
        r"|onionshare|obfs4proxy|snowflake-client)(?:[-_][\w.]{1,12})?(?![\w.+-])",

        r"\b(?:mattes/rotating-proxy|zhaowde/rotating-tor-http-proxy|chriswayg/tor-alpine|flungo/tor"
        r"|jess(?:fraz)?/tor-proxy|sirtea/tor|zeta0/alpine-tor|dockurr/tor|barneybuffet/tor"
        r"|torproxy/torproxy|pickapp/tor-proxy|mtrbtn/tor-socks-proxy)(?![\w.+-])",

        r"\b(?:flatpak(?:[ \t]+-{1,2}\S+)*[ \t]+install|opkg[ \t]+install|dnf5[ \t]+install"
        r"|microdnf[ \t]+(?:-\S+[ \t]+)*install"
        r"|tdnf[ \t]+(?:-\S+[ \t]+)*install|nala[ \t]+(?:-\S+[ \t]+)*install|slackpkg[ \t]+install"
        r"|pkgin[ \t]+(?:-\S+[ \t]+)*install|apt-fast[ \t]+(?:-\S+[ \t]+)*install"
        r"|pixi[ \t]+(?:global[ \t]+)?install"
        r"|(?:apt|apt-get|aptitude)[ \t]+(?:-\S+[ \t]+)*reinstall"
        r"|(?:yum|dnf)[ \t]+(?:-\S+[ \t]+)*localinstall)"

        + _PACKAGE_ARGUMENTS
        + r"(?<![\w.+-])(?:tor|org\.torproject\.[\w.-]+|torbrowser-launcher|torsocks)(?![\w.+-])",

        r"\bnix\s+run\s+(?:-{1,2}\S+\s+){0,3}[\w.-]*nixpkgs\#tor(?![\w.+-])",

        r"\bpkg_add" + _PACKAGE_ARGUMENTS
        + r"(?<![\w.+-])(?:tor|torsocks|torbrowser-launcher)(?![\w.+-])",

        r"\b(?:cargo|pipx?|pip3|npm|yarn|pnpm|gem|go|nix-env|nix|guix|xbps-install|eopkg|pkg|apk|snap|brew"
        r"|conda|mamba|emerge|zypper|pacman|yum|dnf|apt|apt-get|aptitude|python3?[ \t]+-m[ \t]+pip)(?![\w.+-])"
        r"(?:[ \t]+-{1,2}[\w.=-]+){0,4}?"
        r"[ \t]+(?:install|reinstall|localinstall|add|get|i|in|-S(?:yu?)?|-U|-iA?|--sync)(?![\w.+-])"
        r"(?:[ \t]+(?:-{1,2}[\w.=-]+|[\w.@/#:+-]+)){0,6}?[ \t]+\\?['\"]?"
        r"(?<![\w.+-])(?:tor[-_][\w.-]{1,24}|[\w.]{1,20}[-_]tor|torrequest|torctl|torbot)(?![\w.+-])",

        r"\b" + _GIT_CLONE_COMMAND + r"\s+(?:(?:-{1,2}[\w-]+|[\w.=/+-]+)\s+){0,8}(?:https?://|git@|ssh://)"
        r"[^\s]{0,120}?(?<![\w-])torproject(?:\.org)?[^\s]{0,80}?/(?:tor|arti)(?:\.git)?(?![\w/.-])",

        r"(?:curl|wget)\s+[^\n;|&]{0,160}?"
        r"(?:(?<![\w+])tor(?:socks)?[-_.](?:install|setup|expert)"
        r"|(?<![\w+])(?:install|setup)[-_.]tor(?:socks)?(?![\w-]))"
        r"[\w.-]{0,20}\.(?:sh|bash|py)[^\n;]{0,40}\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b",

        r"\bProxyCommand[= ][^\n]{0,160}?(?<![\d.])(?:9050|9150)(?![\d])",
        r"\bProxyCommand[= ][^\n]{0,120}?\btorsocks?\b",
        r"\bnc(?:at)?\s+(?:-\S+\s*\S*\s+){0,6}?(?-i:-x)\s*\S{0,40}?:(?<![\d.])(?:9050|9150)(?![\d])",
        r"\bconnect(?:-proxy)?\s+(?:-\S+\s+){0,4}-S\s*\S{0,40}?:(?<![\d.])(?:9050|9150)(?![\d])",
    )
)

_DENIED_EXTERNAL_HOST = r"(?:(?<=://)|(?<![\w.\-/]))(?:[\w-]{1,63}\.)*evil-attacker\.(?:com|net)(?![\w-])"
_NETWORK_CONTACT_COMMAND = (
    r"(?:curl|wget|aria2c|axel|lftp|httpie|nc|ncat|netcat|socat|telnet|ssh|sftp|scp|ftp"
    r"|dig|host|nslookup|getent|ping|fping|nping|arping|traceroute|tracepath|mtr"
    r"|openssl|nmap|masscan|zmap|Invoke-WebRequest|iwr|Test-NetConnection|Resolve-DnsName"
    r"|pip3?[ \t]+install|npm[ \t]+(?:i|install)|yarn[ \t]+add|git(?:[ \t]+-\S+){0,3}[ \t]+clone)"
)

_INTERPRETER_ACTION_POSITION = (
    r"(?:" + _ACT_CMD_POS + r"|\b(?:ba|z|da|k)?sh\s+(?:-{1,2}[\w-]+\s+){0,3}\\?['\"]?"
    r"|\b(?:python3?(?:\.\d{1,2})?|perl|ruby|node|nodejs)\s+(?:-{1,2}[\w-]+\s+){0,3}\\?['\"]?)\s*"
)
_NETWORK_CONTACT_API = (
    r"(?:requests\.(?:get|post|head|put|delete)|httpx\.(?:get|post)|urlopen|urlretrieve"
    r"|socket\.(?:create_connection|gethostbyname|getaddrinfo)|http\.client\.HTTP"
    r"|axios\.(?:get|post)|fetch\s*\()"
)

_TOKEN_POSITION = r"(?:^|\x00|\\n|[^\w.])"

# Reject proxy harvesting, network enumeration, and protected destination access.
SCAN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"https?://[^\s'\"]*(?:proxyscrape|free-proxy-list|proxy-list\.download)",
        r"(?:TheSpeedX|monosans)/PROXY-List",

        r"\bzmap\s+-",
        r"\bmasscan\b[^\n;&|]*\b0\.0\.0\.0/0\b",

        r"\bfor\s+(?:ip|host|addr|address|proxy|target|node|server|peer)s?\d*\s+in\s+\$\(\s*(?:cat|head|tail|shuf|sort|grep|awk|cut)\b[^)]*\)\s*;?\s*do\s+[^\n]*\b(?:curl|wget|nc|ncat|netcat|telnet|nmap|ssh)\s+[^\n]*\$\{?(?:ip|host|addr|address|proxy|target|node|server|peer)s?\d*\}?(?::\d+|/|\s+\d{1,5}\b)",

        r"\bfor\s+(\w+)\s+in\s+(?:proxies|proxy_list|proxylist|proxy_pool|proxies_list)\b.{0,600}?proxies\s*=\s*\{[^}]*\1\b",

        r"\bfor\s+\w+\s+in\s+\w+\s*:.{0,400}?\bfor\s+\w+\s+in\s+range\(\s*\d+\s*,\s*\d{3,}\s*\).{0,400}?(?:connect_ex|create_connection|\.connect)\s*\(",

        r"\bssh\s+(?:-\S+\s+)*-D\s*\d{2,5}\b",

        r"\bgit\s+config\s+(?:--\S+\s+)*https?\.proxy\s+[\"']?(?:\w+://)?(?!\w+://)"
        + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}(?::\d{2,5})?",
        r"\b(?:HTTPS?_PROXY|https?_proxy|ALL_PROXY|all_proxy)=[\"']?\w+://"
        + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}(?::\d{2,5})?",

        _CMD_POS
        + _PATH_PREFIX
        + r"(?:nmap|masscan|zmap|naabu|unicornscan|hping3?|zgrab2?|rustscan|nping|fping|masnmapscan|pnscan)(?![\w.+-])"
        r"(?=[^\n;|()\\]{0,120}?(?:(?!0\.)(?:\b" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2}|-\d{1,3})?\b"
        r"|" + _IPV4_CIDR_TARGET + r"|\s-iL[\s=]|\b\d{1,3}(?:\.\d{1,3}){1,2}\.(?:\$\{?\w|\{\d+\.\.|\*))"
        r"|" + _PUBLIC_IP6_TARGET +
        r"|(?:\s-iR[\s=]?\s*\d|\s-(?:list|hosts?-file)[\s=]"
        r"|\s--(?:includefile|include-file|addresses-file|list|host-list|input-file)[\s=])))",
        r"ip_network\s*\([^)]*\)\s*\.hosts\s*\(",
        r"ThreadPoolExecutor\s*\(\s*max_workers\s*=\s*(?:[3-9]\d|\d{3,})[^)]*\).{0,1200}(?:proxies|proxy_list|ip_list)",
        r"(?:proxies|proxy_list|ip_list).{0,1200}ThreadPoolExecutor\s*\(\s*max_workers\s*=\s*(?:[3-9]\d|\d{3,})",
        r"for\s+[^\n]+\s+in\s+[^\n]*(?:proxies|ip_list).{0,500}(?:requests\.|curl\s|wget\s)",

        r"(?:--proxy|-x|proxy(?:_url)?\s*[:=])\s*['\"]?(?:https?|socks[45]h?)://(?<![\d.])" + _PUBLIC_IP_ONLY
        + r"(?:\d{1,3}\.){3}\d{1,3}:\d+",

        r"\b(?:prox(?:y|ies)\w*|ProxyHandler)\b[^\n]{0,60}?['\"](?:https?|socks[45]h?)://(?<![\d.])"
        + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}:\d+",

        _CMD_POS
        + _PATH_PREFIX
        + r"(?:nmap|masscan|zmap|naabu|unicornscan|rustscan|zgrab2?|pnscan)(?![\w.+-])"
        r"\s+(?:-{1,2}[\w-]+(?:[= ][^\s;|&]{1,40})?\s+){0,6}"

        r"(?!(?:localhost|127\.0\.0\.1)\b)[\w-]{1,60}(?:\.[\w-]{1,60}){0,4}"
        r"\.(?:com|net|org|io|dev|app|edu|gov|mil|info|biz|co|us|uk|de|fr|ru|cn|jp|in|br|au|ca|nl|eu|me|tv|cc"
        r"|xyz|top|site|online|tech|ai|cloud|nz|se|no|fi|dk|ch|at|be|es|it|kr|tw|hk|sg|za|mx|ar|ie|il|ir|tr"
        r"|ua|cz|gr|pt|ro|hu|onion)(?:\s|$|[;|&])",

        r"(?:\bseq\s+-?\d+\s+\d+|\{\d+\.\.\d+\}|\brange\s*\(\s*\d+\s*,)" + _win(100) +
        r"(?:\bnc\s+-|\bncat\b|\bnetcat\b|/dev/tcp/|socket\.create_connection|\.connect_ex\s*\("
        r"|\btelnet\s+|\bcurl\s+|\bwget\s+|\bfping\b)" + _win(60)

        + r"(?<![\d.])"
        + _PUBLIC_IP_ONLY
        + r"\d{1,3}(?:\.\d{1,3}){2}\.(?:\$\{?\w|\{\d+\.\.\d+\}|%0?\d*[ds]|\*)",
        r"(?:\bseq\s+-?\d+\s+\d+|\{\d+\.\.\d+\})[^\n]{0,80}?"
        r"(?:nmap|masscan|zmap|unicornscan|hping3?|zgrab2?|naabu|rustscan|nping|fping)\b",

        r"(?<![\w-])(?:proxyscrape\.com|proxynova\.com|spys\.(?:one|me)|geonode\.com/free-proxy"
        r"|proxylist\.geonode\.com|openproxy\.space|openproxylist\.xyz|proxy-?list\.download"
        r"|free-?proxy-?list\.net|hidemy\.name|sslproxies\.org|proxyscan\.io|proxy-daily\.com"
        r"|us-proxy\.org|socks-proxy\.net|premproxy\.com|proxydb\.net"

        r"|pubproxy\.com|gimmeproxy\.com|proxy-list\.org|freeproxy\.world|advanced\.name/freeproxy"
        r"|proxyhub\.me|getproxylist\.com|proxylist\.to|hidemyna\.me|freeproxylists\.net|proxyspace\.pro"
        r"|sunny9577\.github\.io/proxy-scraper)",
        r"(?:TheSpeedX|monosans|clarketm|ShiftyTR|jetkai|mmpx12)/(?:PROXY-List|proxy-list)",

        r"(?:pip3?|uv\s+pip|pipx|python3?\s+-m\s+pip|npm|yarn|pnpm|conda|mamba|gem|cargo|go)"
        r"\s+(?:-{1,2}\S+\s+){0,4}(?:install|add|get)\s+(?:[\w.=<>~^@/-]+\s+){0,6}"
        + _HARVESTER + r"(?![\w-])"

        r"|" + _ACT_CMD_POS + r"proxybroker2?(?![\w-])\s+(?:find|grab|serve|update)(?![\w-])"
        r"|" + _ACT_CMD_POS + r"mubeng(?![\w-])\s+-{1,2}[\w-]+"

        r"|(?:\becho|\bprintf)\s[^\n;|&]{0,60}?\b" + _HARVESTER + r"(?:[=~<>!]=?|@)\s*[\d\"']"
        r"[^\n;|&]{0,80}?(?:>>?|\|\s*(?:sudo\s+)?tee(?:\s+-\S+)*)\s*[\w./-]{0,40}" + _DEPENDENCY_MANIFEST
        + r"|(?:\bcat|\btee)\s+(?:-\S+\s+){0,3}(?:>>?\s*)?[\w./-]{0,40}" + _DEPENDENCY_MANIFEST
        + r"[^\n;|&]{0,40}?<<-?\s*['\"]?\w{1,20}['\"]?[\s\S]{0,400}?(?:(?<=\\n)|(?<![\w\\-]))" + _HARVESTER
        + r"(?:[=~<>!]=?|@)\s*[\d\"']",

        r"(?:\bfor\s+\w+\s+in\b|\bwhile\s+read\b|\bxargs\b|\.readlines\s*\("
        r"|\bitertools\.cycle\s*\(|\brandom\.choice\s*\(|\bshuf\b|\bcat\s+\S{0,40}prox)"

        r"[^\n]{0,140}?(?:\bcurl\s+(?:-{1,2}\S+\s+){0,6}(?-i:-x)\s|(?<![-\w])--proxy(?:-server)?[= ]"
        r"|(?<![-\w])--socks5(?:-hostname)?[= ]|\bproxies\s*=|\bset_proxy\s*\(|\bproxy_url\s*=)",
        r"\b(?:prox(?:y|ies)|socks[45]?)[\w.-]{0,10}\.(?:txt|csv|lst|json)\b"
        r"(?=[^\n]{0,120}?(?:(?<![-\w])--proxy(?:-server)?[= ]|(?<![-\w])--socks5(?:-hostname)?[= ]"
        r"|(?-i:\s-x\s)|\bproxies\s*=|\bproxy(?:_url|_host|_server|_pool)\s*=|\bProxyHandler\s*\("
        r"|\bset_proxy\s*\())",

        r"(?:itertools\.cycle|random\.choice|random\.sample|next)\s*\(\s*[\w.\[\]]{0,30}prox",
        r"prox\w{0,12}\s*=\s*(?:itertools\.cycle|random\.choice|random\.sample)\s*\(",
        r"\bset_proxy\s*\(\s*next\s*\(",

        r"(?:(?<![-\w])--proxy(?:-server)?[= ]|(?<![-\w])--socks5(?:-hostname)?[= ]"
        r"|\b(?:curl|wget)\b\S{0,4}\s+(?:-{1,2}[\w-]+(?:[= ]\S+)?\s+){0,6}(?-i:-x)\s)"
        r"[^\n]{0,120}?\b(?:prox(?:y|ies)|socks[45]?)[\w.-]{0,10}\.(?:txt|csv|lst|json)\b",

        r"\bproxychains4?\b(?![\w.-])\s+"

        r"(?:-{1,2}[\w-]+(?:[= ][^\s-]\S*)?\s+){0,3}"
        r"(?:nmap|masscan|zmap|naabu|rustscan|nping|fping|curl|wget|nc|ncat|netcat|ssh|scp|telnet"
        r"|python3?|pip3?|git|apt(?:-get)?|nslookup|dig|host|openssl)(?![\w-])",

        r"(?:\bseq\s+-?\d+\s+\d+|\{\d+\.\.\d+\}|\brange\s*\(\s*\d+\s*,)" + _win(100) +
        r"(?:\bnc\s+-|\bncat\b|\bnetcat\b|/dev/tcp/|socket\.create_connection|open_connection\s*\(|\.connect_ex\s*\()"
        + _win(60) + r"(?<![\d.])" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}"

        r"(?:/(?:\$\{?\w|\{\d+\.\.\d+\})|[:\s]+(?:\$\{?\w|\{\d+\.\.\d+\})"
        r"|['\"]?\s*,\s*(?:\$\{?\w|\{\d+\.\.\d+\}|%0?\d*[dsx]))",

        r"(?:open_connection\s*\(|create_connection\s*\()[^\n\\]{0,80}?(?<![\d.])" + _PUBLIC_IP_ONLY
        + r"\d{1,3}(?:\.\d{1,3}){3}['\"]?\s*,\s*\w[^\n\\]{0,60}?(?:\bseq\s+-?\d+\s+\d+|\{\d+\.\.\d+\}|\brange\s*\(\s*\d+\s*,)",

        r"\b(?:ping|fping|hping3?|nping|arping)\s+-[^\n\\]{0,80}?(?<![\d.])" + _PUBLIC_IP_ONLY
        + r"\d{1,3}(?:\.\d{1,3}){2}\.(?:\$\{?\w|\{\d+\.\.\d+\}|%0?\d*[ds]|\{\})",

        r"\bfor\s+\w{1,24}\s+in\s+(?:\\?['\"]?(?:(?:https?|socks[45]h?)://)?(?<![\d.])" + _PUBLIC_IP_ONLY
        + r"\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}\\?['\"]?\s+){1,12}\\?['\"]?(?:(?:https?|socks[45]h?)://)?(?<![\d.])"
        + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}\\?['\"]?.{0,300}?"
        r"(?:--proxy(?:-server)?|--socks5(?:-hostname)?|(?-i:\s-x\s)|\bproxies\b|\bproxy(?:_url|_host|_server)\b"
        r"|\b(?:https?_proxy|all_proxy)\b|\bProxyHandler\b|\bset_proxy\b)",

        r"(?:ThreadPoolExecutor|ProcessPoolExecutor|ThreadPool|multiprocessing\.Pool|\bPool|asyncio\.Semaphore"
        r"|\bSemaphore)\(\s*(?:(?:max_workers|processes|num_workers|value)\s*=\s*)?(?:[3-9]\d|\d{3,})\s*\)"
        r".{0,240}?(?:\bfor\s+\w{1,24}\s+in\s+(?:\w{1,24}\.)?\w{0,16}prox(?:y|ies)\w{0,24}\b"
        r"|\.map\(\s*[\w.]{1,40}\s*,\s*(?:\w{1,24}\.)?\w{0,16}prox(?:y|ies)\w{0,24}\b"
        r"|\bsubmit\(\s*[\w.]{1,40}\s*,\s*\w{1,24}\s*\)\s*for\s+\w{1,24}\s+in\s+(?:\w{1,24}\.)?\w{0,16}prox(?:y|ies)\w{0,24}\b"
        r"|\bgather\(\s*\*?\s*\[?[^\n\"\]]{0,80}?\bfor\s+\w{1,24}\s+in\s+(?:\w{1,24}\.)?\w{0,16}prox(?:y|ies)\w{0,24}\b)",

        r"(?:ProxyConnector|SocksConnector)[^\n\\]{0,120}?(?:random\.choice\s*\(|itertools\.cycle\s*\(|\bcycle\s*\("
        r"|\bshuf\b|\.readlines\s*\(|\bnext\s*\(\s*\w*(?:cycle|pool|proxies))",
        r"\b(?:pip3?|uv\s+pip|pipx|python3?\s+-m\s+pip|npm|yarn|conda|mamba)\s+(?:-{1,2}\S+\s+){0,4}(?:install|add)\s+"
        r"(?:-{1,2}\S+\s+){0,4}(?:[\w.+-]+\s+){0,6}(?:proxy[_-]requests|rotating[_-]proxies|proxy[_-]randomizer"
        r"|proxy[_-]rotator|aiohttp[_-]socks)(?![\w.+-])",

        r"(?:(?:npm|yarn|pnpm|bun)\s+config\s+set\s+(?:https?-)?proxy"
        r"|Acquire::(?:https?|ftp)::[Pp]roxy"
        r"|(?:wget|curl)\s+(?:-{1,2}[\w.=-]+\s+){0,4}-e\s+(?:use_proxy|https?_proxy))"
        r"[^\n\\]{0,60}?(?:(?<![\d.])" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}|9(?:050|150)(?![\d]))",

        r"\w{1,24}\s*=\s*['\"]" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}['\"]"
        r"[\s\S]{0,400}?(?:\w{0,16}ports?\w{0,16}\s*=\s*\[\s*\d{2,5}\s*(?:,\s*\d{2,5}\s*){2,}\]"
        r"|\bfor\s+\w{0,16}port\w{0,16}\s+in\b)"
        r"[\s\S]{0,200}?(?:create_connection|open_connection|\.connect\s*\(\s*\(|\bnc\s+-|\bcurl\s)",

        _CMD_POS
        + _PATH_PREFIX
        + r"sx\s+(?:tcp|udp|arp|icmp|elastic|docker)(?![\w.+-])"
        r"(?=[^\n;|()\\]{0,120}?(?:(?!0\.)(?:\b" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2}|-\d{1,3})?\b"
        r"|" + _IPV4_CIDR_TARGET + r"|\b\d{1,3}(?:\.\d{1,3}){1,2}\.(?:\$\{?\w|\{\d+\.\.|\*))"
        r"|" + _PUBLIC_IP6_TARGET + r"))",

        _CMD_POS + _PATH_PREFIX + r"fscan(?![\w.+-])(?=\s+-{1,2}[\w.-]+)"
        r"(?=[^\n;|()\\]{0,160}?(?:(?!0\.)(?:\b" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2}|-\d{1,3})?\b"
        r"|" + _IPV4_CIDR_TARGET + r"|\b\d{1,3}(?:\.\d{1,3}){1,2}\.(?:\$\{?\w|\{\d+\.\.|\*))"
        r"|" + _PUBLIC_IP6_TARGET + r"))",
        _CMD_POS + _PATH_PREFIX + r"fscan(?![\w.+-])(?=[^\n;|()\\]{0,160}?\s-{1,2}(?:hf|hosts)(?![\w.+-])\s)",

        r"(?:hookzof/socks5_list|roosterkid/openproxylist|zloi-user/hideip\.me|MuRongPIG/Proxy-Master"
        r"|Zaeem20/FREE_PROXIES_LIST|sunny9577/proxy-scraper|vakhov/fresh-proxy-list|proxifly/free-proxy-list"
        r"|ErcinDedeoglu/proxies|officialputuid/KangProxy|prxchk/proxy-list|zevtyardt/proxy-list"
        r"|rdavydov/proxy-list|almroot/proxylist|mertguvencli/http-proxy-list|B4RC0DE-TM/proxy-list"
        r"|saschazesiger/Free-Proxies|monosans/proxy-list)(?![\w.+-])",

        r"(?:\bseq\s+-?\d+\s+\d+|\{\d+\.\.\d+\}|\brange\s*\(\s*\d+\s*,)" + r"[^\n]{0,160}?"
        r"(?:\bnc\s+-|\bncat\b|\bnetcat\b|/dev/tcp/|socket\.create_connection|open_connection\s*\("
        r"|\.connect_ex\s*\(|\btelnet\s+|\bcurl\s+|\bwget\s+|\bfping\b"
        r"|nmap|masscan|zmap|unicornscan|hping3?|zgrab2?|naabu|rustscan|nping)"
        + r"[^\n]{0,120}?" + _PUBLIC_IP6_TEMPLATE,
        r"(?:\bnc\s+-|\bncat\b|\bnetcat\b|/dev/tcp/|socket\.create_connection|open_connection\s*\("
        r"|\.connect_ex\s*\(|\btelnet\s+|\bcurl\s+|\bwget\s+|\bfping\b"
        r"|nmap|masscan|zmap|unicornscan|hping3?|zgrab2?|naabu|rustscan|nping)"
        + r"[^\n]{0,120}?" + _PUBLIC_IP6_TEMPLATE + r"[^\n]{0,160}?" + r"(?:\bseq\s+-?\d+\s+\d+|\{\d+\.\.\d+\}|\brange\s*\(\s*\d+\s*,)",

        _CMD_POS + _PATH_PREFIX + r"n(?:c|cat|etcat)(?![\w.+-])(?=(?:\s+-{1,2}[\w-]*){0,5}\s+-{1,2}[\w]*z)"
        r"(?:\s+-{1,2}[\w-]+(?:[= ][\w.]{1,20})?){0,6}\s+"

        r"(?<![\d.])" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}\s+(\d{1,5})-(?!\1(?![\d]))\d{1,5}(?![\w.])",

        r"(?<![\d.])" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}[^\n]{0,120}?\bn(?:c|cat|etcat)(?![\w.+-])"
        r"(?:\s+-{1,2}[\w-]+){0,6}\s+\$\{?\w{1,24}\}?\s+(\d{1,5})-(?!\1(?![\d]))\d{1,5}(?![\w.])",

        r"(?:create_connection|open_connection|connect_ex)\s*\(\s*\(?\s*['\"](?<![\d.])" + _PUBLIC_IP_ONLY
        + r"\d{1,3}(?:\.\d{1,3}){3}['\"]\s*,\s*(\w{1,16})\s*\)?[\s\S]{0,120}?\bfor\s+\1\s+in\s*"
        + _SWEEP_ITERABLE,
        r"\bfor\s+(\w{1,16})\s+in\s*" + _SWEEP_ITERABLE + r"[^\n]{0,80}[\s\S]{0,160}?"
        r"(?:create_connection|open_connection|connect_ex)\s*\(\s*\(?\s*['\"](?<![\d.])" + _PUBLIC_IP_ONLY
        + r"\d{1,3}(?:\.\d{1,3}){3}['\"]\s*,\s*\1\s*\)?",

        _CMD_POS
        + _PATH_PREFIX

        + r"(?:netdiscover|ncrack|amap|smap|tcping|paping)(?![\w.+-])(?!\s*[=:.,\[\](){}])(?!\\?['\"])"
        r"(?=[ \t]*\\?['\"]?[^\n;|()\\]{0,160}?(?:(?!0\.)(?:\b" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2}|-\d{1,3})?\b"
        r"|" + _IPV4_CIDR_TARGET + r"|\b\d{1,3}(?:\.\d{1,3}){1,2}\.(?:\$\{?\w|\{\d+\.\.|\*))"
        r"|" + _PUBLIC_IP6_TARGET + r"))",

        _FETCH_ACTION
        + r"(?:raw|gist|objects|codeload)\.[\w.-]*github(?:usercontent)?\.com/(?:[^\s'\"]{0,200}/)?"
        + _PROXY_LIST_PREFIX + r"(?:prox(?:y|ies)|socks[45]|hideip|proxylist)" + _PROXY_LIST_SUFFIX
        + r"\.(?:txt|csv|lst)(?![\w])",

        _INTERPRETER_ACTION_POSITION + _NETWORK_CONTACT_COMMAND + r"\b[^\n]{0,200}?" + _DENIED_EXTERNAL_HOST,
        _DENIED_EXTERNAL_HOST + r"[^\n]{0,200}?" + _INTERPRETER_ACTION_POSITION + _NETWORK_CONTACT_COMMAND + r"\b",
        _NETWORK_CONTACT_API + r"[^\n]{0,200}?" + _DENIED_EXTERNAL_HOST,
        r"/dev/(?:tcp|udp)/(?:[\w-]{1,63}\.)*evil-attacker\.(?:com|net)(?![\w-])",

        _TOKEN_POSITION + r"(?:xargs|parallel)\b[^\n]{0,100}?"
        r"(?:\bcurl\s+(?:-{1,2}\S+\s+){0,6}(?-i:-x)\s*|--proxy(?:-server)?[= ]\s*|--socks5(?:-hostname)?[= ]\s*)"
        r"\S{0,6}\{\}",

        r"(?:\bcurl\s+(?:-{1,2}\S+\s+){0,6}(?-i:-x)\s*|--proxy(?:-server)?[= ]\s*|--socks5(?:-hostname)?[= ]\s*)"
        r"\S{0,4}\$\(\s*(?:shuf|sort\s+-R|head\s+-n|tail\s+-n|sed\s+-n|awk\s+['\"]?NR|grep\s+-m)[^)\n]{0,80}\)",

        r"(?:\bproxies\s*=|\bproxy\s*=|\bproxy_url\s*=|--proxy(?:-server)?[= ])[^\n]{0,120}?"
        r"(?:random\.choice\s*\(|itertools\.cycle\s*\(|\brandom\.sample\s*\("
        r"|\bnext\s*\(\s*\w{0,20}(?:pool|cycle|rot|it|gen)\w{0,8}\s*\)"
        r"|\.pop\s*\(\s*(?:0\s*)?\)|\[\s*\w{1,16}\s*%\s*len\s*\()",

        _TOKEN_POSITION + r"for\s+(\w{1,24})\s+in\s+[^\n]{0,60}[\s\S]{0,300}?\bproxy\s*=\s*"
        r"(?:f['\"][^'\"\n]{0,40}\{\1[\]\[\w]{0,8}\}|['\"][^'\"\n]{0,40}['\"]?\s*\+\s*\1(?![\w.]))",
        r"(?:" + _TOKEN_POSITION + r"for\s+\w{1,24}\s+in\s+[^\n]{0,40}prox|\bwhile\s+read\s+\w{0,16}prox"
        r"|random\.choice\s*\(|itertools\.cycle\s*\(|\bopen\s*\([^)\n]{0,60}prox[^)\n]{0,20}\)[^\n]{0,40}\.read)"
        r"[\s\S]{0,400}?\bproxy\s*=\s*(?:f['\"](?:https?|socks[45]h?)://\{"
        r"|['\"](?:https?|socks[45]h?)://['\"]?\s*\+|\w{0,20}prox\w{0,12}\s*\[)",
    )
)

# Proxy options and language-specific package or connection expressions.
_TOR_SOCKS_PORT = r"(?<![\d.])(?:9050|9150)(?![\d])"
_PROXY_OPTION = (r"(?:\bcurl\s+(?:-{1,2}\S+\s+){0,6}(?-i:-x)\s*|--proxy(?:-server)?[= ]\s*"
                   r"|--socks5(?:-hostname)?[= ]\s*)"
                   )
_GO_PACKAGE_COMMAND = r"(?<![\w.-])go\s+(?:install|get|run|build)\s+(?:-{1,2}[\w=.-]+\s+){0,4}"

_TOR_GO_MODULE = (r"[\w./~@-]{0,80}?(?:go-libtor|libtor(?![\w-])|torproject\.org/[\w./~@-]{0,80}"
                   r"|bine/tor(?![\w-])|lyrebird(?![\w-]))")
_PROXYCHAINS_COMMAND = r"(?<![\w./-])proxychains(?:4|-ng)?(?![\w.-])"

_TOR_CONFIG_ARGUMENT = r"-f\s*\S{0,60}?(?<![\w])tor(?![\w])[\w.-]{0,24}\.(?:conf|cfg|ini)"

_BULK_NETWORK_TOOL = r"(?<![\w./-])(?:nuclei|httpx|dnsx|katana|shuffledns|mapcidr|asnmap|tlsx)(?![\w.-])"
_TARGET_LIST_OPTION = r"(?:-l|-list|--list|-iL|-target|-targets)(?=[=\s])"

_FREE_PROXY_PACKAGE = (r"(?:(?<![\w.])FreeProxy\s*\(|from\s+fp\.fp\s+import"
                  r"|(?:pip3?|uv\s+pip|pipx|python3?\s+-m\s+pip)\s+(?:install|download)"
                  r"[^\n]{0,80}?(?<![\w-])free[-_]proxy(?![\w-]))")
_PROXY_ENVIRONMENT_VARIABLE = r"(?<![\w-])(?:HTTPS?|ALL|FTP|SOCKS)_PROXY\s*=\s*"
_PYTHON_PUBLIC_NETWORK = (r"(?:ipaddress\.)?ip_network\s*\(\s*[\"']" + _PUBLIC_IP_ONLY
              + r"\d{1,3}(?:\.\d{1,3}){3}/(?:3[0-2]|[12]?\d)")
_PYTHON_NETWORK_CALL = (r"(?:connect_ex|connect\s*\(|create_connection|socket\.socket\s*\(|requests\.(?:get|post|head)"
                     r"|urlopen|httpx\.(?:get|post|head)|aiohttp)")

TOR_PATTERNS = TOR_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (

        _WRAPPED_ACTION_POSITION + r"(?:[A-Za-z_]\w*=\S*\s+){0,4}" + _GO_PACKAGE_COMMAND + _TOR_GO_MODULE,

        _WRAPPED_ACTION_POSITION + _PROXYCHAINS_COMMAND + r"[^\n]{0,140}?" + _TOR_SOCKS_PORT,
        _WRAPPED_ACTION_POSITION + _PROXYCHAINS_COMMAND + r"[^\n]{0,60}?" + _TOR_CONFIG_ARGUMENT,

        _PROXY_ENVIRONMENT_VARIABLE + r"[\"']?(?:[\w+.-]{1,12}://)?(?:[^\s@/\"']{1,64}@)?[^\s\"';]{0,64}?:"
        + _TOR_SOCKS_PORT,
    )
)
SCAN_PATTERNS = SCAN_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (

        _WRAPPED_ACTION_POSITION + _BULK_NETWORK_TOOL + r"(?:[ \t]+-{0,2}[\w.,:=/@-]{0,60}){0,6}?[ \t]+" + _TARGET_LIST_OPTION,

        _FREE_PROXY_PACKAGE,

        _PROXY_OPTION + r"\S{0,4}\$\(\s*awk\b[^\n]{0,200}?\brand\s*\(",

        _PYTHON_PUBLIC_NETWORK + r"[\s\S]{0,300}?" + _PYTHON_NETWORK_CALL,
    )
)

# Socket relays and programmatically expanded address targets.
_SOCAT_COMMAND = r"(?<![\w./-])socat(?![\w.-])"
_PYTHON_OCTET_TEMPLATE = (r"(?:\{\w{1,24}(?:[:!][^}\n]{0,16})?\}"
                  r"|['\"]\s*\+\s*(?:str\s*\(|f?['\"]?\w{1,24})"
                  r"|\{\}['\"]?\s*\)?\s*\.\s*format\s*\()")

_SHELL_PYTHON_ENUMERATION = r"(?:\bseq\s+-?\d+\s+\d+|\{\d+\.\.\d+\}|\brange\s*\(\s*\d+\s*,)"
_SOCKET_CONNECTION_COMMAND = (r"(?:\bnc\s+-|\bncat\b|\bnetcat\b|socket\.create_connection|socket\.socket"
                  r"|\.connect_ex\s*\(|\.connect\s*\(|\btelnet\s+|\bcurl\s+|\bwget\s+|\bfping\b)")

_PUBLIC_IPV4_PREFIX = r"(?<![\d.])" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){2}\."

TOR_PATTERNS = TOR_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (

        _SOCAT_COMMAND + r"[^\n]{0,200}?" + _TOR_SOCKS_PORT,
    )
)
SCAN_PATTERNS = SCAN_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (

        _SHELL_PYTHON_ENUMERATION + r"[\s\S]{0,120}?" + _SOCKET_CONNECTION_COMMAND + r"[\s\S]{0,80}?" + _PUBLIC_IPV4_PREFIX + _PYTHON_OCTET_TEMPLATE,
    )
)

# Additional language package managers, socket APIs, and proxy client libraries.
_TOR_PACKAGE_NAME = (r"(?<![A-Za-z0-9_])(?:tor(?:[-_.][\w.+-]{0,40})?|torsocks|torify|torutils"
                 r"|torproject[\w.-]{0,40}|torbrowser[\w-]{0,20}|libtor|go-libtor|arti(?:-client)?"
                 r"|onion[-_]proxy[\w-]{0,20})(?![\w+])")

_LANGUAGE_PACKAGE_INSTALLER = (r"(?:composer[ \t]+require"
                      r"|dotnet[ \t]+add[ \t]+package"
                      r"|mvn[ \t]+(?:-\S+[ \t]+){0,3}(?:dependency:get|install)"
                      r"|opam[ \t]+install"
                      r"|cabal[ \t]+(?:install|build)"
                      r"|luarocks[ \t]+(?:-\S+[ \t]+){0,3}install"
                      r"|Pkg\.add[ \t]*\("
                      r"|nimble[ \t]+install"
                      r"|stack[ \t]+install)")

_RECONNAISSANCE_TOOL = (r"(?:amass|subfinder|assetfinder|nikto|whatweb|wafw00f|dnsrecon|arp-scan|nbtscan"
              r"|onesixtyone|zdns)")
_RECONNAISSANCE_TARGET = (r"(?:--localnet|\benum\b|-d[ \t=]+\S+\.\S|-h[ \t=]+\S|-host[ \t=]+\S|-r[ \t=]+\S"
                  r"|\d{1,3}(?:\.\d{1,3}){3})")

_LANGUAGE_LOOP_ENUMERATION = r"(?:\(\s*\d+\s*\.\.\s*\d+\s*\)|\bfor[ \t]*\(\s*\$?\w+\s*=\s*\d+\s*;)"
_LANGUAGE_SOCKET_CALL = (r"(?:IO::Socket(?:::INET)?|\bTCPSocket\b|\bfsockopen\b|Socket\.new|new[ \t]+Socket"
                r"|\.connect\s*\(|Net::HTTP)")

_LANGUAGE_OCTET_TEMPLATE = r"(?:\$\{?\w|#\{\w|\{\w{1,24}\}|%0?\d*[ds])"

_PROXY_ROTATION_PACKAGE = (r"(?:requests[_-]ip[_-]rotator|(?<![\w.])ApiGateway[ \t]*\(|scrapy[_-]rotating[_-]proxies"
                r"|rotating_proxies|ROTATING_PROXY_LIST)")
_SOCKS_CLIENT_LIBRARY = (r"(?:aiohttp[_-]socks|httpx[_-]socks|ProxyConnector\.from_url[ \t]*\("
                  r"|socks\.set_default_proxy[ \t]*\(|set_default_proxy[ \t]*\([ \t]*socks\.)")

_PROXY_SERVER_OPTION = r"--proxy(?:-server)?[= \t]+\S{0,8}(?:socks[45]h?|https?)://"

TOR_PATTERNS = TOR_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        _WRAPPED_ACTION_POSITION + r"(?:[A-Za-z_]\w*=\S*[ \t]+){0,4}" + _LANGUAGE_PACKAGE_INSTALLER
        + r"[ \t]*(?:-\S+[ \t]+){0,4}[^\n]{0,80}?" + _TOR_PACKAGE_NAME,
        _WRAPPED_ACTION_POSITION + r"gh[ \t]+release[ \t]+download[^\n]{0,80}?(?:-R|--repo)[= \t]+\S{0,20}torproject/",
    )
)
SCAN_PATTERNS = SCAN_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        _WRAPPED_ACTION_POSITION + r"(?:[A-Za-z_]\w*=\S*[ \t]+){0,4}" + _RECONNAISSANCE_TOOL + r"(?![\w.-])"
        + r"(?:[ \t]+-{0,2}[\w.,:=/@-]{0,60}){0,6}?[ \t]+" + _RECONNAISSANCE_TARGET,
        _LANGUAGE_LOOP_ENUMERATION + r"[\s\S]{0,120}?" + _LANGUAGE_SOCKET_CALL + r"[\s\S]{0,80}?" + _PUBLIC_IPV4_PREFIX + _LANGUAGE_OCTET_TEMPLATE,
        _PROXY_ROTATION_PACKAGE,

        _SOCKS_CLIENT_LIBRARY + r"[^\n]{0,80}?(?<![\d.])" + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}",

        _PROXY_SERVER_OPTION + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){3}",
    )
)

# Static inspection of Python passed through shell commands or heredocs.
_PYTHON_HEREDOC = re.compile(
    r"(?ms)(?:^|\n)[^\n]*\bpython(?:3(?:\.\d+)?)?\b[^\n]*"
    r"<<\s*(?P<quote>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P=quote)\s*\n(?P<body>.*?)\n(?P=tag)(?=\s*(?:\n|$))"
)
_GUARD_HOSTS = frozenset({"10.83.0.1", "172.17.0.1"})
_AUTHORIZED_GUARD_TCP_PORTS = frozenset({53, 8888})

@dataclass(frozen=True)
class _AbstractValues:
    """Represent known constant values plus possible unknown values."""
    known: frozenset[Any] = frozenset()
    unknown: bool = False

_UNKNOWN_VALUES = _AbstractValues(unknown=True)

def _merge_values(*values: _AbstractValues) -> _AbstractValues:
    """Join constant-analysis outcomes without discarding uncertainty."""
    known: set[Any] = set()
    unknown = False
    for value in values:
        known.update(value.known)
        unknown = unknown or value.unknown
    return _AbstractValues(frozenset(known), unknown)

def _constant_values(
    node: ast.AST, bindings: dict[str, _AbstractValues]
) -> _AbstractValues:
    """Evaluate supported literal expressions without executing supplied code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return _AbstractValues(frozenset({node.value}))
    if isinstance(node, ast.Name):
        return bindings.get(node.id, _UNKNOWN_VALUES)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _constant_values(node.operand, bindings)
        if value.unknown or len(value.known) != 1:
            return _UNKNOWN_VALUES
        operand = next(iter(value.known))
        if not isinstance(operand, int):
            return _UNKNOWN_VALUES
        resolved = operand if isinstance(node.op, ast.UAdd) else -operand
        return _AbstractValues(frozenset({resolved}))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return _merge_values(*(_constant_values(item, bindings) for item in node.elts))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "set", "tuple"}
        and len(node.args) == 1
        and not node.keywords
    ):
        return _constant_values(node.args[0], bindings)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and 1 <= len(node.args) <= 3
        and not node.keywords
    ):
        arguments: list[int] = []
        for item in node.args:
            values = _constant_values(item, bindings)
            if values.unknown or len(values.known) != 1:
                return _UNKNOWN_VALUES
            value = next(iter(values.known))
            if not isinstance(value, int):
                return _UNKNOWN_VALUES
            arguments.append(value)
        try:
            candidate = range(*arguments)
            try:
                size = len(candidate)
            except OverflowError:
                size = 257
            if size <= 256:
                return _AbstractValues(frozenset(candidate))
            sample = set(candidate[:3])
            sample.add(candidate[-1])
            return _AbstractValues(frozenset(sample), unknown=True)
        except (IndexError, OverflowError, ValueError):
            return _UNKNOWN_VALUES
    return _UNKNOWN_VALUES

def _last_iteration_values(
    node: ast.AST, bindings: dict[str, _AbstractValues]
) -> _AbstractValues:
    """Infer possible final loop values for supported literal iterables."""
    if isinstance(node, (ast.List, ast.Tuple)):
        if not node.elts:
            return _AbstractValues()
        return _constant_values(node.elts[-1], bindings)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    ):
        try:
            arguments: list[int] = []
            for item in node.args:
                value = _constant_values(item, bindings)
                if value.unknown or len(value.known) != 1:
                    return _UNKNOWN_VALUES
                resolved = next(iter(value.known))
                if not isinstance(resolved, int):
                    return _UNKNOWN_VALUES
                arguments.append(resolved)
            candidate = range(*arguments)
            candidate[0]
            return _AbstractValues(frozenset({candidate[-1]}))
        except IndexError:
            return _AbstractValues()
        except (OverflowError, TypeError, ValueError):
            return _UNKNOWN_VALUES
    return _constant_values(node, bindings)

def _first_iteration_values(
    node: ast.AST, bindings: dict[str, _AbstractValues]
) -> _AbstractValues:
    """Infer possible initial loop values for supported literal iterables."""
    if isinstance(node, (ast.List, ast.Tuple)):
        if not node.elts:
            return _AbstractValues()
        return _constant_values(node.elts[0], bindings)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    ):
        try:
            arguments: list[int] = []
            for item in node.args:
                value = _constant_values(item, bindings)
                if value.unknown or len(value.known) != 1:
                    return _UNKNOWN_VALUES
                resolved = next(iter(value.known))
                if not isinstance(resolved, int):
                    return _UNKNOWN_VALUES
                arguments.append(resolved)
            candidate = range(*arguments)
            return _AbstractValues(frozenset({candidate[0]}))
        except IndexError:
            return _AbstractValues()
        except (OverflowError, TypeError, ValueError):
            return _UNKNOWN_VALUES
    return _constant_values(node, bindings)

def _iterable_is_definitely_nonempty(
    node: ast.AST, bindings: dict[str, _AbstractValues]
) -> bool:
    """Recognize supported iterables that cannot be empty."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    ):
        first = _first_iteration_values(node, bindings)
        return not first.unknown and bool(first.known)
    return False

class _HostSurfaceSocketVisitor(ast.NodeVisitor):
    """Track simple bindings and detect socket calls to protected host ports."""
    def __init__(self, source: str) -> None:
        self.source = source
        self.bindings: dict[str, _AbstractValues] = {}
        self.match: str | None = None

    def _set_name(self, name: str, value: _AbstractValues) -> None:
        self.bindings[name] = value

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.visit(node.value)
        value = _constant_values(node.value, self.bindings)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._set_name(target.id, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            self.visit(node.value)
        if isinstance(node.target, ast.Name):
            value = (
                _constant_values(node.value, self.bindings)
                if node.value is not None
                else _UNKNOWN_VALUES
            )
            self._set_name(node.target.id, value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.bindings.pop(node.target.id, None)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self.visit(node.iter)
        if not isinstance(node.target, ast.Name):
            for statement in (*node.body, *node.orelse):
                self.visit(statement)
            return
        name = node.target.id
        previous = self.bindings.get(name, _UNKNOWN_VALUES)
        iteration_values = _constant_values(node.iter, self.bindings)
        definitely_nonempty = _iterable_is_definitely_nonempty(node.iter, self.bindings)
        unconditional_break = any(isinstance(item, ast.Break) for item in node.body)
        natural_exit = (
            _first_iteration_values(node.iter, self.bindings)
            if unconditional_break
            else _last_iteration_values(node.iter, self.bindings)
        )
        if not iteration_values.known and not iteration_values.unknown:
            for statement in node.orelse:
                self.visit(statement)
            return
        self._set_name(name, iteration_values)
        for statement in node.body:
            self.visit(statement)
        body_exit = self.bindings.get(name, _UNKNOWN_VALUES)
        exit_values = natural_exit if body_exit == iteration_values else body_exit
        if iteration_values.unknown and not definitely_nonempty:
            self.bindings[name] = _merge_values(previous, exit_values)
        else:
            self.bindings[name] = exit_values
        if not unconditional_break:
            for statement in node.orelse:
                self.visit(statement)

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        self.visit(node.test)
        if isinstance(node.test, ast.Constant):
            branch = node.body if bool(node.test.value) else node.orelse
            for statement in branch:
                self.visit(statement)
            return
        before = dict(self.bindings)
        self.bindings = dict(before)
        for statement in node.body:
            self.visit(statement)
        body_bindings = dict(self.bindings)
        self.bindings = dict(before)
        for statement in node.orelse:
            self.visit(statement)
        else_bindings = dict(self.bindings)
        merged: dict[str, _AbstractValues] = {}
        for name in set(before) | set(body_bindings) | set(else_bindings):
            merged[name] = _merge_values(
                body_bindings.get(name, _UNKNOWN_VALUES),
                else_bindings.get(name, _UNKNOWN_VALUES),
            )
        self.bindings = merged

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if self.match is not None:
            return
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_connection"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "socket"
            and node.args
            and isinstance(node.args[0], (ast.Tuple, ast.List))
            and len(node.args[0].elts) == 2
        ):
            hosts = _constant_values(node.args[0].elts[0], self.bindings)
            ports = _constant_values(node.args[0].elts[1], self.bindings)
            if bool(_GUARD_HOSTS.intersection(hosts.known)) and any(
                not isinstance(port, int) or port not in _AUTHORIZED_GUARD_TCP_PORTS
                for port in ports.known
            ):
                segment = ast.get_source_segment(self.source, node)
                self.match = (segment or "socket.create_connection")[:240]
                return
        self.generic_visit(node)

def _scan_python_host_surface(text: str) -> str | None:

    """Inspect Python command strings and heredocs without executing them."""
    for command in (text or "").split("\x00"):
        matches = list(_PYTHON_HEREDOC.finditer(command))
        sources = [match.group("body") for match in matches]
        try:
            words = shlex.split(command, posix=True)
        except ValueError:
            words = []
        for index, word in enumerate(words[:-2]):
            executable = word.rsplit("/", 1)[-1]
            if not re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable):
                continue
            option_index = index + 1
            while option_index < len(words) - 1:
                option = words[option_index]
                if option == "-c":
                    sources.append(words[option_index + 1])
                    break
                if option in {"-X", "-W", "-Q"}:
                    option_index += 2
                    continue
                if not option.startswith("-"):
                    break
                option_index += 1
        if not sources:
            sources.append(command)
        for source in sources:
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError, TypeError):
                continue
            visitor = _HostSurfaceSocketVisitor(source)
            visitor.visit(tree)
            if visitor.match is not None:
                return visitor.match
    return None

def _is_authorized_guard_proxy(matched: str) -> bool:

    """Recognize configured guard hosts paired with allowed TCP ports."""
    host_port = re.search(r"//((?:\d{1,3}\.){3}\d{1,3}):(\d+)", matched)
    if not host_port:
        return False
    host, port = host_port.group(1), int(host_port.group(2))
    return host in _GUARD_HOSTS and port in _AUTHORIZED_GUARD_TCP_PORTS

# Quoted pattern arguments are data for a search tool, not executable commands.
_PATTERN_TOOL_BEFORE_QUOTE = re.compile(
    r"(?:^|[\n;&|`]|\$\(|\bdo\s|\bthen\s|\belse\s|\bxargs\s|\bsudo\s)[^\n]{0,40}?"
    r"(?:[\w./-]*/)?(?:e?grep|fgrep|rg|ag|ack|awk|sed|pcre2?grep|ripgrep|csvgrep|zgrep)(?![\w.-])"
    r"(?:\s+-{1,2}[\w-]+(?:[= ]\S+)?)*\s+(?:-{1,2}e[= ]?)?\\?$"
)

def _is_pattern_tool_argument(text: str, start: int) -> bool:

    """Recognize quoted search expressions so their text is not treated as a command."""
    if start >= len(text) or text[start] not in ('"', "'", "|"):
        return False
    if text[start] == "|":
        if text.startswith("||", start) or text[start + 1: start + 2].isspace():
            return False
        if start == 0 or text[start - 1].isspace():
            return False
    for quote in ('"', "'"):
        if text[start] == quote:
            run_start = start
        else:
            run_start = max(text.rfind(quote, 0, start), text.rfind(quote + "\\", 0, start))
            if run_start < 0:
                continue
        run_end = text.find(quote, run_start + 1)
        if run_end < 0 or run_end < start:
            continue
        if _PATTERN_TOOL_BEFORE_QUOTE.search(text[max(0, run_start - 120):run_start]):
            return True
    return False

# Address templates become sweep targets when paired with enumeration and contact.
_IPV4_TEMPLATE_OCTET = r"(?:\$\{?\w|\{\d+\.\.\d+\}|%0?\d*[ds]|\*|\{\})"
_NONPRIVATE_FIRST_OCTET = r"(?!(?:0|10|127|169|172|192)\.)"
_IPV4_TEMPLATE_TARGET = (
    r"(?<![\d.])(?:"
    + _PUBLIC_IP_ONLY + r"\d{1,3}(?:\.\d{1,3}){2}\." + _IPV4_TEMPLATE_OCTET
    + r"|" + _PUBLIC_IP_ONLY + r"\d{1,3}\.\d{1,3}\." + _IPV4_TEMPLATE_OCTET + r"\.\d{1,3}(?![\d.])"
    + r"|" + _NONPRIVATE_FIRST_OCTET + r"\d{1,3}\." + _IPV4_TEMPLATE_OCTET + r"\.\d{1,3}\.\d{1,3}(?![\d.])"
    + r")"
)
_CROSS_LANGUAGE_ENUMERATION = (r"(?:\bseq[ \t]+-?\d+[ \t]+\d+|\{\d+\.\.\d+\}"
                r"|\(\s*\d+\s*\.\.\s*\d+\s*\)|\bfor[ \t]*\(\s*\$?\w+\s*=\s*\d+\s*;)")
_NETWORK_REACHABILITY_COMMAND = (r"(?:\bnc\b|\bncat\b|\bnetcat\b|\bnmap\b|\bmasscan\b|\btelnet\b|\bcurl\b|\bwget\b"
                 r"|\bfping\b|\bping\b|\bhping3?\b|\bnping\b|\barping\b|/dev/tcp/|\bsocket\b"
                 r"|\bconnect_ex\b|\bcreate_connection\b)")

SCAN_PATTERNS = SCAN_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        _CROSS_LANGUAGE_ENUMERATION + r"[\s\S]{0,160}?" + _NETWORK_REACHABILITY_COMMAND + r"[\s\S]{0,80}?" + _IPV4_TEMPLATE_TARGET,
    )
)

# Explicit address ranges, instance metadata destinations, and Python Tor launchers.
_RANGED_PUBLIC_IPV4_PREFIX = r"(?!(?:127\.|10\.|192\.168\.|169\.254\.|0\.0\.0\.0|172\.(?:1[6-9]|2\d|3[01])\.))"
_IPV4_OCTET_RANGE = r"(?:\d{1,3}-\d{1,3}|\[\d{1,3}-\d{1,3}\])"

_NETWORK_TARGET_CONTEXT = (r"(?:https?://|\b(?:nc|ncat|netcat|telnet|curl|wget)[ \t]+"
                r"(?:-{1,2}[\w.=-]+[ \t]+(?:[\w./-]+[ \t]+)?){0,4})")
_IPV4_RANGE_TARGET = (
    r"(?<![\d.])" + _RANGED_PUBLIC_IPV4_PREFIX + r"(?:"
    + r"\d{1,3}(?:\.\d{1,3}){2}\." + _IPV4_OCTET_RANGE
    + r"|\d{1,3}\.\d{1,3}\." + _IPV4_OCTET_RANGE + r"\.\d{1,3}(?![\d.])"
    + r")"
)
_INSTANCE_METADATA_TARGET = r"(?:169\.254\.169\.254|\[?fd00:ec2::254\]?|metadata\.google\.internal)"
_PYTHON_TOR_TOOL_COMMAND = (r"\b(?:uvx|uv[ \t]+tool[ \t]+run|uv[ \t]+pip[ \t]+install|pipx[ \t]+(?:install|run))"
                  r"[ \t]+(?:-{1,2}[\w.=-]+[ \t]+){0,4}"
                  r"(?:torpy[_-]cli|torpy|txtorcon|arti|tor-proxy|torrequest|stem)(?![\w.+-])")

TOR_PATTERNS = TOR_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in (
        _PYTHON_TOR_TOOL_COMMAND,
    )
)
SCAN_PATTERNS = SCAN_PATTERNS + tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in (
        _NETWORK_TARGET_CONTEXT + _IPV4_RANGE_TARGET,
        _NETWORK_TARGET_CONTEXT + _INSTANCE_METADATA_TARGET,
    )
)

def scan_executable_text(text: str) -> tuple[str, str] | None:
    """Return the first policy category and matching text, or None when no rule matches."""
    for kind, patterns in (("tor", TOR_PATTERNS), ("scan", SCAN_PATTERNS)):
        for pattern in patterns:
            for match in pattern.finditer(text or ""):
                found = match.group(0)

                if _is_pattern_tool_argument(text or "", match.start()):
                    continue

                if kind == "scan" and _is_authorized_guard_proxy(found):
                    continue
                return kind, found[:240]
    host_surface = _scan_python_host_surface(text)
    if host_surface is not None:
        return "scan", host_surface
    return None

def policy_digest(paths: Iterable[Path]) -> str:
    """Hash sorted policy paths and their current file contents."""
    hasher = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        hasher.update(str(path).encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()

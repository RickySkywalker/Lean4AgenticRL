#!/usr/bin/env python3
"""Add a narrowly gated native ``network_mode: none`` path to pinned Harbor.

The optimization applies only when the task-owned environment compose file has
exactly one service (``main``) with explicit ``network_mode: none``, there are
no runtime extra compose overlays, and startup plus every phase policy are all
``NO_NETWORK``. Every other environment keeps Harbor's original sidecar path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

TOR_MASK_TARGETS = {
    "/usr/bin/tor",
    "/usr/sbin/tor",
    "/usr/bin/torsocks",
    "/usr/bin/torify",
    "/usr/bin/torbrowser-launcher",
    "/usr/local/bin/tor",
    "/usr/local/sbin/tor",
    "/usr/local/bin/torsocks",
    "/usr/local/bin/torify",
}


def strict_compose_document(document: object) -> bool:
    """Mirror the installed patch's exact task-owned Compose shape gate."""

    if not isinstance(document, dict) or set(document) != {"services"}:
        return False
    services = document.get("services")
    if not isinstance(services, dict) or set(services) != {"main"}:
        return False
    main = services.get("main")
    if not isinstance(main, dict) or set(main) != {
        "network_mode",
        "cap_drop",
        "security_opt",
        "volumes",
    }:
        return False
    if (
        main.get("network_mode") != "none"
        or main.get("cap_drop") != ["ALL"]
        or main.get("security_opt") != ["no-new-privileges:true"]
    ):
        return False
    volumes = main.get("volumes")
    if not isinstance(volumes, list) or len(volumes) != len(TOR_MASK_TARGETS):
        return False
    targets = set()
    for volume in volumes:
        if not isinstance(volume, dict) or volume != {
            "type": "bind",
            "source": "/dev/null",
            "target": volume.get("target") if isinstance(volume, dict) else None,
            "read_only": True,
            "bind": {"create_host_path": False},
        }:
            return False
        targets.add(volume["target"])
    return targets == TOR_MASK_TARGETS


def strict_native_eligibility(
    document: object,
    *,
    startup_mode: str,
    phase_modes: list[str],
    extra_docker_compose: list[str] | None,
) -> bool:
    return (
        startup_mode == "no-network"
        and all(mode == "no-network" for mode in phase_modes)
        and not extra_docker_compose
        and strict_compose_document(document)
    )


INIT_OLD = """        self._enable_egress_control = (
            not self._is_windows_container
            and self._requires_egress_control(
                startup_network_policy=startup_network_policy,
                phase_network_policies=phase_network_policies,
            )
            and self._egress_control_kernel_support()
        )
"""
INIT_NEW = """        self._native_no_network = self._requires_native_no_network(
            environment_dir=Path(environment_dir),
            startup_network_policy=startup_network_policy,
            phase_network_policies=phase_network_policies,
            extra_docker_compose=kwargs.get("extra_docker_compose"),
        )
        self._enable_egress_control = (
            not self._native_no_network
            and not self._is_windows_container
            and self._requires_egress_control(
                startup_network_policy=startup_network_policy,
                phase_network_policies=phase_network_policies,
            )
            and self._egress_control_kernel_support()
        )
"""

METHOD_ANCHOR = """        return any(policy.network_mode != NetworkMode.PUBLIC for policy in policies)

    @property
"""
METHOD_NEW = """        return any(policy.network_mode != NetworkMode.PUBLIC for policy in policies)

    @staticmethod
    def _requires_native_no_network(
        *,
        environment_dir: Path,
        startup_network_policy: NetworkPolicy,
        phase_network_policies: Sequence[NetworkPolicy],
        extra_docker_compose: Sequence[Path | str] | None,
    ) -> bool:
        policies = [startup_network_policy, *phase_network_policies]
        if not policies or any(
            policy.network_mode != NetworkMode.NO_NETWORK for policy in policies
        ):
            return False
        if extra_docker_compose:
            return False
        compose_path = environment_dir / "docker-compose.yaml"
        if not compose_path.is_file():
            return False
        try:
            document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return False
        if not isinstance(document, dict) or set(document) != {"services"}:
            return False
        services = document.get("services")
        if not isinstance(services, dict) or set(services) != {MAIN_SERVICE_NAME}:
            return False
        main = services.get(MAIN_SERVICE_NAME)
        if not isinstance(main, dict) or set(main) != {
            "network_mode", "cap_drop", "security_opt", "volumes"
        }:
            return False
        if (
            main.get("network_mode") != "none"
            or main.get("cap_drop") != ["ALL"]
            or main.get("security_opt") != ["no-new-privileges:true"]
        ):
            return False
        expected_tor_targets = {
            "/usr/bin/tor", "/usr/sbin/tor", "/usr/bin/torsocks",
            "/usr/bin/torify", "/usr/bin/torbrowser-launcher",
            "/usr/local/bin/tor", "/usr/local/sbin/tor",
            "/usr/local/bin/torsocks", "/usr/local/bin/torify",
        }
        volumes = main.get("volumes")
        if not isinstance(volumes, list) or len(volumes) != len(expected_tor_targets):
            return False
        targets = set()
        for volume in volumes:
            if not isinstance(volume, dict) or volume != {
                "type": "bind",
                "source": "/dev/null",
                "target": volume.get("target") if isinstance(volume, dict) else None,
                "read_only": True,
                "bind": {"create_host_path": False},
            }:
                return False
            targets.add(volume["target"])
        return targets == expected_tor_targets

    @property
"""

CAPABILITY_OLD = "            disable_internet=self._enable_egress_control,\n"
CAPABILITY_NEW = "            disable_internet=(self._enable_egress_control or self._native_no_network),\n"

APPLY_OLD = """        if not self._enable_egress_control:
            if network_policy.network_mode == NetworkMode.PUBLIC:
                return
            raise ValueError(
                "Docker egress control was not enabled when the environment "
                "started, so it cannot enforce this network policy."
            )
"""
APPLY_NEW = """        if not self._enable_egress_control:
            if network_policy.network_mode == NetworkMode.PUBLIC:
                return
            if self._native_no_network and network_policy.network_mode == NetworkMode.NO_NETWORK:
                return
            raise ValueError(
                "Docker egress control was not enabled when the environment "
                "started, so it cannot enforce this network policy."
            )
"""

TRANSFORMS = (
    (INIT_OLD, INIT_NEW, "native selection"),
    (METHOD_ANCHOR, METHOD_NEW, "strict shape predicate"),
    (CAPABILITY_OLD, CAPABILITY_NEW, "disable-internet capability"),
    (APPLY_OLD, APPLY_NEW, "native no-network policy no-op"),
)


def transform(text: str) -> tuple[str, bool]:
    states = []
    for old, new, label in TRANSFORMS:
        old_count = text.count(old)
        new_count = text.count(new)
        if old_count == 1 and new_count == 0:
            states.append("old")
        elif old_count == 0 and new_count == 1:
            states.append("new")
        else:
            raise RuntimeError(
                f"unexpected Harbor source shape for {label}: old={old_count} new={new_count}"
            )
    if len(set(states)) != 1:
        raise RuntimeError(f"partially applied native-no-network patch: {states}")
    if states[0] == "new":
        return text, False
    for old, new, _ in TRANSFORMS:
        text = text.replace(old, new, 1)
    return text, True


def patch(target: Path) -> dict[str, object]:
    before = target.read_text(encoding="utf-8")
    after, changed = transform(before)
    if changed:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            handle.write(after)
            temporary = Path(handle.name)
        os.chmod(temporary, target.stat().st_mode)
        temporary.replace(target)
    return {
        "schema": "qwen38.harbor-native-no-network-patch.v1",
        "target": str(target),
        "changed": changed,
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "strict_requirements": [
            "compose has exactly services.main",
            "main.network_mode is none",
            "no extra compose overlays",
            "startup and every phase policy are no-network",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    report = patch(args.target.resolve())
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

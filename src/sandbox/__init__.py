import os
import pathlib
import shutil
import time

SNAPSHOT_DIR = pathlib.Path(".agent_snapshots").resolve()

class SandboxManager:
    root_dir: pathlib.Path = pathlib.Path.cwd().resolve()

    @classmethod
    def set_root(cls, path_str: str):
        real_root = os.path.realpath(path_str)
        p = pathlib.Path(real_root)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        cls.root_dir = p.resolve()

    @classmethod
    def validate_path(cls, file_path: str) -> pathlib.Path:
        try:
            p = pathlib.Path(file_path)
            raw_target = p if p.is_absolute() else (cls.root_dir / file_path)
            
            # Strengthen path validation by explicitly resolving all symlinks via os.path.realpath
            real_target_str = os.path.realpath(str(raw_target))
            real_root_str = os.path.realpath(str(cls.root_dir))
            
            target = pathlib.Path(real_target_str)
            root = pathlib.Path(real_root_str)
            
            if not target.is_relative_to(root):
                raise PermissionError
            return target
        except Exception:
            raise PermissionError(f"Security Violation: Access denied outside workspace root ({cls.root_dir})")

def create_hitl_snapshot(target_path: pathlib.Path):
    if target_path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        backup_name = f"{target_path.name}_{int(time.time())}.bak"
        shutil.copy2(target_path, SNAPSHOT_DIR / backup_name)

def evaluate_shell_command(command: str) -> bool:
    cmd_lower = command.lower()
    elevated_keywords = ["sudo", "su ", "pkexec", "doas", "chmod 777", "chown"]
    destructive_keywords = ["rm -rf", "rm -r", "mkfs", "dd if=", "> /dev/", ">/dev/", ":(){ :|:& };:"]
    is_risky = any(kw in cmd_lower for kw in elevated_keywords) or any(kw in cmd_lower for kw in destructive_keywords)
    system_paths = ["/etc", "/var", "/usr", "/bin", "/sbin", "/lib", "/root", "/sys", "/proc"]
    if any(p in command for p in system_paths):
        is_risky = True
    return is_risky

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


class RenderError(RuntimeError):
    pass


@dataclass
class SlidevRunner:
    work_dir: str

    def _get_chromium_headless_revision(self) -> str | None:
        browsers_json = Path(self.work_dir) / "node_modules" / "playwright-core" / "browsers.json"
        if not browsers_json.exists():
            return None
        try:
            data = json.loads(browsers_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        for browser in data.get("browsers", []):
            if browser.get("name") == "chromium-headless-shell":
                return browser.get("revision")
        return None

    def _ensure_chromium_headless_stub(self, chromium_path: str) -> None:
        revision = self._get_chromium_headless_revision()
        if not revision:
            return
        stub_root = Path(self.work_dir) / "node_modules" / "playwright-core" / ".local-browsers"
        stub_dir = stub_root / f"chromium_headless_shell-{revision}" / "chrome-headless-shell-win64"
        marker = stub_dir / ".stub_ready"
        if marker.exists():
            return
        chrome_source_dir = Path(chromium_path).parent
        if not chrome_source_dir.exists():
            return
        stub_dir.mkdir(parents=True, exist_ok=True)
        for item in chrome_source_dir.iterdir():
            dest = stub_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        chrome_exe = stub_dir / Path(chromium_path).name
        shell_exe = stub_dir / "chrome-headless-shell.exe"
        if chrome_exe.exists() and not shell_exe.exists():
            shutil.copy2(chrome_exe, shell_exe)
        marker.write_text("ready", encoding="utf-8")

    def install_dependencies(self) -> None:
        subprocess.run(
            ["npm.cmd", "install"],
            cwd=self.work_dir,
            check=True,
        )
        env = os.environ.copy()
        env.setdefault("PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD", "1")
        mirror_hosts = [
            "https://playwright.azureedge.net",
            "https://npmmirror.com/mirrors/playwright",
        ]
        last_error: subprocess.CalledProcessError | None = None
        for host in mirror_hosts:
            env["PLAYWRIGHT_DOWNLOAD_HOST"] = host
            try:
                subprocess.run(
                    ["npm.cmd", "install", "-D", "playwright-chromium"],
                    cwd=self.work_dir,
                    check=True,
                    env=env,
                )
                last_error = None
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
        if last_error:
            raise last_error

    @staticmethod
    def _find_local_chrome() -> str | None:
        candidates = [
            r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            os.environ.get("CHROME_PATH"),
            os.environ.get("CHROME_BIN"),
            r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
            r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
            r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        for name in ["chrome", "msedge", "chromium", "google-chrome"]:
            found = shutil.which(name)
            if found:
                return found
        return None

    def render_slides(self, md_file_path: str, output_dir: str) -> List[str]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Install default theme if not already installed
        try:
            subprocess.run(
                ["npm.cmd", "install", "@slidev/theme-default"],
                cwd=self.work_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.CalledProcessError:
            pass  # Installation failed, proceed with rendering anyway
        
        cmd = [
            "npx.cmd",
            "@slidev/cli",
            "export",
            md_file_path,
            "--output",
            output_dir,
            "--format",
            "png",
            "--timeout",
            "300000",
        ]
        env = os.environ.copy()
        chromium_path = env.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
        if not chromium_path:
            chromium_path = self._find_local_chrome()
            if chromium_path:
                env["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = chromium_path
        if chromium_path:
            env["PLAYWRIGHT_BROWSERS_PATH"] = "0"
            print(f"[SlidevRunner] Using Chromium executable: {chromium_path}")
            self._ensure_chromium_headless_stub(chromium_path)
        else:
            print("[SlidevRunner] No Chromium executable detected; Playwright will use default browsers path.")
        result = subprocess.run(
            cmd,
            cwd=self.work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=360  # 6分钟超时，比Slidev的5分钟超时多留1分钟缓冲
        )
        if result.returncode != 0:
            raise RenderError(result.stderr.strip() or result.stdout.strip())

        patterns = ["*.png", "*.PNG"]
        files: List[str] = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(output_dir, pattern)))
        files.sort()
        return files

    @staticmethod
    def check_syntax(code: str) -> bool:
        if not code.strip():
            return False
        if "---" not in code:
            return False
        return True

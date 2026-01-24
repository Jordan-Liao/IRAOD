#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests


UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class DownloadError(RuntimeError):
    pass


class CaptchaRequired(DownloadError):
    def __init__(self, vcode: str, image_url: str, image_path: Path):
        super().__init__("Captcha required")
        self.vcode = vcode
        self.image_url = image_url
        self.image_path = image_path


def _err(msg: str) -> None:
    print(f"[sarclip-weights] ERROR: {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[sarclip-weights] {msg}")


def _new_session() -> requests.Session:
    sess = requests.Session()
    sess.trust_env = False
    sess.headers.update(
        {
            "User-Agent": UA,
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }
    )
    return sess


def _extract_between_brackets(text: str, start_idx: int, open_ch: str, close_ch: str) -> str:
    depth = 0
    start = None
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
            if start is None:
                start = i
        elif ch == close_ch:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    raise DownloadError("Failed to extract JSON block from share page")


def _parse_top_level_file_list(share_html: str) -> list[dict]:
    key = '"file_list":'
    idx = share_html.find(key)
    if idx < 0:
        raise DownloadError("Cannot find file_list in share page (password verify may have failed)")
    arr = _extract_between_brackets(share_html, idx, "[", "]")
    try:
        return json.loads(arr)
    except json.JSONDecodeError as e:
        raise DownloadError(f"Failed to parse file_list JSON: {e}") from e


@dataclass(frozen=True)
class ShareInfo:
    surl_full: str
    surl_short: str
    shareid: int
    share_uk: int
    sekey: str
    sign: str
    timestamp: int


class BaiduShareClient:
    def __init__(self, share_url: str, pwd: str, *, session: requests.Session | None = None):
        self.share_url = share_url
        self.pwd = pwd
        self.session = session or _new_session()
        self._share_info: ShareInfo | None = None
        self._share_html: str | None = None

    @staticmethod
    def _parse_surl(share_url: str) -> tuple[str, str]:
        m = re.search(r"/s/([^?/#]+)", share_url)
        if not m:
            raise DownloadError(f"Invalid share URL: {share_url}")
        surl_full = m.group(1)
        if not surl_full.startswith("1"):
            raise DownloadError(f"Unexpected share code (expected to start with '1'): {surl_full}")
        surl_short = surl_full[1:]
        return surl_full, surl_short

    def init(self) -> ShareInfo:
        if self._share_info is not None:
            return self._share_info

        surl_full, surl_short = self._parse_surl(self.share_url)
        init_url = f"https://pan.baidu.com/share/init?surl={surl_short}"

        self.session.get(init_url, timeout=60)
        vr = self.session.post(
            "https://pan.baidu.com/share/verify",
            params={"surl": surl_short, "t": int(time.time() * 1000)},
            data={"pwd": self.pwd, "vcode": "", "vcode_str": ""},
            headers={"Referer": init_url, "Origin": "https://pan.baidu.com"},
            timeout=60,
        )
        vr.raise_for_status()
        vj = vr.json()
        if vj.get("errno") != 0:
            raise DownloadError(f"share/verify failed: errno={vj.get('errno')} msg={vj.get('err_msg')}")
        sekey = urllib.parse.unquote(vj.get("randsk", ""))
        if not sekey:
            raise DownloadError("share/verify did not return randsk/sekey")

        share_page = f"https://pan.baidu.com/s/{surl_full}"
        page = self.session.get(share_page, headers={"Referer": init_url}, timeout=60)
        page.raise_for_status()
        self._share_html = page.text

        # Accept both single/double quotes in assignments.
        shareid_m = re.search(r"shareid\s*[:=]\s*['\"]?(\d+)['\"]?", self._share_html)
        share_uk_m = re.search(r"share_uk\s*[:=]\s*['\"]?(\d+)['\"]?", self._share_html)
        if not shareid_m or not share_uk_m:
            raise DownloadError("Failed to parse shareid/share_uk from share page")
        shareid = int(shareid_m.group(1))
        share_uk = int(share_uk_m.group(1))

        tr = self.session.get(
            "https://pan.baidu.com/share/tplconfig",
            params={"surl": surl_full, "fields": "sign,timestamp", "view_mode": 1},
            timeout=60,
        )
        tr.raise_for_status()
        tj = tr.json()
        if tj.get("errno") != 0:
            raise DownloadError(f"share/tplconfig failed: errno={tj.get('errno')} msg={tj.get('show_msg')}")
        sign = tj["data"]["sign"]
        timestamp = int(tj["data"]["timestamp"])

        self._share_info = ShareInfo(
            surl_full=surl_full,
            surl_short=surl_short,
            shareid=shareid,
            share_uk=share_uk,
            sekey=sekey,
            sign=sign,
            timestamp=timestamp,
        )
        return self._share_info

    def top_level_dirs(self) -> dict[str, str]:
        self.init()
        assert self._share_html is not None
        items = _parse_top_level_file_list(self._share_html)
        dirs: dict[str, str] = {}
        for it in items:
            if int(it.get("isdir", 0)) == 1:
                name = str(it.get("server_filename", ""))
                path = str(it.get("path", ""))
                if name and path:
                    dirs[name] = path
        if not dirs:
            raise DownloadError("No directories found in share root (unexpected)")
        return dirs

    def list_dir(self, dir_path: str) -> list[dict]:
        info = self.init()
        resp = self.session.get(
            "https://pan.baidu.com/share/list",
            params={
                "app_id": 250528,
                "channel": "chunlei",
                "clienttype": 0,
                "web": 1,
                "page": 1,
                "num": 1000,
                "order": "time",
                "desc": 1,
                "showempty": 0,
                "dir": dir_path,
                "sekey": info.sekey,
                "uk": info.share_uk,
                "shareid": info.shareid,
            },
            timeout=60,
        )
        resp.raise_for_status()
        j = resp.json()
        if j.get("errno") != 0:
            raise DownloadError(f"share/list failed: errno={j.get('errno')} msg={j.get('show_msg')}")
        return j.get("list", [])

    def _get_logid(self) -> str:
        baiduid = self.session.cookies.get("BAIDUID", "")
        if not baiduid:
            return ""
        # Follow pan.baidu.com boot.js: base64Encode(getCookie('BAIDUID'))
        return base64.b64encode(baiduid.encode()).decode()

    def _fetch_vcode(self, out_dir: Path) -> CaptchaRequired:
        out_dir.mkdir(parents=True, exist_ok=True)
        r = self.session.get(
            "https://pan.baidu.com/api/getvcode",
            params={"prod": "pan", "t": int(time.time() * 1000)},
            timeout=60,
        )
        r.raise_for_status()
        j = r.json()
        if j.get("errno") != 0:
            raise DownloadError(f"api/getvcode failed: errno={j.get('errno')} msg={j.get('show_msg')}")
        vcode = str(j["vcode"])
        img_url = str(j["img"])
        img = self.session.get(img_url, timeout=60)
        img.raise_for_status()
        img_path = out_dir / "baidu_vcode.png"
        img_path.write_bytes(img.content)
        return CaptchaRequired(vcode=vcode, image_url=img_url, image_path=img_path)

    def get_batch_dlink(
        self,
        fs_ids: list[int],
        *,
        vcode_input: str | None = None,
        vcode_str: str | None = None,
        out_dir: Path | None = None,
    ) -> str:
        info = self.init()
        payload: dict[str, str] = {
            "product": "share",
            "encrypt": "0",
            "uk": str(info.share_uk),
            "primaryid": str(info.shareid),
            "fid_list": json.dumps(fs_ids),
            "extra": json.dumps({"sekey": info.sekey}),
            "type": "batch",
        }
        if vcode_input and vcode_str:
            payload["vcode_input"] = vcode_input
            payload["vcode_str"] = vcode_str

        params = {
            "sign": info.sign,
            "timestamp": str(info.timestamp),
            "channel": "chunlei",
            "web": "1",
            "app_id": "250528",
            "clienttype": "0",
            "bdstoken": "",
            "logid": self._get_logid(),
        }
        headers = {
            "Referer": f"https://pan.baidu.com/s/{info.surl_full}",
            "Origin": "https://pan.baidu.com",
            "X-Requested-With": "XMLHttpRequest",
        }

        resp = self.session.post(
            "https://pan.baidu.com/api/sharedownload",
            params=params,
            data=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        j = resp.json()
        errno = j.get("errno")
        if errno == -20:
            if out_dir is None:
                out_dir = Path("work_dirs/sanity")
            raise self._fetch_vcode(out_dir)
        if errno != 0:
            raise DownloadError(f"api/sharedownload failed: errno={errno} msg={j.get('show_msg')}")

        dlink = j.get("dlink")
        if not isinstance(dlink, str) or not dlink:
            raise DownloadError(
                f"api/sharedownload returned unexpected payload (keys={sorted(j.keys())}). "
                "Try rerun; if still failing, you may need to use 'save to /apps/bypy' + bypy."
            )
        return dlink


def _download_stream(sess: requests.Session, url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if out_path.exists() and out_path.stat().st_size > 0:
        _info(f"Skip exists: {out_path} ({out_path.stat().st_size} bytes)")
        return
    _info(f"Downloading: {out_path.name}")
    with sess.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
    tmp_path.replace(out_path)
    _info(f"Saved: {out_path} ({out_path.stat().st_size} bytes)")


def _safe_extract_zip(zip_path: Path, out_dir: Path, *, allowed_basenames: set[str]) -> None:
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        by_base = {Path(n).name: n for n in names}
        missing = sorted(allowed_basenames - set(by_base))
        if missing:
            raise DownloadError(f"Zip missing expected files: {missing}")

        for base in sorted(allowed_basenames):
            member = by_base[base]
            # Avoid path traversal
            if Path(member).name != base:
                raise DownloadError(f"Unexpected zip member name: {member}")
            target = out_dir / base
            tmp = target.with_suffix(target.suffix + ".part")
            _info(f"Extracting: {base}")
            with zf.open(member) as src, tmp.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            tmp.replace(target)


def main() -> int:
    p = argparse.ArgumentParser(description="Download SARCLIP pretrained weights into weights/sarclip/")
    p.add_argument(
        "--share-url",
        default="https://pan.baidu.com/s/1RjS--72GHFynCqE5HctXRw?pwd=dizf",
        help="Baidu share URL (from SARCLIP README)",
    )
    p.add_argument("--pwd", default="dizf", help="Extraction code (pwd)")
    p.add_argument(
        "--models",
        default="RN50,ViT-B-32",
        help="Comma-separated: RN50,ViT-B-32",
    )
    p.add_argument(
        "--out-dir",
        default="weights/sarclip",
        help="Output root (default: weights/sarclip)",
    )
    p.add_argument(
        "--captcha",
        default="",
        help="If api/sharedownload requires captcha: provide 4-char code here (image saved under work_dirs/sanity/baidu_vcode.png).",
    )
    p.add_argument(
        "--vcode-str",
        default="",
        help="Captcha token (vcode_str). Leave empty to auto-fetch when needed.",
    )
    args = p.parse_args()

    out_root = Path(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    wanted = {m.lower(): m for m in models}

    client = BaiduShareClient(args.share_url, args.pwd)
    try:
        dirs = client.top_level_dirs()
    except DownloadError as e:
        _err(str(e))
        return 2

    # Map share folder names -> local folder names
    share_name_map = {
        "rn50": "RN50",
        "vit-b-32": "ViT-B-32",
    }
    download_plan: list[tuple[str, str, list[str]]] = []
    for share_name, local_name in share_name_map.items():
        if share_name not in wanted and local_name.lower() not in wanted:
            continue
        # Find share dir by exact key (case-sensitive in share listing)
        share_dir = None
        for k, v in dirs.items():
            if k.lower() == share_name:
                share_dir = v
                break
        if not share_dir:
            raise DownloadError(f"Share directory not found: {share_name} (available: {sorted(dirs)[:10]}...)")
        if share_name == "rn50":
            filenames = ["rn50_model.safetensors", "merges.txt", "vocab.json"]
        else:
            filenames = ["vit_b_32_model.safetensors", "merges.txt", "vocab.json"]
        download_plan.append((share_dir, local_name, filenames))

    for share_dir, local_dir, filenames in download_plan:
        items = client.list_dir(share_dir)
        by_name = {it.get("server_filename"): it for it in items}
        fs_ids: list[int] = []
        for fn in filenames:
            if fn not in by_name:
                raise DownloadError(f"File not found in share dir {share_dir}: {fn}")
            fs_id = int(by_name[fn]["fs_id"])
            fs_ids.append(fs_id)

        vcode_input = args.captcha.strip() or None
        vcode_str = args.vcode_str.strip() or None
        try:
            dlink = client.get_batch_dlink(
                fs_ids,
                vcode_input=vcode_input,
                vcode_str=vcode_str,
                out_dir=Path("work_dirs/sanity"),
            )
        except CaptchaRequired as cap:
            _info(f"Captcha required. Image saved to: {cap.image_path}")
            _info(f"Open it and re-run with: --captcha <code> --vcode-str '{cap.vcode}'")
            return 3

        zip_path = Path("work_dirs/download_cache") / "sarclip" / f"{local_dir}.zip"
        _download_stream(client.session, dlink, zip_path)
        _safe_extract_zip(zip_path, out_root / local_dir, allowed_basenames=set(filenames))
        zip_path.unlink(missing_ok=True)

    _info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

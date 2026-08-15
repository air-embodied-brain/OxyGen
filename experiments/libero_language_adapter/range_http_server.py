"""Serve review assets with HTTP byte-range support."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
import re


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            return super().send_head()
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        file = path.open("rb")
        size = path.stat().st_size
        start = 0
        end = size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                file.close()
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None
            start_text, end_text = match.groups()
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else end
            elif end_text:
                start = max(0, size - int(end_text))
            if start >= size or start > end:
                file.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            end = min(end, size - 1)

        self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        file.seek(start)
        self._range_bytes = end - start + 1
        return file

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_range_bytes", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining:
            chunk = source.read(min(256 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    handler = lambda *handler_args, **kwargs: RangeRequestHandler(  # noqa: E731
        *handler_args, directory=str(args.directory), **kwargs
    )
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving {args.directory} on http://{args.bind}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

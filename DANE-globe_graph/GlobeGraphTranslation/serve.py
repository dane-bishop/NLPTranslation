import http.server
import os


def main() -> None:
    base_port = int(os.environ.get("PORT", "8000"))

    handler = http.server.SimpleHTTPRequestHandler

    httpd = None
    port = base_port
    last_err = None
    for candidate in range(base_port, base_port + 20):
        try:
            httpd = http.server.ThreadingHTTPServer(("localhost", candidate), handler)
            port = candidate
            break
        except OSError as err:
            last_err = err

    if httpd is None:
        raise last_err or RuntimeError("Failed to bind local server")

    url = f"http://localhost:{port}/viewer.html"

    print(f"Serving from {os.getcwd()}", flush=True)
    print(f"Open {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()


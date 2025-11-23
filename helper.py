# helper.py
def get_embedding(prompt: str, model: str = "nomic-embed-text"):
    """
    Fetch a 768-dim embedding from Ollama (http://localhost:11434).
    Returns a Python list[float].
    """
    import requests
    url = "http://localhost:11434/api/embeddings"
    data = {"model": model, "prompt": prompt}
    resp = requests.post(url, json=data, timeout=60)
    resp.raise_for_status()
    emb = resp.json().get("embedding", None)
    if emb is None:
        raise RuntimeError("Ollama embedding API returned no 'embedding' field")
    # Ensure float32-compatible list
    return [float(x) for x in emb]

_OPENSEARCH_CLIENTS: dict[tuple, "OpenSearch"] = {}


def get_opensearch_client(
    host: str = "localhost",
    port: int = 9200,
    use_ssl: bool = False,
    username: str | None = None,
    password: str | None = None,
):
    """
    Build an OpenSearch client. Defaults to http://localhost:9200 with no auth.
    """
    from opensearchpy import OpenSearch

    cache_key = (host, port, use_ssl, username, password)
    cached = _OPENSEARCH_CLIENTS.get(cache_key)
    if cached is not None:
        return cached

    kwargs = {
        "hosts": [{"host": host, "port": port}],
        "http_compress": True,
        "timeout": 60,
        "max_retries": 3,
        "retry_on_timeout": True,
    }
    if use_ssl:
        kwargs.update({"use_ssl": True, "verify_certs": False, "ssl_show_warn": False})
    if username and password:
        kwargs["http_auth"] = (username, password)

    client: OpenSearch = OpenSearch(**kwargs)
    if not client.ping():
        raise RuntimeError(f"Cannot reach OpenSearch at {host}:{port}")
    print("Connected to OpenSearch")
    _OPENSEARCH_CLIENTS[cache_key] = client
    return client

if __name__ == "__main__":
    get_opensearch_client("localhost", 9200)

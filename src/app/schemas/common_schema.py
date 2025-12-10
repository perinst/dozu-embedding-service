"""Common schemas used across multiple features."""

from typing import List, Optional
from pydantic import BaseModel


class ProxyConfig(BaseModel):
    """Proxy configuration for external requests."""

    kind: Optional[str] = None  # 'webshare' or 'generic'
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    filter_ip_locations: Optional[List[str]] = None
    http_url: Optional[str] = None
    https_url: Optional[str] = None

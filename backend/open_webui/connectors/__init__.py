"""
Connector type registry.

Maps connector_type strings (stored in DataConnector.connector_type) to their
BaseConnector subclass. When a new connector is added, register it here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_webui.connectors.base import BaseConnector

# Lazy imports to avoid circular dependencies and heavy imports at startup.
# Each value is a tuple of (module_path, class_name).
_CONNECTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "k4mi": ("open_webui.connectors.k4mi", "K4miConnector"),
    "invoice_db": ("open_webui.connectors.invoice_db", "InvoiceDBConnector"),
}


def get_connector_class(connector_type: str) -> type["BaseConnector"]:
    """Get the connector class for a given type string. Raises KeyError if unknown."""
    if connector_type not in _CONNECTOR_REGISTRY:
        raise KeyError(
            f"Unknown connector type: {connector_type!r}. "
            f"Available: {list(_CONNECTOR_REGISTRY.keys())}"
        )
    module_path, class_name = _CONNECTOR_REGISTRY[connector_type]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_available_types() -> list[dict]:
    """Return metadata about all registered connector types (for the /types endpoint)."""
    result = []
    for type_key, (module_path, class_name) in _CONNECTOR_REGISTRY.items():
        try:
            cls = get_connector_class(type_key)
            result.append({
                "type": type_key,
                "label": cls.connector_label(),
                "config_schema": cls.config_schema(),
            })
        except Exception:
            result.append({
                "type": type_key,
                "label": type_key,
                "config_schema": {},
            })
    return result

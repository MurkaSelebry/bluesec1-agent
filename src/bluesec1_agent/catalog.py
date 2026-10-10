"""Validate tool arguments against the current runtime's JSON schemas offline."""

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry

from bluesec1_agent.models import (
    FinishInvestigationRequest,
    GetADObjectInfoRequest,
    GetEntityRequest,
    GetRelationRequest,
    SearchRequest,
)


class CatalogCompatibilityError(ValueError):
    """The runtime catalog cannot be used by the fixed local tool contract."""


_MODELS = {
    "search": SearchRequest,
    "get_entity": GetEntityRequest,
    "get_relation": GetRelationRequest,
    "get_ad_object_info": GetADObjectInfoRequest,
    "finish_investigation": FinishInvestigationRequest,
}


def _validator(schema: dict[str, Any]) -> Draft202012Validator:
    """Reject unresolved/external references instead of ever retrieving a URL."""
    pending: list[Any] = [schema]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            reference = value.get("$ref")
            if reference is not None:
                if not isinstance(reference, str) or not reference.startswith("#/"):
                    raise CatalogCompatibilityError("Unsupported schema reference.")
                target: Any = schema
                try:
                    for part in reference[2:].split("/"):
                        target = target[part.replace("~1", "/").replace("~0", "~")]
                except (KeyError, TypeError) as exc:
                    raise CatalogCompatibilityError("Unresolved schema reference.") from exc
            pending.extend(value.values())
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise CatalogCompatibilityError("Invalid runtime argument schema.") from exc
    return Draft202012Validator(schema, registry=Registry())


def check_catalog(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return validated schemas, rejecting unknown mandatory top-level arguments."""
    schemas: dict[str, dict[str, Any]] = {}
    for tool in catalog:
        name = tool.get("name")
        if not isinstance(name, str):
            raise CatalogCompatibilityError("A runtime tool has no string name.")
        if name in schemas:
            raise CatalogCompatibilityError(f"Duplicate runtime tool: {name}")
        schema = tool.get("parameters", {})
        if not isinstance(schema, dict):
            raise CatalogCompatibilityError(f"Invalid argument schema for {name}.")
        _validator(schema)
        model = _MODELS.get(name)
        if model is not None:
            known = {
                field.alias or key for key, field in model.model_fields.items() if key != "tool"
            }
            unknown = set(schema.get("required", [])) - known
            if unknown:
                raise CatalogCompatibilityError(
                    f"Unsupported required arguments for {name}: {sorted(unknown)}"
                )
        schemas[name] = schema
    return schemas


def validate_arguments(arguments: Any, schema: dict[str, Any]) -> list[str]:
    """Validate nested arguments without fetching references or echoing their values."""
    errors = list(_validator(schema).iter_errors(arguments))
    return [
        f"{'.'.join(map(str, error.absolute_path)) or '$'}: "
        f"{error.validator} constraint failed"
        + (f" ({','.join(error.validator_value)})" if error.validator == "required" else "")
        for error in errors[:20]
    ]

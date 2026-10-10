"""Validate tool arguments against the current runtime's JSON schemas offline."""

from itertools import islice
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


def _resolve(reference: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        raise CatalogCompatibilityError("Unsupported schema reference.")
    target: Any = schema
    try:
        for part in reference[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
    except (KeyError, TypeError) as exc:
        raise CatalogCompatibilityError("Unresolved schema reference.") from exc
    return target


def _validator(schema: dict[str, Any]) -> Draft202012Validator:
    """Use an explicitly bounded, acyclic local-reference schema subset."""
    remaining = 20_000

    def inspect(value: Any, ancestors: frozenset[int] = frozenset()) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or len(ancestors) > 64:
            raise CatalogCompatibilityError("Runtime schema exceeds complexity limits.")
        if not isinstance(value, (dict, list)):
            return
        if id(value) in ancestors:
            raise CatalogCompatibilityError("Recursive schema references are unsupported.")
        ancestors = ancestors | {id(value)}
        if isinstance(value, dict):
            if value.keys() & {"$dynamicRef", "$recursiveRef", "$id", "$anchor", "$dynamicAnchor"}:
                raise CatalogCompatibilityError("Unsupported schema reference scope.")
            if "$ref" in value:
                inspect(_resolve(value["$ref"], schema), ancestors)
            children = value.values()
        else:
            children = value
        for child in children:
            inspect(child, ancestors)

    inspect(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except (SchemaError, RecursionError) as exc:
        raise CatalogCompatibilityError("Invalid runtime argument schema.") from exc
    return Draft202012Validator(schema, registry=Registry())


def _required_fields(value: Any, root: dict[str, Any]) -> set[str]:
    """Requirements at this instance level; never merge nested object fields."""
    if not isinstance(value, dict):
        return set()
    required = set(value.get("required", []))
    if "$ref" in value:
        required |= _required_fields(_resolve(value["$ref"], root), root)
    for branch in value.get("allOf", []):
        required |= _required_fields(branch, root)
    for keyword in ("anyOf", "oneOf"):
        alternatives = [_required_fields(branch, root) for branch in value.get(keyword, [])]
        if alternatives:
            required |= set.intersection(*alternatives)
    if "then" in value and "else" in value:
        required |= _required_fields(value["then"], root) & _required_fields(value["else"], root)
    return required


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
            unknown = _required_fields(schema, schema) - known
            if unknown:
                raise CatalogCompatibilityError(
                    f"Unsupported required arguments for {name}: {sorted(unknown)}"
                )
        schemas[name] = schema
    return schemas


def validate_arguments(arguments: Any, schema: dict[str, Any]) -> list[str]:
    """Validate nested arguments without fetching references or echoing their values."""
    errors = list(islice(_validator(schema).iter_errors(arguments), 20))
    return [
        f"{'.'.join(map(str, error.absolute_path)) or '$'}: "
        f"{error.validator} constraint failed"
        + (f" ({','.join(error.validator_value)})" if error.validator == "required" else "")
        for error in errors[:20]
    ]

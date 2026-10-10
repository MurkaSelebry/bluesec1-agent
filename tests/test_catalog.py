import pytest

from bluesec1_agent.catalog import CatalogCompatibilityError, check_catalog, validate_arguments


def test_unknown_required_argument_is_rejected_before_model_or_rpc():
    catalog = [
        {"name": "finish_investigation", "parameters": {"type": "object"}},
        {
            "name": "get_entity",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}, "new_required": {}},
                "required": ["entity_id", "new_required"],
            },
        },
    ]
    with pytest.raises(CatalogCompatibilityError, match="new_required"):
        check_catalog(catalog)


def test_arguments_follow_nested_required_fields_and_namespace_schema():
    schema = {
        "type": "object",
        "properties": {"submission": {"$ref": "#/$defs/submission"}},
        "required": ["submission"],
        "$defs": {
            "submission": {
                "type": "object",
                "properties": {"verdict": {"enum": ["benign", "malicious"]}, "new_field": {}},
                "required": ["verdict", "new_field"],
            }
        },
    }
    errors = validate_arguments({"submission": {"verdict": "benign"}}, schema)
    assert errors and any("new_field" in error for error in errors)


def test_schema_branch_selection_does_not_merge_benign_and_malicious_requirements():
    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {"verdict": {"const": "benign"}},
                "required": ["verdict", "legitimacy_evidence"],
            },
            {
                "type": "object",
                "properties": {"verdict": {"const": "malicious"}},
                "required": ["verdict", "ir_artifacts"],
            },
        ]
    }
    assert validate_arguments({"verdict": "malicious", "ir_artifacts": []}, schema) == []


def test_false_is_not_a_json_integer():
    assert validate_arguments(False, {"type": "integer"})


def test_unknown_local_reference_is_a_compatibility_error():
    with pytest.raises(CatalogCompatibilityError, match="reference"):
        validate_arguments({}, {"$ref": "#/$defs/missing"})


def test_unknown_required_argument_hidden_in_allof_is_rejected():
    with pytest.raises(CatalogCompatibilityError, match="new_required"):
        check_catalog(
            [
                {
                    "name": "get_entity",
                    "parameters": {
                        "allOf": [{"required": ["entity_id"]}, {"required": ["new_required"]}],
                    },
                }
            ]
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"$dynamicRef": "https://example.invalid/schema"},
        {"$ref": "#/$defs/loop", "$defs": {"loop": {"$ref": "#/$defs/loop"}}},
        {"$id": "https://example.invalid/schema", "type": "object"},
    ],
)
def test_unsupported_or_recursive_schema_fails_during_catalog_preflight(schema):
    with pytest.raises(CatalogCompatibilityError):
        check_catalog([{"name": "get_entity", "parameters": schema}])

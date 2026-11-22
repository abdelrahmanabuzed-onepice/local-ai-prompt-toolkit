"""Versioned prompt template repository and parameter interpolation manager."""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


class PromptTemplateVault:
    """In-memory and file-backed prompt template registry."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.templates: Dict[str, Dict[str, Any]] = {}
        if storage_path and os.path.isfile(storage_path):
            with open(storage_path, "r", encoding="utf-8") as f:
                self.templates = json.load(f)

    def register(
        self,
        name: str,
        template: str,
        version: str = "1.0.0",
        description: str = "",
        required_params: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register a new template or revision."""
        # Find parameters defined as {{param_name}}
        inferred = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", template)
        req_params = required_params or sorted(list(set(inferred)))

        entry = {
            "name": name,
            "version": version,
            "description": description,
            "template": template,
            "required_params": req_params,
        }
        key = f"{name}@{version}"
        self.templates[key] = entry

        if self.storage_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)) or ".", exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, indent=2)

        return entry

    def render(self, name: str, version: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
        """Interpolate parameters into the specified prompt template."""
        params = params or {}
        # Find template
        matching = [v for k, v in self.templates.items() if v["name"] == name]
        if not matching:
            raise KeyError(f"Template '{name}' not found in vault.")

        if version:
            selected = next((m for m in matching if m["version"] == version), None)
            if not selected:
                raise KeyError(f"Version '{version}' for template '{name}' not found.")
        else:
            # Pick latest version
            selected = matching[-1]

        # Validate required parameters
        missing = [p for p in selected["required_params"] if p not in params]
        if missing:
            raise ValueError(f"Missing required template parameter(s): {missing}")

        rendered = selected["template"]
        for k, v in params.items():
            rendered = rendered.replace(f"{{{{{k}}}}}", str(v))

        return rendered


def execute_prompt_template_vault(
    action: str,
    vault_path: str,
    name: str,
    version: str = "1.0.0",
    template_text: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Manage and render versioned prompt templates.

    Args:
        action: Vault operation ('register', 'render', 'list').
        vault_path: Path to JSON storage vault file.
        name: Template name identifier.
        version: Semantic version string.
        template_text: Template string with {{params}} for registration.
        params: Key-value parameters for rendering.
        output_path: Optional destination to write JSON result.

    Returns:
        Structured dictionary reporting operation result.
    """
    vault = PromptTemplateVault(storage_path=vault_path)

    if action == "register":
        if not template_text:
            raise ValueError("Template text is required for registration.")
        reg = vault.register(name=name, template=template_text, version=version)
        result = {
            "status": "success",
            "action": "register",
            "template": reg,
        }

    elif action == "render":
        rendered = vault.render(name=name, version=version, params=params or {})
        result = {
            "status": "success",
            "action": "render",
            "name": name,
            "version": version,
            "rendered_prompt": rendered,
        }

    elif action == "list":
        result = {
            "status": "success",
            "action": "list",
            "total_templates": len(vault.templates),
            "templates": list(vault.templates.values()),
        }
    else:
        raise ValueError(f"Unknown action: {action}. Choose 'register', 'render', or 'list'.")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Versioned prompt template repository and parameter interpolation manager."
    )
    parser.add_argument(
        "--action",
        "-a",
        choices=["register", "render", "list"],
        required=True,
        help="Vault action",
    )
    parser.add_argument("--vault", "-v", default="prompt_vault.json", help="Path to vault JSON file")
    parser.add_argument("--name", "-n", default="default_prompt", help="Template name")
    parser.add_argument("--version", default="1.0.0", help="Template version")
    parser.add_argument("--template", "-t", help="Template content string")
    parser.add_argument("--params-json", "-p", help="JSON dictionary of parameters for rendering")
    parser.add_argument("--output", "-o", help="Path to write JSON output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        p_dict = json.loads(args.params_json) if args.params_json else {}
        res = execute_prompt_template_vault(
            action=args.action,
            vault_path=args.vault,
            name=args.name,
            version=args.version,
            template_text=args.template,
            params=p_dict,
            output_path=args.output,
        )
        if args.action == "render":
            print(f"Rendered Prompt:\n{res['rendered_prompt']}")
        else:
            print(f"Action '{res['action']}' completed successfully")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

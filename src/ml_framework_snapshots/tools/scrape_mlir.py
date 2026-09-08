"""Script to generate an exhaustive JSON dump of MLIR operations.

Extracts operations across core ML dialects (arith, math, tensor, linalg, scf, func,
memref, gpu, vector) via TableGen ODS parsing, live Python dialect inspection,
or official MLIR documentation.
"""

import importlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import urllib.request
import warnings

MLIR_DOCS_URL = "https://mlir.llvm.org/docs/Dialects/"
CORE_MLIR_DIALECTS: List[str] = [
    "arith",
    "math",
    "tensor",
    "linalg",
    "scf",
    "func",
    "memref",
    "gpu",
    "vector",
    "llvm",
    "nvvm",
    "rocdl",
]


def fetch_html(url: str) -> str:  # pragma: no cover
    """Fetch HTML content from a URL.

    Args:
        url: The URL to fetch.

    Returns:
        The decoded HTML string content, or empty string on error.
    """
    try:
        with urllib.request.urlopen(url) as response:
            return str(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def parse_table(html_chunk: str, expected_cols: int) -> List[str]:
    """Parse an HTML table to extract column contents.

    .. deprecated:: 0.2.0
        HTML doc scraping is deprecated in favor of TableGen ODS parsing.

    Args:
        html_chunk: Raw HTML snippet containing the table.
        expected_cols: Expected minimum column count.

    Returns:
        List of extracted column strings.
    """
    warnings.warn(
        "HTML doc scraping is deprecated in favor of TableGen ODS parsing.",
        DeprecationWarning,
        stacklevel=2,
    )
    results = []
    rows = re.findall(r"<tr>(.*?)</tr>", html_chunk, re.DOTALL)
    for row in rows:
        cells = re.findall(r"<td>(.*?)</td>", row, re.DOTALL)
        if len(cells) >= 1:
            name_cell = cells[0]
            clean_name = re.sub(r"<[^>]+>", "", name_cell).strip()
            if clean_name and not clean_name.startswith("&"):
                results.append(clean_name)
    return results


def parse_dialect_page(url: str, dialect_name: str) -> List[Dict[str, Any]]:
    """Parse an MLIR dialect doc page for its operations.

    .. deprecated:: 0.2.0
        HTML doc scraping is deprecated in favor of TableGen ODS parsing.

    Args:
        url: The documentation URL to fetch.
        dialect_name: The dialect prefix name.

    Returns:
        List of dictionaries detailing parsed dialect operations.
    """
    warnings.warn(
        "HTML doc scraping is deprecated in favor of TableGen ODS parsing.",
        DeprecationWarning,
        stacklevel=2,
    )
    html = fetch_html(url)
    if not html:  # pragma: no cover
        return []

    ops = []
    sections = re.split(r'<h3 id="([^"]+)">', html)
    if len(sections) <= 1:
        return []

    for i in range(1, len(sections), 2):
        sec_content = sections[i + 1]

        header_match = re.search(
            r"<code>([a-zA-Z0-9_\.]+)</code>\s*\(([^)]+)\)", sec_content
        )
        if not header_match:
            continue

        api_path = header_match.group(1)
        class_name = header_match.group(2).split("::")[-1]

        operands: List[str] = []
        attributes: List[str] = []

        op_match = re.search(
            r"<h4>Operands:</h4>\s*<table>(.*?)</table>", sec_content, re.DOTALL
        )
        if op_match:
            operands = parse_table(op_match.group(1), 2)

        attr_match = re.search(
            r"<h4>Attributes:</h4>\s*<table>(.*?)</table>",
            sec_content,
            re.DOTALL,
        )
        if attr_match:
            attributes = parse_table(attr_match.group(1), 2)

        desc_match = re.search(r"<p><em>(.*?)</em></p>", sec_content, re.DOTALL)
        description = None
        if desc_match:
            description = re.sub(r"<[^>]+>", "", desc_match.group(1)).strip()

        ops.append(
            {
                "api_path": api_path,
                "dialect": dialect_name,
                "class_name": class_name,
                "operands": operands,
                "attributes": attributes,
                "description": description,
            }
        )

    return ops


def strip_tablegen_comments(text: str) -> str:
    """Strip block and line comments from TableGen source cleanly.

    Args:
        text: Raw TableGen source code.

    Returns:
        Source code with comments removed.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


class TableGenASTParser:
    """TableGen AST parser resolving multiclasses, defm instances, foreach loops, and concatenations."""

    def __init__(self, content: str) -> None:
        """Initialize TableGen AST parser with source content.

        Args:
            content: Raw TableGen source string.
        """
        self.raw_content = content
        self.content = strip_tablegen_comments(content)
        self.multiclasses: Dict[str, Dict[str, Any]] = {}

    def _extract_multiclasses(self, text: str) -> str:
        """Extract multiclass definitions and replace them with whitespace.

        Args:
            text: TableGen source code.

        Returns:
            TableGen source with multiclasses extracted into internal registry.
        """
        pattern = re.compile(
            r"\bmulticlass\s+([A-Za-z0-9_]+)(?:<([^>]*)>)?\s*\{", re.DOTALL
        )
        pos = 0
        chunks: List[str] = []
        while True:
            m = pattern.search(text, pos)
            if not m:
                chunks.append(text[pos:])
                break
            chunks.append(text[pos : m.start()])
            name = m.group(1)
            raw_params = m.group(2) or ""
            params = [p.strip() for p in raw_params.split(",") if p.strip()]

            start_idx = m.end()
            depth = 1
            i = start_idx
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            body = text[start_idx : i - 1]
            self.multiclasses[name] = {"params": params, "body": body}
            pos = i
        return "".join(chunks)

    def _expand_foreach(self, text: str) -> str:
        """Expand foreach loops into unrolled blocks.

        Args:
            text: TableGen source code.

        Returns:
            TableGen source with foreach loops unrolled.
        """
        pattern = re.compile(
            r"\bforeach\s+([A-Za-z0-9_]+)\s*=\s*\[(.*?)\]\s*in\s*\{", re.DOTALL
        )
        pos = 0
        chunks: List[str] = []
        while True:
            m = pattern.search(text, pos)
            if not m:
                chunks.append(text[pos:])
                break
            chunks.append(text[pos : m.start()])
            var_name = m.group(1)
            raw_items = m.group(2)
            items = [
                item.strip().strip('"').strip("'")
                for item in raw_items.split(",")
                if item.strip()
            ]

            start_idx = m.end()
            depth = 1
            i = start_idx
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            body = text[start_idx : i - 1]

            unrolled_parts: List[str] = []
            for item in items:
                sub_body = re.sub(rf"\b{var_name}\b", item, body)
                unrolled_parts.append(sub_body)
            chunks.append("\n".join(unrolled_parts))
            pos = i
        return "".join(chunks)

    def _expand_defm(self, text: str) -> str:
        """Expand defm declarations into instantiated def blocks.

        Args:
            text: TableGen source code.

        Returns:
            TableGen source with defm declarations instantiated.
        """
        pattern = re.compile(
            r'\bdefm\s+([A-Za-z0-9_"]*)\s*:\s*([A-Za-z0-9_]+)(?:<([^>]*)>)?\s*;',
            re.DOTALL,
        )
        pos = 0
        chunks: List[str] = []
        while True:
            m = pattern.search(text, pos)
            if not m:
                chunks.append(text[pos:])
                break
            chunks.append(text[pos : m.start()])
            inst_name = (m.group(1) or "").strip().strip('"')
            mc_name = m.group(2)
            raw_args = m.group(3) or ""
            args = [
                a.strip().strip('"').strip("'")
                for a in raw_args.split(",")
                if a.strip()
            ]

            if mc_name in self.multiclasses:
                mc_info = self.multiclasses[mc_name]
                mc_body = mc_info["body"]
                mc_params = mc_info["params"]

                inst_body = mc_body
                for p_idx, p_name in enumerate(mc_params):
                    clean_p = p_name.split()[-1]
                    val = args[p_idx] if p_idx < len(args) else ""
                    inst_body = re.sub(rf"\b{clean_p}\b", val, inst_body)

                inst_body = re.sub(r"\bNAME\b", inst_name, inst_body)
                chunks.append(inst_body)
            else:
                chunks.append(m.group(0))
            pos = m.end()
        return "".join(chunks)

    def _resolve_concatenations(self, text: str) -> str:
        """Resolve '#' string concatenation operators.

        Args:
            text: TableGen source code.

        Returns:
            TableGen source with string concatenations resolved.
        """
        prev = None
        curr = text
        while prev != curr:
            prev = curr
            curr = re.sub(r'"([^"]*)"\s*#\s*"([^"]*)"', r'"\1\2"', curr)
            curr = re.sub(
                r'([A-Za-z0-9_]+)\s*#\s*"([^"]*)"',
                lambda m: str(m.group(1)) + str(m.group(2)),
                curr,
            )
            curr = re.sub(
                r'"([^"]*)"\s*#\s*([A-Za-z0-9_]+)',
                lambda m: str(m.group(1)) + str(m.group(2)),
                curr,
            )
            curr = re.sub(
                r"([A-Za-z0-9_]+)\s*#\s*([A-Za-z0-9_]+)",
                lambda m: str(m.group(1)) + str(m.group(2)),
                curr,
            )
        return curr

    def expand(self) -> str:
        """Expand all multiclasses, defm instances, foreach loops, and concatenations.

        Returns:
            Fully expanded TableGen source code.
        """
        expanded = self._extract_multiclasses(self.content)
        expanded = self._expand_foreach(expanded)
        expanded = self._resolve_concatenations(expanded)
        expanded = self._expand_defm(expanded)
        expanded = self._resolve_concatenations(expanded)
        expanded = re.sub(r'\bdef\s+"([^"]+)"', r"def \1", expanded)
        return expanded


def parse_tablegen_args(
    args_str: Optional[str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Parse TableGen (ins ...) argument declarations into operands and attributes.

    Args:
        args_str: The inner argument string from (ins ...).

    Returns:
        Tuple of (operands list, attributes list).
    """
    operands: List[Dict[str, str]] = []
    attributes: List[Dict[str, str]] = []
    if not args_str:
        return operands, attributes

    depth = 0
    parts: List[str] = []
    curr: List[str] = []
    for c in args_str:
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(curr).strip())
            curr = []
            continue
        curr.append(c)
    if curr:
        parts.append("".join(curr).strip())

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if ":$" in p:
            atype, aname = p.split(":$", 1)
        elif ":" in p:
            atype, aname = p.split(":", 1)
            aname = aname.lstrip("$").strip()
        else:
            atype, aname = p, f"arg{len(operands) + len(attributes)}"

        atype = atype.strip()
        aname = aname.strip()
        if not aname:
            aname = f"arg{len(operands) + len(attributes)}"

        is_attr = any(
            k in atype.lower()
            for k in (
                "attr",
                "enum",
                "constant",
                "dictionary",
                "string",
                "bool",
                "i64",
                "i32",
                "f32",
                "dimension",
            )
        )
        if is_attr:
            attributes.append({"name": aname, "type": atype})
        else:
            operands.append({"name": aname, "type": atype})

    return operands, attributes


def parse_tablegen_results(res_str: Optional[str]) -> List[Dict[str, str]]:
    """Parse TableGen (outs ...) result declarations into structured results.

    Args:
        res_str: The inner result string from (outs ...).

    Returns:
        List of result dictionaries containing name and type.
    """
    results: List[Dict[str, str]] = []
    if not res_str:
        return results

    depth = 0
    parts: List[str] = []
    curr: List[str] = []
    for c in res_str:
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(curr).strip())
            curr = []
            continue
        curr.append(c)
    if curr:
        parts.append("".join(curr).strip())

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if ":$" in p:
            rtype, rname = p.split(":$", 1)
        elif ":" in p:
            rtype, rname = p.split(":", 1)
            rname = rname.lstrip("$").strip()
        else:
            rtype, rname = p, "result"
        results.append({"name": rname.strip() or "result", "type": rtype.strip()})

    return results


def parse_tablegen_regions(reg_str: Optional[str]) -> List[str]:
    """Parse TableGen (region ...) declarations into region names.

    Args:
        reg_str: The inner region string from (region ...).

    Returns:
        List of region name strings.
    """
    regions: List[str] = []
    if not reg_str:
        return regions

    for p in reg_str.split(","):
        p = p.strip()
        if not p:
            continue
        if ":$" in p:
            rname = p.split(":$")[1].strip()
        elif ":" in p:
            rname = p.split(":")[1].lstrip("$").strip()
        else:
            rname = p
        regions.append(rname)

    return regions


def parse_mlir_tablegen(content: str, dialect_name: str) -> List[Dict[str, Any]]:
    """Parse MLIR TableGen / ODS definitions into structured operations.

    Captures:
        - SSA Operands with types and variadic/optional attributes
        - Buildable attributes distinguished from runtime SSA values
        - SSA Return types and multi-result signatures
        - Region and Block argument structures (linalg.generic, scf.for, scf.while)
        - Dialect verification traits

    Args:
        content: Raw TableGen file content.
        dialect_name: Name of the dialect (e.g. 'arith', 'linalg').

    Returns:
        List of structured operation dictionaries.
    """
    clean_content = TableGenASTParser(content).expand()
    ops: List[Dict[str, Any]] = []

    # 1. First pass: extract all TableGen class definitions and their superclasses
    classes: Dict[str, Dict[str, Any]] = {}
    for m in re.finditer(
        r"\bclass\s+([A-Za-z0-9_]+)(?:<.*?>)?\s*(?::\s*([^{;]+))?\s*(\{|;)",
        clean_content,
    ):
        cname = m.group(1)
        supers_raw = m.group(2) or ""
        term = m.group(3)
        body = ""
        if term == "{":
            start = m.end()
            depth = 1
            pos = start
            while pos < len(clean_content) and depth > 0:
                if clean_content[pos] == "{":
                    depth += 1
                elif clean_content[pos] == "}":
                    depth -= 1
                pos += 1
            body = clean_content[start : pos - 1]

        args_m = re.search(
            r"let\s+arguments\s*=\s*\(ins(.*?)\);", body, re.DOTALL
        ) or re.search(r"Arguments<\s*\(ins(.*?)\)>", supers_raw, re.DOTALL)
        res_m = re.search(
            r"let\s+results\s*=\s*\(outs(.*?)\);", body, re.DOTALL
        ) or re.search(r"Results<\s*\(outs(.*?)\)>", supers_raw, re.DOTALL)
        reg_m = re.search(r"let\s+regions\s*=\s*\(region(.*?)\);", body, re.DOTALL)

        supers = [
            s.split("<")[0].strip() for s in re.split(r",", supers_raw) if s.strip()
        ]
        classes[cname] = {
            "args": args_m.group(1).strip() if args_m else None,
            "res": res_m.group(1).strip() if res_m else None,
            "reg": reg_m.group(1).strip() if reg_m else None,
            "supers": supers,
        }

    def resolve_field(cls_name: str, field_name: str) -> Optional[str]:
        """Recursively resolve inherited TableGen fields from base classes.

        Args:
            cls_name: Name of the class to resolve.
            field_name: Field key to lookup ('args', 'res', 'reg').

        Returns:
            Resolved field string value or None.
        """
        if cls_name not in classes:
            return None
        val = classes[cls_name].get(field_name)
        if val is not None:
            return str(val)
        for s in classes[cls_name]["supers"]:
            res = resolve_field(s, field_name)
            if res is not None:
                return str(res)
        return None

    # 2. Second pass: extract all def declarations
    pattern = re.compile(
        r"\bdef\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)(?:<([^>]*)>)?(?:\s*,\s*([^{;]+))?\s*(\{|;)",
        re.DOTALL,
    )

    pos = 0
    while True:
        match = pattern.search(clean_content, pos)
        if not match:
            break
        raw_class = match.group(1)
        base_cls = match.group(2)
        targs = match.group(3) or ""
        extra_supers = match.group(4) or ""
        term = match.group(5)

        body = ""
        if term == "{":
            start_idx = match.end()
            depth = 1
            i = start_idx
            while i < len(clean_content) and depth > 0:
                if clean_content[i] == "{":
                    depth += 1
                elif clean_content[i] == "}":
                    depth -= 1
                i += 1
            body = clean_content[start_idx : i - 1]
            pos = i
        else:
            pos = match.end()

        clean_class = raw_class.split("_")[-1]
        class_name = (
            f"{clean_class}Op" if not clean_class.endswith("Op") else clean_class
        )

        # Extract operation mnemonic
        mnem_m = re.search(r'let\s+mnemonic\s*=\s*"([a-z0-9_.]+)"', body)
        if mnem_m:
            op_name = mnem_m.group(1)
        else:
            first_targ = (
                targs.split(",")[0].strip().strip('"') if targs else clean_class.lower()
            )
            if first_targ and not first_targ.startswith("["):
                op_name = first_targ
            else:
                op_name = clean_class.lower()

        # Extract traits
        traits: List[str] = []
        traits_match = re.search(r"\[(.*?)\]", targs)
        if traits_match:
            traits.extend(
                [
                    t.strip()
                    for t in traits_match.group(1).split(",")
                    if t.strip() and not t.strip().startswith("//")
                ]
            )
        body_traits = re.search(r"let\s+traits\s*=\s*\[(.*?)\];", body, re.DOTALL)
        if body_traits:
            traits.extend(
                [
                    t.strip()
                    for t in body_traits.group(1).split(",")
                    if t.strip() and not t.strip().startswith("//")
                ]
            )

        # Resolve arguments
        args_match = re.search(
            r"let\s+arguments\s*=\s*\(ins(.*?)\);", body, re.DOTALL
        ) or re.search(r"Arguments<\s*\(ins(.*?)\)>", extra_supers, re.DOTALL)
        args_str = args_match.group(1).strip() if args_match else None
        if not args_str:
            args_str = resolve_field(base_cls, "args")

        if not args_str and ("Binary" in base_cls or "Binary" in raw_class):
            args_str = "Type:$lhs, Type:$rhs"
        elif not args_str and ("Unary" in base_cls or "Unary" in raw_class):
            args_str = "Type:$in"
        elif not args_str and ("Cast" in base_cls or "Cast" in raw_class):
            args_str = "Type:$in"

        operands, attributes = parse_tablegen_args(args_str)

        # Resolve results
        results_match = re.search(
            r"let\s+results\s*=\s*\(outs(.*?)\);", body, re.DOTALL
        ) or re.search(r"Results<\s*\(outs(.*?)\)>", extra_supers, re.DOTALL)
        res_str = results_match.group(1).strip() if results_match else None
        if not res_str:
            res_str = resolve_field(base_cls, "res")

        if not res_str and (
            "Binary" in base_cls or "Unary" in base_cls or "Cast" in base_cls
        ):
            res_str = "Type:$result"

        results = parse_tablegen_results(res_str)

        # Resolve regions
        regions_match = re.search(
            r"let\s+regions\s*=\s*\(region(.*?)\);", body, re.DOTALL
        )
        reg_str = regions_match.group(1).strip() if regions_match else None
        if not reg_str:
            reg_str = resolve_field(base_cls, "reg")

        regions = parse_tablegen_regions(reg_str)

        summary_match = re.search(r'let\s+summary\s*=\s*"([^"]*)";', body)
        description = summary_match.group(1) if summary_match else None

        # Check for variadic operand and result segment sizing traits
        if (
            any("AttrSizedOperandSegments" in t for t in traits)
            or "AttrSizedOperandSegments" in base_cls
            or "AttrSizedOperandSegments" in extra_supers
        ):
            if not any(a.get("name") == "operand_segment_sizes" for a in attributes):
                attributes.append(
                    {"name": "operand_segment_sizes", "type": "DenseI32ArrayAttr"}
                )
            if "AttrSizedOperandSegments" not in traits:
                traits.append("AttrSizedOperandSegments")

        if (
            any("AttrSizedResultSegments" in t for t in traits)
            or "AttrSizedResultSegments" in base_cls
            or "AttrSizedResultSegments" in extra_supers
        ):
            if not any(a.get("name") == "result_segment_sizes" for a in attributes):
                attributes.append(
                    {"name": "result_segment_sizes", "type": "DenseI32ArrayAttr"}
                )
            if "AttrSizedResultSegments" not in traits:
                traits.append("AttrSizedResultSegments")

        ops.append(
            {
                "api_path": f"{dialect_name}.{op_name}",
                "dialect": dialect_name,
                "class_name": class_name,
                "operands": operands,
                "attributes": attributes,
                "results": results,
                "regions": regions,
                "traits": traits,
                "description": description,
            }
        )

    return ops


def parse_llvm_tblgen_json(
    json_data: Union[str, Dict[str, Any]], dialect_name: str
) -> List[Dict[str, Any]]:
    """Parse JSON dump from llvm-tblgen -dump-json into structured MLIR operations.

    Args:
        json_data: Raw JSON string or dictionary loaded from llvm-tblgen -dump-json.
        dialect_name: Target dialect identifier (e.g. 'arith', 'math').

    Returns:
        List of structured operation dictionaries conforming to snapshot schema.
    """
    if isinstance(json_data, str):
        data: Dict[str, Any] = json.loads(json_data)
    else:
        data = json_data

    ops: List[Dict[str, Any]] = []
    instance_of = data.get("!instanceof", {})
    op_def_names = set(instance_of.get("Op", []))

    for rec_name, rec in data.items():
        if rec_name.startswith("!") or not isinstance(rec, dict):
            continue

        superclasses = rec.get("!superclasses", [])
        is_op = (
            rec_name in op_def_names
            or "Op" in superclasses
            or any(str(s).endswith("_Op") or str(s) == "Op" for s in superclasses)
            or rec_name.endswith("Op")
            or "mnemonic" in rec
        )
        if not is_op:
            continue

        clean_class = rec_name.split("_")[-1]
        class_name = clean_class if clean_class.endswith("Op") else f"{clean_class}Op"

        mnemonic = rec.get("mnemonic")
        if not mnemonic:
            mnemonic = (
                clean_class[:-2].lower()
                if clean_class.endswith("Op")
                else clean_class.lower()
            )

        operands: List[Dict[str, str]] = []
        attributes: List[Dict[str, str]] = []

        raw_args = rec.get("arguments")
        if isinstance(raw_args, dict) and raw_args.get("kind") == "dag":
            for arg in raw_args.get("args", []):
                if isinstance(arg, (list, tuple)) and len(arg) >= 2:
                    arg_type_obj, arg_name = arg[0], str(arg[1])
                    arg_type = (
                        arg_type_obj.get("def", "Type")
                        if isinstance(arg_type_obj, dict)
                        else str(arg_type_obj)
                    )
                    is_attr = any(
                        k in arg_type.lower()
                        for k in (
                            "attr",
                            "enum",
                            "constant",
                            "string",
                            "bool",
                            "i64",
                            "i32",
                            "f32",
                            "dimension",
                        )
                    )
                    if is_attr:
                        attributes.append({"name": arg_name, "type": arg_type})
                    else:
                        operands.append({"name": arg_name, "type": arg_type})
        elif isinstance(raw_args, list):
            for arg in raw_args:
                aname = arg.get("name", "arg") if isinstance(arg, dict) else str(arg)
                atype = arg.get("type", "Type") if isinstance(arg, dict) else "Type"
                is_attr = any(
                    k in atype.lower()
                    for k in (
                        "attr",
                        "enum",
                        "constant",
                        "string",
                        "bool",
                        "i64",
                        "i32",
                        "f32",
                        "dimension",
                    )
                )
                if is_attr:
                    attributes.append({"name": aname, "type": atype})
                else:
                    operands.append({"name": aname, "type": atype})

        results: List[Dict[str, str]] = []
        raw_results = rec.get("results")
        if isinstance(raw_results, dict) and raw_results.get("kind") == "dag":
            for res in raw_results.get("args", []):
                if isinstance(res, (list, tuple)) and len(res) >= 2:
                    res_type_obj, res_name = res[0], str(res[1])
                    res_type = (
                        res_type_obj.get("def", "Type")
                        if isinstance(res_type_obj, dict)
                        else str(res_type_obj)
                    )
                    results.append({"name": res_name or "result", "type": res_type})
        elif isinstance(raw_results, list):
            for res in raw_results:
                rname = res.get("name", "result") if isinstance(res, dict) else "result"
                rtype = res.get("type", "Type") if isinstance(res, dict) else "Type"
                results.append({"name": rname, "type": rtype})

        regions: List[str] = []
        raw_regions = rec.get("regions")
        if isinstance(raw_regions, dict) and raw_regions.get("kind") == "dag":
            for reg in raw_regions.get("args", []):
                if isinstance(reg, (list, tuple)) and len(reg) >= 2:
                    regions.append(str(reg[1]))
        elif isinstance(raw_regions, list):
            regions.extend(str(r) for r in raw_regions)

        traits: List[str] = []
        raw_traits = rec.get("traits", [])
        if isinstance(raw_traits, list):
            for t in raw_traits:
                t_name = t.get("def") if isinstance(t, dict) else str(t)
                if t_name:
                    traits.append(t_name)

        summary = rec.get("summary")
        description = rec.get("description") or summary

        ops.append(
            {
                "api_path": f"{dialect_name}.{mnemonic}",
                "dialect": dialect_name,
                "class_name": class_name,
                "operands": operands,
                "attributes": attributes,
                "results": results,
                "regions": regions,
                "traits": traits,
                "description": description,
            }
        )

    return ops


def dump_ast_via_llvm_tooling(
    td_path: str, dialect_name: str, tool_binary: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """Dump AST using host llvm-tblgen or mlir-tblgen binary when available.

    Args:
        td_path: Path to the TableGen .td file.
        dialect_name: Dialect identifier (e.g. 'arith', 'math').
        tool_binary: Optional explicit path to TableGen executable.

    Returns:
        List of parsed operations if successful, or None if tooling unavailable or fails.
    """
    import shutil
    import subprocess

    binary = tool_binary or shutil.which("llvm-tblgen") or shutil.which("mlir-tblgen")
    if not binary:
        return None

    try:
        proc = subprocess.run(
            [binary, "--dump-json", td_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return parse_llvm_tblgen_json(proc.stdout, dialect_name)
    except Exception as e:
        print(f"Error running TableGen binary {binary}: {e}")
    return None


def validate_dialect_ops_ods(ops: List[Dict[str, Any]], dialect_name: str) -> List[str]:
    """Validate extracted dialect operations against official ODS structural rules.

    Args:
        ops: List of parsed operation dictionaries.
        dialect_name: Dialect identifier (e.g. 'arith', 'math').

    Returns:
        List of validation error descriptions.
    """
    errors: List[str] = []
    prefix = f"{dialect_name}."
    for op in ops:
        path = op.get("api_path", "")
        if not path.startswith(prefix):
            errors.append(
                f"Operation '{path}' does not match dialect prefix '{prefix}'."
            )
        if not op.get("class_name"):
            errors.append(f"Operation '{path}' missing class_name.")
        if "operands" not in op or not isinstance(op["operands"], list):
            errors.append(f"Operation '{path}' missing valid operands list.")
        if "attributes" not in op or not isinstance(op["attributes"], list):
            errors.append(f"Operation '{path}' missing valid attributes list.")
        if "results" not in op or not isinstance(op["results"], list):
            errors.append(f"Operation '{path}' missing valid results list.")
    return errors


def inspect_mlir_python_module(
    dialects: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Inspect installed mlir.dialects Python modules for operation classes.

    Args:
        dialects: Optional list of dialect names to inspect.

    Returns:
        List of extracted operation dictionaries.
    """
    target_dialects = dialects or CORE_MLIR_DIALECTS
    ops: List[Dict[str, Any]] = []

    for dialect in target_dialects:
        try:
            mod = importlib.import_module(f"mlir.dialects.{dialect}")
            for attr_name in dir(mod):
                if attr_name.endswith("Op") and not attr_name.startswith("_"):
                    cls = getattr(mod, attr_name)
                    op_mnemonic = getattr(
                        cls, "OPERATION_NAME", f"{dialect}.{attr_name[:-2].lower()}"
                    )
                    ops.append(
                        {
                            "api_path": str(op_mnemonic),
                            "dialect": dialect,
                            "class_name": attr_name,
                            "operands": [],
                            "attributes": [],
                            "description": cls.__doc__
                            or f"MLIR {attr_name} operation.",
                        }
                    )
        except ImportError:
            continue

    return ops


def scrape_stablehlo() -> List[Dict[str, Any]]:
    """Scrape the StableHLO spec markdown to extract operations.

    .. deprecated:: 0.2.0
        Use build_stablehlo_snapshot instead.

    Returns:
        List of scraped StableHLO operation dictionaries.
    """
    warnings.warn(
        "scrape_stablehlo is deprecated. Use build_stablehlo_snapshot instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"
    md = fetch_html(url)
    ops: List[Dict[str, Any]] = []
    if not md:  # pragma: no cover
        return ops

    sections = re.split(r"\n### (?![#])", md)
    for sec in sections[1:]:
        lines = sec.split("\n")
        op_name = lines[0].strip()
        if not re.match(r"^[a-z0-9_]+$", op_name) or op_name in (
            "functions",
            "types",
            "constants",
        ):
            continue

        operands: List[str] = []
        attributes: List[str] = []
        desc_lines = []
        current_header = ""
        for line in lines[1:]:
            if line.startswith("#### "):
                current_header = line[5:].strip().lower()
            elif current_header == "inputs":
                if line.startswith("|") and not line.startswith("|---"):
                    parts = [p.strip() for p in line.split("|")[1:-1]]
                    if len(parts) >= 2 and parts[0] != "Name" and parts[0] != "Label":
                        p_name = (
                            parts[1].strip("`")
                            if len(parts) >= 3 and parts[0].startswith("(")
                            else parts[0].strip("`")
                        )
                        if p_name:
                            operands.append(p_name)
            elif current_header == "semantics":
                if line.strip() and not line.startswith("#"):
                    desc_lines.append(line.strip())

        description = " ".join(desc_lines) if desc_lines else None
        class_name = "".join(part.capitalize() for part in op_name.split("_")) + "Op"

        ops.append(
            {
                "api_path": f"stablehlo.{op_name}",
                "dialect": "stablehlo",
                "class_name": class_name,
                "operands": operands,
                "attributes": attributes,
                "description": description,
            }
        )

    return ops


def main() -> None:
    """Run the scraper and output mlir_exhaustive.json."""
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "frameworks", "mlir_exhaustive.json"
    )

    all_ops: List[Dict[str, Any]] = []
    seen_paths: Set[str] = set()

    dialect_map = {
        "arith": ["ArithOps.td"],
        "math": ["MathOps.td"],
        "tensor": ["TensorOps.td"],
        "linalg": ["LinalgOps.td", "LinalgStructuredOps.td"],
        "scf": ["SCFOps.td"],
        "func": ["FuncOps.td"],
        "memref": ["MemRefOps.td"],
        "gpu": ["GPUOps.td"],
        "vector": ["VectorOps.td"],
        "llvm": ["LLVMOps.td"],
        "nvvm": ["NVVMOps.td"],
        "rocdl": ["ROCDLOps.td"],
    }
    capital_map = {
        "arith": "Arith",
        "math": "Math",
        "tensor": "Tensor",
        "linalg": "Linalg",
        "scf": "SCF",
        "func": "Func",
        "memref": "MemRef",
        "gpu": "GPU",
        "vector": "Vector",
        "llvm": "LLVMIR",
        "nvvm": "LLVMIR",
        "rocdl": "LLVMIR",
    }

    for d, files in dialect_map.items():
        d_cap = capital_map[d]
        for td_file in files:
            url = (
                f"https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/include/mlir/Dialect/{d_cap}/{td_file}"
                if d_cap == "LLVMIR"
                else f"https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/include/mlir/Dialect/{d_cap}/IR/{td_file}"
            )
            td_content = fetch_html(url)
            if td_content:
                parsed = parse_mlir_tablegen(td_content, d)
                for op in parsed:
                    if op["api_path"] not in seen_paths:
                        seen_paths.add(op["api_path"])
                        all_ops.append(op)

    # Live inspection fallback/supplement
    live_ops = inspect_mlir_python_module()
    for op in live_ops:
        if op["api_path"] not in seen_paths:
            seen_paths.add(op["api_path"])
            all_ops.append(op)

    if not all_ops:  # pragma: no cover
        print(f"Fetching dialects from {MLIR_DOCS_URL}...")
        index_html = fetch_html(MLIR_DOCS_URL)
        dialect_links = re.findall(
            r'<a href="([^"]+)">\'([a-zA-Z0-9_]+)\' Dialect</a>', index_html
        )
        if not dialect_links:
            dialect_links = re.findall(
                r"<a href=([^>]+)>\'([a-zA-Z0-9_]+)\' Dialect</a>",
                index_html,
            )

        seen_urls = set()
        for link, dialect_name in dialect_links:
            dialect_url = (
                f"https://mlir.llvm.org{link}" if link.startswith("/") else link
            )
            if dialect_url in seen_urls:
                continue
            seen_urls.add(dialect_url)

            ops = parse_dialect_page(dialect_url, dialect_name)
            for op in ops:
                if op["api_path"] not in seen_paths:
                    seen_paths.add(op["api_path"])
                    all_ops.append(op)

    stablehlo_ops = scrape_stablehlo()
    for op in stablehlo_ops:
        if op["api_path"] not in seen_paths:
            seen_paths.add(op["api_path"])
            all_ops.append(op)

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(all_ops, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Dumped exhaustive MLIR operations to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

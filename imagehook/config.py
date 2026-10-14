"""Optional settings file (TOML). Precedence: built-in defaults < config file < command line.

The file is looked up in this order: --config PATH, $IMAGEHOOK_CONFIG, ./imagehook.toml, <project>/imagehook.toml.
Sections only group the options; a key is the long option name (`ui_threshold` or `ui-threshold`). The values
become the argparse defaults of every subcommand that has the option, so `--help` shows them and the command line
still overrides them. imagehook.example.toml lists every option with its built-in default.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import tomllib
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
FILE_NAME = "imagehook.toml"
EXAMPLE = PROJECT / "imagehook.example.toml"


class ConfigError(Exception):
    pass


def find_config(explicit: Path | None = None) -> Path | None:
    if explicit is not None or os.environ.get("IMAGEHOOK_CONFIG"):
        p = Path(explicit or os.environ["IMAGEHOOK_CONFIG"])
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        return p
    for p in (Path.cwd() / FILE_NAME, PROJECT / FILE_NAME):
        if p.is_file():
            return p
    return None


def load(path: Path) -> dict[str, tuple[str, object]]:
    """TOML file -> {dest: (section, value)}. Top-level keys are allowed too (section "")."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name}: {e}") from None
    out: dict[str, tuple[str, object]] = {}
    for key, value in data.items():
        items = value.items() if isinstance(value, dict) else [(key, value)]
        section = key if isinstance(value, dict) else ""
        for k, v in items:
            dest = k.replace("-", "_")
            if dest in out:
                raise ConfigError(f"{path.name}: {k!r} is set twice ([{out[dest][0]}] and [{section}])")
            out[dest] = (section, v)
    return out


def boxes(value) -> tuple[tuple[float, float, float, float], ...]:
    """--change-ignore: a JSON string (command line) or a list (config) of [x1, y1, x2, y2] boxes."""
    from .filters import check_box

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentTypeError(f"not a JSON list of boxes: {e}") from None
    if not isinstance(value, (list, tuple)):
        raise argparse.ArgumentTypeError("expected a list of [x1, y1, x2, y2] boxes")
    try:
        return tuple(check_box("change_ignore", b) for b in value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Optional arguments of a (sub)parser by dest, without --help and --config."""
    return {a.dest: a for a in parser._actions if a.option_strings and a.dest not in ("help", "config")}


def _scalar(action: argparse.Action, value, base: Path):
    t = action.type
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise ValueError("expected true or false")
        return value
    if t is boxes:
        return boxes(value)
    if isinstance(value, bool):
        raise ValueError("expected a value, not true/false")
    if t is float:
        if not isinstance(value, (int, float)):
            raise ValueError("expected a number")
        value = float(value)
    elif t is int:
        if not isinstance(value, int):
            raise ValueError("expected a whole number")
    elif t is Path:
        if not isinstance(value, str):
            raise ValueError("expected a path in quotes")
        p = Path(os.path.expandvars(os.path.expanduser(value)))
        value = p if p.is_absolute() else base / p
    elif not isinstance(value, str):
        raise ValueError("expected a string in quotes")
    if action.choices is not None and value not in action.choices:
        raise ValueError(f"expected one of {', '.join(map(str, action.choices))}")
    return value


def convert(action: argparse.Action, value, base: Path):
    """A config value -> what argparse would have stored for the same option on the command line."""
    flag = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
    if not flag and action.type is not boxes and (action.nargs in ("+", "*") or isinstance(action.nargs, int)):
        if not isinstance(value, list):
            raise ValueError("expected a list in [brackets]")
        if isinstance(action.nargs, int) and len(value) != action.nargs:
            raise ValueError(f"expected {action.nargs} values")
        if action.nargs == "+" and not value:
            raise ValueError("expected at least one value")
        return [_scalar(action, v, base) for v in value]
    return _scalar(action, value, base)


def apply(subparsers: dict[str, argparse.ArgumentParser], path: Path) -> dict[str, object]:
    """Set the values of the config file as defaults of the subcommands. Returns {dest: value} that was applied."""
    values = load(path)
    known: dict[str, argparse.Action] = {}
    for p in subparsers.values():
        known.update(options(p))
    applied: dict[str, object] = {}
    for dest, (section, value) in values.items():
        where = f"{path.name}: [{section}] {dest}" if section else f"{path.name}: {dest}"
        if dest not in known:
            close = difflib.get_close_matches(dest, known, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else f"; see {EXAMPLE.name} for all options"
            raise ConfigError(f"{where}: unknown option{hint}")
        try:
            applied[dest] = convert(known[dest], value, path.parent)
        except (ValueError, argparse.ArgumentTypeError) as e:
            raise ConfigError(f"{where} = {value!r}: {e}") from None
    for p in subparsers.values():
        mine = options(p)
        p.set_defaults(**{k: v for k, v in applied.items() if k in mine})
    return applied


def example_sections(text: str | None = None) -> list[tuple[str, list[str]]]:
    """[(section, [dest, ...])] in the order of imagehook.example.toml, where every key is commented out."""
    text = EXAMPLE.read_text(encoding="utf-8") if text is None else text
    out: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if m := re.match(r"^\[(\w+)\]", line):
            out.append((m[1], []))
        elif (m := re.match(r"^#\s?([a-z][a-z0-9_]*)\s*=", line)) and out:
            out[-1][1].append(m[1])
    return out


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return json.dumps(str(v).replace("\\", "/"))

import argparse
import inspect
from dataclasses import MISSING, Field, fields
from types import UnionType
from typing import Callable, Literal, Sequence, Type, TypeVar, get_args, get_origin

T = TypeVar("T")


def _arg_type(tp):
    origin = get_origin(tp)
    if origin is Literal:
        return str
    if tp == int | list[int]:
        return lambda s: [int(x) for x in s.split(",")] if "," in s else int(s)
    if origin is UnionType:
        args = tuple(a for a in get_args(tp) if a is not type(None))
        return args[0] if len(args) == 1 else tp
    return tp


def _add_dataclass_to_parser_(
    parser: argparse.ArgumentParser, name: str, dc: Type[T]
) -> None:
    for f in fields(dc):
        arg_type = _arg_type(f.type)

        arg_name = f"--{name}.{f.name.replace('_', '-')}"

        if arg_type is bool:
            default = f.default if f.default is not MISSING else False
            parser.add_argument(
                arg_name, action=argparse.BooleanOptionalAction, default=default
            )
        else:
            parser.add_argument(arg_name, type=arg_type, required=f.default is MISSING)


def create_subparser(
    name: str,
    subparsers: argparse._SubParsersAction,
    dcs: Sequence[tuple[str, Type[T]]],
):
    parser: argparse.ArgumentParser = subparsers.add_parser(name)
    for name, dc in dcs:
        _add_dataclass_to_parser_(parser, name, dc)


def load_dc_from_arg_parser_args(name: str, dc: Type[T], args: argparse.Namespace) -> T:
    def _get_value(field: Field):
        val = getattr(args, f"{name}.{field.name}")
        if val is None:
            val = field.default
        if field.type is bool and val is MISSING:
            val = False

        assert val is not MISSING
        return val

    return dc(**{f.name: _get_value(f) for f in fields(dc)})


def _build_parser(
    experiments: Callable | list[tuple[str, Callable]],
) -> tuple[argparse.ArgumentParser, dict[str | None, list[inspect.Parameter]]]:
    parser = argparse.ArgumentParser()
    parameters: dict[str | None, list[inspect.Parameter]] = {}

    if not isinstance(experiments, list):
        sig = inspect.signature(experiments)
        parameters[None] = [p for p in sig.parameters.values()]
        for name, dc in [(p.name, p.annotation) for p in parameters[None]]:
            _add_dataclass_to_parser_(parser, name, dc)

    else:
        subparsers = parser.add_subparsers(dest="env")
        for name, fn in experiments:
            sig = inspect.signature(fn)
            parameters[name] = [p for p in sig.parameters.values()]

            create_subparser(
                name=name,
                subparsers=subparsers,
                dcs=[(p.name, p.annotation) for p in parameters[name]],
            )

    return parser, parameters


def run_experiments_parser(experiments: Callable | list[tuple[str, Callable]]):
    parser, parameters = _build_parser(experiments)
    args = parser.parse_args()

    if not isinstance(experiments, list):
        kwargs = {}
        for p in parameters[None]:
            kwargs[p.name] = load_dc_from_arg_parser_args(p.name, p.annotation, args)

        return experiments(**kwargs)

    for name, fn in experiments:
        if args.env == name:
            kwargs = {}
            for p in parameters[name]:
                param_class = p.annotation
                kwargs[p.name] = load_dc_from_arg_parser_args(p.name, param_class, args)

            return fn(**kwargs)

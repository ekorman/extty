"""Tests for ``extty.arg_parser``."""

from dataclasses import dataclass
from typing import Literal

import pytest

from extty.arg_parser import _build_parser, get_value, run_experiments_parser


def _kwargs(fn, argv, env=None):
    """Build a parser for ``fn``, parse ``argv``, return the kwargs dict.

    Parameters
    ----------
    fn : Callable | list[tuple[str, Callable]]
        Single callable or a list of named callables (subparser mode).
    argv : list[str]
        Command-line arguments to feed argparse.
    env : str | None
        For subparser mode, which subcommand's parameters to extract.
    """
    parser, parameters = _build_parser(fn)
    args = parser.parse_args(argv)
    key = env if env is not None else None
    return {p.name: get_value(args, p) for p in parameters[key]}


class TestPrimitives:
    def test_str_default_used_when_omitted(self):
        def fn(name: str = "foo"): ...

        assert _kwargs(fn, []) == {"name": "foo"}

    def test_str_override(self):
        def fn(name: str = "foo"): ...

        assert _kwargs(fn, ["--name", "bar"]) == {"name": "bar"}

    def test_int_default_and_override(self):
        def fn(n: int = 5): ...

        assert _kwargs(fn, []) == {"n": 5}
        assert _kwargs(fn, ["--n", "12"]) == {"n": 12}

    def test_float_default_and_override(self):
        def fn(lr: float = 0.1): ...

        assert _kwargs(fn, []) == {"lr": 0.1}
        assert _kwargs(fn, ["--lr", "0.001"]) == {"lr": 0.001}

    def test_required_when_no_default(self):
        def fn(name: str): ...

        with pytest.raises(SystemExit):
            _kwargs(fn, [])

    def test_required_satisfied(self):
        def fn(name: str): ...

        assert _kwargs(fn, ["--name", "x"]) == {"name": "x"}

    def test_underscore_renamed_to_dash(self):
        def fn(my_param: str = "a"): ...

        assert _kwargs(fn, ["--my-param", "b"]) == {"my_param": "b"}


class TestBool:
    def test_bool_default_false(self):
        def fn(verbose: bool = False): ...

        assert _kwargs(fn, []) == {"verbose": False}
        assert _kwargs(fn, ["--verbose"]) == {"verbose": True}
        assert _kwargs(fn, ["--no-verbose"]) == {"verbose": False}

    def test_bool_default_true(self):
        def fn(verbose: bool = True): ...

        assert _kwargs(fn, []) == {"verbose": True}
        assert _kwargs(fn, ["--no-verbose"]) == {"verbose": False}

    def test_bool_no_default_treated_as_false(self):
        def fn(verbose: bool): ...

        assert _kwargs(fn, []) == {"verbose": False}
        assert _kwargs(fn, ["--verbose"]) == {"verbose": True}


class TestLiteralAndUnion:
    def test_literal_accepts_strings(self):
        def fn(mode: Literal["a", "b"] = "a"): ...

        assert _kwargs(fn, []) == {"mode": "a"}
        assert _kwargs(fn, ["--mode", "b"]) == {"mode": "b"}

    def test_optional_int(self):
        def fn(n: int | None = None): ...

        assert _kwargs(fn, []) == {"n": None}
        assert _kwargs(fn, ["--n", "3"]) == {"n": 3}

    def test_int_or_list_int_scalar(self):
        def fn(seeds: int | list[int] = 0): ...

        assert _kwargs(fn, ["--seeds", "7"]) == {"seeds": 7}

    def test_int_or_list_int_csv(self):
        def fn(seeds: int | list[int] = 0): ...

        assert _kwargs(fn, ["--seeds", "1,2,3"]) == {"seeds": [1, 2, 3]}


@dataclass
class _Cfg:
    x: int = 3
    name: str = "hi"
    flag: bool = False


@dataclass
class _CfgRequired:
    x: int
    y: float


class TestDataclass:
    def test_defaults(self):
        def fn(cfg: _Cfg): ...

        assert _kwargs(fn, []) == {"cfg": _Cfg(x=3, name="hi", flag=False)}

    def test_overrides(self):
        def fn(cfg: _Cfg): ...

        result = _kwargs(fn, ["--cfg.x", "9", "--cfg.name", "hey", "--cfg.flag"])
        assert result == {"cfg": _Cfg(x=9, name="hey", flag=True)}

    def test_required_fields(self):
        def fn(cfg: _CfgRequired): ...

        with pytest.raises(SystemExit):
            _kwargs(fn, [])

        result = _kwargs(fn, ["--cfg.x", "1", "--cfg.y", "2.5"])
        assert result == {"cfg": _CfgRequired(x=1, y=2.5)}

    def test_dashed_field_name(self):
        @dataclass
        class C:
            my_field: int = 0

        def fn(c: C): ...

        assert _kwargs(fn, ["--c.my-field", "4"]) == {"c": C(my_field=4)}


class TestMixed:
    def test_primitive_and_dataclass(self):
        def fn(cfg: _Cfg, n: int = 1, name: str = "x"): ...

        result = _kwargs(fn, ["--cfg.x", "10", "--n", "5"])
        assert result == {"cfg": _Cfg(x=10), "n": 5, "name": "x"}


class TestSubparsers:
    def test_route_to_first(self):
        def a(cfg: _Cfg): ...
        def b(n: int = 0): ...

        result = _kwargs([("a", a), ("b", b)], ["a", "--cfg.x", "11"], env="a")
        assert result == {"cfg": _Cfg(x=11)}

    def test_route_to_second(self):
        def a(cfg: _Cfg): ...
        def b(n: int = 0): ...

        result = _kwargs([("a", a), ("b", b)], ["b", "--n", "42"], env="b")
        assert result == {"n": 42}


class TestRunExperimentsParser:
    def test_single_callable_invokes_fn(self, monkeypatch):
        captured = {}

        def fn(name: str = "foo", n: int = 1):
            captured["name"] = name
            captured["n"] = n
            return "done"

        monkeypatch.setattr("sys.argv", ["prog", "--name", "bar", "--n", "9"])
        assert run_experiments_parser(fn) == "done"
        assert captured == {"name": "bar", "n": 9}

    def test_list_dispatches_by_env(self, monkeypatch):
        captured = {}

        def train(cfg: _Cfg):
            captured["which"] = "train"
            captured["cfg"] = cfg
            return cfg

        def eval_(n: int = 1):
            captured["which"] = "eval"
            captured["n"] = n
            return n

        monkeypatch.setattr("sys.argv", ["prog", "eval", "--n", "7"])
        assert run_experiments_parser([("train", train), ("eval", eval_)]) == 7
        assert captured == {"which": "eval", "n": 7}

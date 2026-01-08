import importlib
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Callable


def _check_allowed_target(target: str, allowed_prefixes: Optional[Iterable[str]]) -> None:
    if allowed_prefixes is None:
        return
    prefixes = tuple(allowed_prefixes)
    if not any(target.startswith(p) for p in prefixes):
        raise ValueError(f"target '{target}' is not allowed. allowed_prefixes={list(prefixes)}")


def instantiate(
    spec: Callable[[], Any],
    *,
    allowed_prefixes: Optional[Iterable[str]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Hydra 없이도 동작하는 최소 _target_ 인스턴스화.
    - _target_: "pkg.module:ClassName" 또는 "pkg.module.ClassName"
    - _args_: positional args (옵션)
    - 나머지 키는 kwargs로 전달
    """
    if "_target_" not in spec:
        raise ValueError("spec must contain '_target_'")

    local = dict(spec)
    target = str(local.pop("_target_"))
    _check_allowed_target(target, allowed_prefixes)

    args = local.pop("_args_", ())
    if args is None:
        args = ()
    if not isinstance(args, (list, tuple)):
        raise TypeError("_args_ must be list or tuple")

    if kwargs:
        local.update(kwargs)

    if ":" in target:
        module_path, obj_name = target.split(":", 1)
    else:
        module_path, obj_name = target.rsplit(".", 1)

    module = importlib.import_module(module_path)
    cls = getattr(module, obj_name)
    return cls(*args, **local)

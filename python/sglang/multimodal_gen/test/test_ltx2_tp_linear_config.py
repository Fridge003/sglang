import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
)


def _module() -> ast.Module:
    return ast.parse(SOURCE_PATH.read_text())


def _class_method(class_name: str, method_name: str) -> ast.FunctionDef:
    module = _module()
    class_def = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_def.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _literal_keywords(call: ast.Call) -> dict[str, object]:
    result = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            result[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            continue
    return result


def _self_assignment_call(
    class_name: str, attr_name: str
) -> tuple[str | None, dict[str, object]]:
    init_fn = _class_method(class_name, "__init__")
    for node in init_fn.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == attr_name
            and isinstance(node.value, ast.Call)
        ):
            func = node.value.func
            func_name = func.id if isinstance(func, ast.Name) else None
            return func_name, _literal_keywords(node.value)
    raise AssertionError(f"self.{attr_name} not found in {class_name}.__init__")


def _self_sequential_first_call(
    class_name: str, attr_name: str
) -> tuple[str | None, dict[str, object]]:
    init_fn = _class_method(class_name, "__init__")
    for node in init_fn.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            not isinstance(target, ast.Attribute)
            or not isinstance(target.value, ast.Name)
            or target.value.id != "self"
            or target.attr != attr_name
            or not isinstance(node.value, ast.Call)
            or not isinstance(node.value.func, ast.Attribute)
            or node.value.func.attr != "Sequential"
            or not node.value.args
            or not isinstance(node.value.args[0], ast.Call)
        ):
            continue
        first_call = node.value.args[0]
        func = first_call.func
        func_name = func.id if isinstance(func, ast.Name) else None
        return func_name, _literal_keywords(first_call)
    raise AssertionError(f"self.{attr_name} Sequential not found in {class_name}.__init__")


def test_text_projection_uses_tp_friendly_linears():
    linear_1_func, linear_1_keywords = _self_assignment_call(
        "LTX2TextProjection", "linear_1"
    )
    linear_2_func, linear_2_keywords = _self_assignment_call(
        "LTX2TextProjection", "linear_2"
    )

    assert linear_1_func == "ColumnParallelLinear"
    assert linear_1_keywords["gather_output"] is False
    assert linear_1_keywords["accumulate_in_fp32"] is True

    assert linear_2_func == "RowParallelLinear"
    assert linear_2_keywords["input_is_parallel"] is True
    assert linear_2_keywords["accumulate_in_fp32"] is True


def test_attention_and_ff_use_tp_friendly_linears():
    for attr_name in ("to_q", "to_k", "to_v"):
        func_name, keywords = _self_assignment_call("LTX2Attention", attr_name)
        assert func_name == "ColumnParallelLinear"
        assert keywords["gather_output"] is False
        assert keywords["accumulate_in_fp32"] is True

    to_out_func, to_out_keywords = _self_sequential_first_call("LTX2Attention", "to_out")
    assert to_out_func == "RowParallelLinear"
    assert to_out_keywords["input_is_parallel"] is True
    assert to_out_keywords["accumulate_in_fp32"] is True

    proj_in_func, proj_in_keywords = _self_assignment_call("LTX2FeedForward", "proj_in")
    proj_out_func, proj_out_keywords = _self_assignment_call("LTX2FeedForward", "proj_out")
    assert proj_in_func == "ColumnParallelLinear"
    assert proj_in_keywords["gather_output"] is False
    assert proj_out_func == "RowParallelLinear"
    assert proj_out_keywords["input_is_parallel"] is True
    assert proj_out_keywords["accumulate_in_fp32"] is True


def test_adaln_and_output_heads_enable_fp32_accumulation():
    linear_func, linear_keywords = _self_assignment_call(
        "LTX2AdaLayerNormSingle", "linear"
    )
    assert linear_func == "ColumnParallelLinear"
    assert linear_keywords["accumulate_in_fp32"] is True

    proj_out_func, proj_out_keywords = _self_assignment_call(
        "LTX2VideoTransformer3DModel", "proj_out"
    )
    audio_proj_out_func, audio_proj_out_keywords = _self_assignment_call(
        "LTX2VideoTransformer3DModel", "audio_proj_out"
    )
    assert proj_out_func == "ColumnParallelLinear"
    assert proj_out_keywords["gather_output"] is True
    assert proj_out_keywords["accumulate_in_fp32"] is True
    assert audio_proj_out_func == "ColumnParallelLinear"
    assert audio_proj_out_keywords["gather_output"] is True
    assert audio_proj_out_keywords["accumulate_in_fp32"] is True


def test_forward_no_longer_has_stage2_patchify_noop():
    forward_fn = _class_method("LTX2VideoTransformer3DModel", "forward")

    assigned_names = {
        target.id
        for node in ast.walk(forward_fn)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "patchify_proj_input" not in assigned_names

    kwargs_accesses = [
        node
        for node in ast.walk(forward_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "kwargs"
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "ltx2_phase"
    ]
    assert not kwargs_accesses

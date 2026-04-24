import json

from tcrb.research import (
    _build_dpo_trainer_kwargs,
    _prepare_model_for_quantized_training,
    _training_dtype,
    _training_precision_kwargs,
    apply_function_name_masking,
    classify_tool_call_failure,
    mine_benchmark_failure_preferences,
    mine_failure_preferences,
    normalize_sharegpt_record,
    normalize_toolpreference_record,
    research_recipe_from_dict,
)


def test_normalize_sharegpt_record_keeps_messages_tools_and_text():
    row = {
        "system": "You are a function-calling model.",
        "conversations": [
            {"from": "human", "value": "What's the weather in Berlin?"},
            {
                "from": "gpt",
                "value": '{"name":"weather.lookup","arguments":{"city":"Berlin"}}',
            },
        ],
        "tools": [{"type": "function", "function": {"name": "weather.lookup"}}],
        "category": "weather",
    }

    normalized = normalize_sharegpt_record(row, source_name="toolace")

    assert [message["role"] for message in normalized["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert normalized["tools"][0]["function"]["name"] == "weather.lookup"
    assert "<|tools|>" in normalized["text"]
    assert "weather.lookup" in normalized["text"]
    assert normalized["metadata"]["category"] == "weather"


def test_function_name_masking_replaces_tool_names_consistently():
    record = {
        "messages": [
            {"role": "user", "content": "Use weather.lookup for Berlin."},
            {
                "role": "assistant",
                "content": '{"name":"weather.lookup","arguments":{"city":"Berlin"}}',
            },
        ],
        "tools": [{"type": "function", "function": {"name": "weather.lookup"}}],
        "metadata": {},
    }

    masked = apply_function_name_masking(record, ratio=1.0, seed=7)

    replacement = masked["metadata"]["masking"]["replacements"]["weather.lookup"]
    assert masked["tools"][0]["function"]["name"] == replacement
    assert replacement in masked["messages"][0]["content"]
    assert replacement in masked["messages"][1]["content"]
    assert replacement in masked["text"]


def test_normalize_toolpreference_record_reshapes_prompt_and_pair():
    row = {
        "instruction": {"query": "Check order status"},
        "input": [{"name": "order.status"}],
        "output": [
            {"name": "order.status", "arguments": {"order_id": "A1"}},
            {"name": "order.lookup", "arguments": {"order_id": "A1"}},
        ],
    }

    normalized = normalize_toolpreference_record(row, source_name="toolpreference")

    assert "Check order status" in normalized["prompt"]
    assert '"order.status"' in normalized["chosen"]
    assert '"order.lookup"' in normalized["rejected"]
    assert normalized["source"] == "toolpreference"


def test_classify_tool_call_failure_distinguishes_common_error_types():
    expected = {"name": "calendar.create", "arguments": {"day": "monday"}}
    hallucinated = {"name": "ghost.tool", "arguments": {"day": "monday"}}
    wrong_name = {"name": "calendar.delete", "arguments": {"day": "monday"}}
    wrong_arg_name = {"name": "calendar.create", "arguments": {"date": "monday"}}
    wrong_arg_value = {"name": "calendar.create", "arguments": {"day": "friday"}}

    assert (
        classify_tool_call_failure(
            expected,
            hallucinated,
            allowed_tool_names={"calendar.create", "calendar.delete"},
        )
        == "hallucinated_function"
    )
    assert classify_tool_call_failure(expected, wrong_name) == "wrong_function_name"
    assert classify_tool_call_failure(expected, wrong_arg_name) == "missing_required_arg"
    assert classify_tool_call_failure(expected, wrong_arg_value) == "wrong_argument_value"


def test_mine_failure_preferences_skips_matches_and_keeps_failure_metadata():
    payload = {
        "cases": [
            {
                "task_id": "ok",
                "prompt": "Query A",
                "expected_output": {"name": "weather.lookup", "arguments": {"city": "Berlin"}},
                "predicted_output": {"name": "weather.lookup", "arguments": {"city": "Berlin"}},
            },
            {
                "task_id": "bad",
                "prompt": "Query B",
                "expected_output": {"name": "weather.lookup", "arguments": {"city": "Berlin"}},
                "predicted_output": {"name": "weather.find", "arguments": {"city": "Berlin"}},
                "source": "bfcl",
            },
        ]
    }

    mined = mine_failure_preferences(payload, allowed_tool_names={"weather.lookup", "weather.find"})

    assert len(mined) == 1
    assert mined[0]["prompt"] == "Query B"
    assert mined[0]["failure_type"] == "wrong_function_name"
    assert json.loads(mined[0]["chosen"])["name"] == "weather.lookup"
    assert mined[0]["metadata"]["task_id"] == "bad"


def test_mine_benchmark_failure_preferences_uses_result_and_eval_cases():
    result_payload = {
        "task_results": [
            {
                "task_id": "t1",
                "policy": "naive_retry",
                "attempts": [
                    {"tool_name": "weather.find"},
                    {"tool_name": "weather.lookup"},
                ],
            },
            {
                "task_id": "t2",
                "policy": "naive_retry",
                "attempts": [{"tool_name": "order.lookup"}],
            },
        ]
    }
    eval_cases_payload = {
        "cases": [
            {
                "task_id": "t1",
                "question": "Find Berlin weather",
                "expected_first_tool": "weather.lookup",
                "expected_tool_sequence": ["weather.lookup"],
            },
            {
                "task_id": "t2",
                "question": "Look up order A1",
                "expected_first_tool": "order.lookup",
                "expected_tool_sequence": ["order.lookup"],
            },
        ]
    }

    mined = mine_benchmark_failure_preferences(
        result_payload,
        eval_cases_payload,
        policy="naive_retry",
    )

    assert len(mined) == 1
    assert mined[0]["prompt"] == "Find Berlin weather"
    assert mined[0]["failure_type"] == "wrong_function_name"
    assert mined[0]["metadata"]["policy"] == "naive_retry"
    chosen = json.loads(mined[0]["chosen"])
    rejected = json.loads(mined[0]["rejected"])
    assert chosen["tool_calls"][0]["name"] == "weather.lookup"
    assert rejected["tool_calls"][0]["name"] == "weather.find"


def test_research_recipe_parses_optional_adapter_path():
    recipe = research_recipe_from_dict(
        {
            "stage": "dpo",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "adapter_path": "outputs/research/qwen25-3b-sft-toolace",
            "output_dir": "outputs/research/qwen25-3b-dpo",
        }
    )

    assert recipe.adapter_path == "outputs/research/qwen25-3b-sft-toolace"


def test_research_recipe_parses_runtime_precision_flags():
    recipe = research_recipe_from_dict(
        {
            "stage": "sft",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "output_dir": "outputs/research/qwen25-3b-sft-toolace",
            "fp16": False,
            "bf16": False,
            "load_in_4bit": "true",
        }
    )

    assert recipe.fp16 is False
    assert recipe.bf16 is False
    assert recipe.load_in_4bit is True


def test_training_precision_kwargs_follow_recipe():
    recipe = research_recipe_from_dict(
        {
            "stage": "dpo",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "output_dir": "outputs/research/qwen25-3b-dpo",
            "fp16": False,
            "bf16": True,
        }
    )

    assert _training_precision_kwargs(recipe) == {"fp16": False, "bf16": True}


def test_training_dtype_follows_recipe_precision_order():
    class DummyTorch:
        float16 = "float16"
        bfloat16 = "bfloat16"
        float32 = "float32"

    fp16_recipe = research_recipe_from_dict(
        {
            "stage": "sft",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "output_dir": "outputs/research/qwen25-3b-sft-toolace",
            "fp16": True,
            "bf16": False,
        }
    )
    bf16_recipe = research_recipe_from_dict(
        {
            "stage": "sft",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "output_dir": "outputs/research/qwen25-3b-sft-toolace",
            "fp16": False,
            "bf16": True,
        }
    )
    full_precision_recipe = research_recipe_from_dict(
        {
            "stage": "sft",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "output_dir": "outputs/research/qwen25-3b-sft-toolace",
            "fp16": False,
            "bf16": False,
        }
    )

    assert _training_dtype(fp16_recipe, torch_module=DummyTorch) == "float16"
    assert _training_dtype(bf16_recipe, torch_module=DummyTorch) == "bfloat16"
    assert _training_dtype(full_precision_recipe, torch_module=DummyTorch) == "float32"


def test_prepare_model_for_quantized_training_only_wraps_4bit_models():
    class DummyPeftModule:
        @staticmethod
        def prepare_model_for_kbit_training(model):
            return {"prepared": model}

    quantized_recipe = research_recipe_from_dict(
        {
            "stage": "sft",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "output_dir": "outputs/research/qwen25-3b-sft-toolace",
            "load_in_4bit": True,
        }
    )
    full_precision_recipe = research_recipe_from_dict(
        {
            "stage": "sft",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "output_dir": "outputs/research/qwen25-3b-sft-toolace",
            "load_in_4bit": False,
        }
    )

    assert _prepare_model_for_quantized_training(
        quantized_recipe,
        model="model",
        peft_module=DummyPeftModule,
    ) == {"prepared": "model"}
    assert _prepare_model_for_quantized_training(
        full_precision_recipe,
        model="model",
        peft_module=DummyPeftModule,
    ) == "model"


def test_build_dpo_trainer_kwargs_skips_peft_config_when_adapter_loaded():
    class DummyTrainer:
        def __init__(
            self,
            model,
            args,
            train_dataset,
            ref_model=None,
            beta=0.5,
            peft_config=None,
            processing_class=None,
        ):
            pass

    class DummyModule:
        DPOTrainer = DummyTrainer

    kwargs = _build_dpo_trainer_kwargs(
        trl_module=DummyModule,
        model=object(),
        training_args=object(),
        train_dataset=object(),
        tokenizer=object(),
        recipe=research_recipe_from_dict(
            {
                "stage": "dpo",
                "base_model": "Qwen/Qwen2.5-3B-Instruct",
                "output_dir": "outputs/research/qwen25-3b-dpo",
            }
        ),
        lora_config=None,
    )

    assert "peft_config" not in kwargs
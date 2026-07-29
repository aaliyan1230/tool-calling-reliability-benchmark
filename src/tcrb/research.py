from __future__ import annotations

import inspect
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROMPT_KEYS = (
    "prompt",
    "instruction",
    "input",
    "messages",
    "conversation",
)
EXPECTED_KEYS = (
    "expected_output",
    "expected",
    "ground_truth",
    "chosen",
    "target",
)
PREDICTED_KEYS = (
    "predicted_output",
    "prediction",
    "rejected",
    "actual",
    "model_output",
)


@dataclass(frozen=True)
class ResearchDatasetSource:
    name: str
    format: str
    dataset_id: str | None = None
    split: str = "train"
    config_name: str | None = None
    path: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class FunctionMaskingConfig:
    ratio: float = 0.0
    seed: int = 0
    placeholder_prefix: str = "[MASK_FUNC_"


@dataclass(frozen=True)
class ResearchRecipe:
    stage: str
    base_model: str
    output_dir: str
    adapter_path: str | None = None
    dataset_sources: list[ResearchDatasetSource] = field(default_factory=list)
    input_path: str | None = None
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    packing: bool = False
    beta: float = 0.5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    load_in_4bit: bool = True
    fp16: bool = True
    bf16: bool = False
    masking: FunctionMaskingConfig = field(default_factory=FunctionMaskingConfig)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def dataset_source_from_dict(payload: dict[str, Any]) -> ResearchDatasetSource:
    source_name = str(payload.get("name", "")).strip()
    if not source_name:
        raise ValueError("research dataset source requires a non-empty name")
    source_format = str(payload.get("format", "sharegpt")).strip().lower()
    return ResearchDatasetSource(
        name=source_name,
        format=source_format,
        dataset_id=(str(payload.get("dataset_id", "")).strip() or None),
        split=str(payload.get("split", "train")).strip() or "train",
        config_name=(str(payload.get("config_name", "")).strip() or None),
        path=(str(payload.get("path", "")).strip() or None),
        limit=(int(payload["limit"]) if payload.get("limit") is not None else None),
    )


def research_recipe_from_dict(payload: dict[str, Any]) -> ResearchRecipe:
    sources = [
        dataset_source_from_dict(item)
        for item in list(payload.get("dataset_sources", []))
    ]
    masking_payload = dict(payload.get("masking", {}))
    return ResearchRecipe(
        stage=str(payload.get("stage", "sft")).strip().lower() or "sft",
        base_model=str(payload.get("base_model", "")).strip(),
        output_dir=str(payload.get("output_dir", "outputs/research")).strip()
        or "outputs/research",
        adapter_path=(str(payload.get("adapter_path", "")).strip() or None),
        dataset_sources=sources,
        input_path=(str(payload.get("input_path", "")).strip() or None),
        learning_rate=float(payload.get("learning_rate", 1e-4)),
        num_train_epochs=float(payload.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(payload.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 1)),
        warmup_ratio=float(payload.get("warmup_ratio", 0.03)),
        max_seq_length=int(payload.get("max_seq_length", 2048)),
        packing=bool(payload.get("packing", False)),
        beta=float(payload.get("beta", 0.5)),
        lora_r=int(payload.get("lora_r", 16)),
        lora_alpha=int(payload.get("lora_alpha", 32)),
        lora_dropout=float(payload.get("lora_dropout", 0.05)),
        target_modules=payload.get("target_modules", "all-linear"),
        load_in_4bit=_coerce_bool(payload.get("load_in_4bit"), default=True),
        fp16=_coerce_bool(payload.get("fp16"), default=True),
        bf16=_coerce_bool(payload.get("bf16"), default=False),
        masking=FunctionMaskingConfig(
            ratio=float(masking_payload.get("ratio", 0.0)),
            seed=int(masking_payload.get("seed", 0)),
            placeholder_prefix=(
                str(masking_payload.get("placeholder_prefix", "[MASK_FUNC_"))
                or "[MASK_FUNC_"
            ),
        ),
    )


def load_research_recipe(path: str | Path) -> ResearchRecipe:
    with Path(path).open("r", encoding="utf-8") as handle:
        return research_recipe_from_dict(json.load(handle))


def _training_precision_kwargs(recipe: ResearchRecipe) -> dict[str, bool]:
    return {
        "fp16": recipe.fp16,
        "bf16": recipe.bf16,
    }


def _training_dtype(recipe: ResearchRecipe, *, torch_module: Any) -> Any:
    if recipe.bf16:
        return torch_module.bfloat16
    if recipe.fp16:
        return torch_module.float16
    return torch_module.float32


def _prepare_model_for_quantized_training(
    recipe: ResearchRecipe,
    *,
    model: Any,
    peft_module: Any,
) -> Any:
    if not recipe.load_in_4bit:
        return model
    return peft_module.prepare_model_for_kbit_training(model)


def _attach_trainable_adapter_if_present(
    recipe: ResearchRecipe,
    *,
    model: Any,
    peft_module: Any,
    lora_config: Any,
) -> tuple[Any, Any]:
    adapter_path = str(recipe.adapter_path or "").strip()
    if not adapter_path:
        return model, lora_config
    return (
        peft_module.PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
        ),
        None,
    )


def _role_from_sharegpt(raw_role: str) -> str:
    role = str(raw_role).strip().lower()
    if role == "human":
        return "user"
    if role == "gpt":
        return "assistant"
    if role in {"tool", "function", "assistant", "user", "system"}:
        return role
    return role or "user"


def normalize_sharegpt_record(row: dict[str, Any], *, source_name: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system_text = str(row.get("system", "")).strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for turn in list(row.get("conversations", [])):
        role = _role_from_sharegpt(str(turn.get("from", "user")))
        content = turn.get("value")
        messages.append({"role": role, "content": content})

    tools = list(row.get("tools", []))
    metadata = {
        "source": source_name,
        "category": row.get("category"),
    }
    normalized = {
        "messages": messages,
        "tools": tools,
        "source": source_name,
        "metadata": metadata,
    }
    normalized["text"] = render_sft_text(normalized)
    return normalized


def normalize_toolpreference_record(
    row: dict[str, Any], *, source_name: str
) -> dict[str, Any]:
    outputs = list(row.get("output", []))
    if len(outputs) < 2:
        raise ValueError("toolpreference record requires two outputs: chosen and rejected")
    prompt_parts = []
    instruction = row.get("instruction")
    if instruction is not None:
        prompt_parts.append(_serialize_prompt_part(instruction))
    model_input = row.get("input")
    if model_input is not None:
        prompt_parts.append(_serialize_prompt_part(model_input))
    return {
        "prompt": "\n\n".join(part for part in prompt_parts if part),
        "chosen": _serialize_prompt_part(outputs[0]),
        "rejected": _serialize_prompt_part(outputs[1]),
        "source": source_name,
        "metadata": {"source": source_name},
    }


def apply_function_name_masking(
    record: dict[str, Any],
    *,
    ratio: float,
    seed: int = 0,
    placeholder_prefix: str = "[MASK_FUNC_",
) -> dict[str, Any]:
    if ratio <= 0.0:
        return json.loads(json.dumps(record))

    tools = list(record.get("tools", []))
    tool_names = [name for name in _extract_tool_names(tools) if name]
    if not tool_names:
        return json.loads(json.dumps(record))

    bounded_ratio = max(0.0, min(1.0, ratio))
    mask_count = max(1, int(round(len(tool_names) * bounded_ratio)))
    rng = random.Random(seed)
    selected = sorted(rng.sample(tool_names, min(mask_count, len(tool_names))))
    replacements = {
        name: f"{placeholder_prefix}{index}]"
        for index, name in enumerate(selected, start=1)
    }

    masked = _deep_replace(json.loads(json.dumps(record)), replacements)
    masked.setdefault("metadata", {})
    masked["metadata"]["masking"] = {
        "ratio": bounded_ratio,
        "replacements": replacements,
    }
    if "messages" in masked:
        masked["text"] = render_sft_text(masked)
    return masked


def render_sft_text(record: dict[str, Any]) -> str:
    lines: list[str] = []
    tools = list(record.get("tools", []))
    if tools:
        lines.append("<|tools|>")
        lines.append(json.dumps(tools, ensure_ascii=True, sort_keys=True))
    for message in list(record.get("messages", [])):
        role = str(message.get("role", "user")).strip() or "user"
        lines.append(f"<|{role}|>")
        content = message.get("content", "")
        lines.append(_serialize_prompt_part(content))
    return "\n".join(lines).strip()


def normalize_tool_calls(payload: Any) -> list[dict[str, Any]]:
    parsed = _coerce_json_payload(payload)
    return _extract_tool_calls(parsed)


def classify_tool_call_failure(
    expected_output: Any,
    predicted_output: Any,
    *,
    allowed_tool_names: set[str] | None = None,
) -> str:
    expected_calls = normalize_tool_calls(expected_output)
    predicted_calls = normalize_tool_calls(predicted_output)

    if not expected_calls and not predicted_calls:
        return "match"
    if not expected_calls and predicted_calls:
        return "spurious_call"
    if expected_calls and not predicted_calls:
        return "missing_required_arg"

    expected = expected_calls[0]
    predicted = predicted_calls[0]
    expected_name = str(expected.get("name", "")).strip()
    predicted_name = str(predicted.get("name", "")).strip()

    if expected_name != predicted_name:
        if allowed_tool_names and predicted_name and predicted_name not in allowed_tool_names:
            return "hallucinated_function"
        return "wrong_function_name"

    expected_args = dict(expected.get("arguments", {}))
    predicted_args = dict(predicted.get("arguments", {}))
    expected_keys = set(expected_args)
    predicted_keys = set(predicted_args)
    if expected_keys - predicted_keys:
        return "missing_required_arg"
    if predicted_keys - expected_keys:
        return "wrong_argument_name"
    for key in sorted(expected_keys):
        if expected_args.get(key) != predicted_args.get(key):
            return "wrong_argument_value"
    if len(expected_calls) != len(predicted_calls):
        return "spurious_call"
    return "match"


def mine_failure_preferences(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    allowed_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = list(payload.get("cases", payload.get("rows", [])))
    else:
        rows = list(payload)

    mined: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt_value = _pick_first_present(row, PROMPT_KEYS)
        expected_value = _pick_first_present(row, EXPECTED_KEYS)
        predicted_value = _pick_first_present(row, PREDICTED_KEYS)
        if expected_value is None or predicted_value is None:
            continue
        failure_type = classify_tool_call_failure(
            expected_value,
            predicted_value,
            allowed_tool_names=allowed_tool_names,
        )
        if failure_type == "match":
            continue
        prompt_text = _serialize_prompt_part(prompt_value)
        mined.append(
            {
                "prompt": prompt_text,
                "chosen": _serialize_prompt_part(expected_value),
                "rejected": _serialize_prompt_part(predicted_value),
                "failure_type": failure_type,
                "source": str(row.get("source", "failure_mining")).strip()
                or "failure_mining",
                "metadata": {
                    "row_index": index,
                    "task_id": row.get("task_id"),
                },
            }
        )
    return mined


def mine_benchmark_failure_preferences(
    result_payload: dict[str, Any],
    eval_cases_payload: dict[str, Any],
    *,
    policy: str | None = None,
    source_name: str = "tcrb_eval_cases",
) -> list[dict[str, Any]]:
    case_map = {
        str(case.get("task_id", "")).strip(): case
        for case in list(eval_cases_payload.get("cases", []))
        if str(case.get("task_id", "")).strip()
    }

    rows: list[dict[str, Any]] = []
    for task_result in list(result_payload.get("task_results", [])):
        task_id = str(task_result.get("task_id", "")).strip()
        active_policy = str(task_result.get("policy", "")).strip()
        if not task_id or task_id not in case_map:
            continue
        if policy is not None and active_policy != str(policy).strip():
            continue

        case = case_map[task_id]
        predicted_sequence = [
            str(attempt.get("tool_name", "")).strip()
            for attempt in list(task_result.get("attempts", []))
            if str(attempt.get("tool_name", "")).strip()
        ]
        expected_sequence = [
            str(tool_name).strip()
            for tool_name in list(case.get("expected_tool_sequence", []))
            if str(tool_name).strip()
        ]
        if not expected_sequence:
            expected_first = str(case.get("expected_first_tool", "")).strip()
            if expected_first:
                expected_sequence = [expected_first]

        rows.append(
            {
                "task_id": task_id,
                "prompt": str(case.get("question", "")).strip(),
                "expected_output": _tool_sequence_to_payload(expected_sequence),
                "predicted_output": _tool_sequence_to_payload(predicted_sequence),
                "source": source_name,
                "policy": active_policy,
                "metadata": {
                    "question": case.get("question"),
                    "expected_first_tool": case.get("expected_first_tool"),
                    "called_first_tool": predicted_sequence[0] if predicted_sequence else "",
                },
            }
        )

    mined = mine_failure_preferences(rows, allowed_tool_names=None)
    for item in mined:
        task_id = item.get("metadata", {}).get("task_id")
        if task_id is None:
            continue
        active_case = case_map.get(str(task_id), {})
        active_policy = next(
            (
                str(task_result.get("policy", "")).strip()
                for task_result in list(result_payload.get("task_results", []))
                if str(task_result.get("task_id", "")).strip() == str(task_id)
                and (policy is None or str(task_result.get("policy", "")).strip() == str(policy).strip())
            ),
            "",
        )
        item.setdefault("metadata", {})
        item["metadata"]["policy"] = active_policy
        item["metadata"]["question"] = active_case.get("question")
    return mined


def load_records_from_path(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    if source_path.suffix.lower() == ".jsonl":
        rows = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return list(payload["rows"])
        if isinstance(payload.get("cases"), list):
            return list(payload["cases"])
    raise ValueError(f"unsupported local record container: {source_path}")


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=True) for row in rows]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_sft_records(recipe: ResearchRecipe) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for source in recipe.dataset_sources:
        rows = _load_source_rows(source)
        for index, row in enumerate(rows):
            if source.format != "sharegpt":
                raise ValueError(
                    f"unsupported SFT dataset format '{source.format}' for source {source.name}"
                )
            normalized = normalize_sharegpt_record(row, source_name=source.name)
            if recipe.masking.ratio > 0.0:
                normalized = apply_function_name_masking(
                    normalized,
                    ratio=recipe.masking.ratio,
                    seed=recipe.masking.seed + len(prepared) + index,
                    placeholder_prefix=recipe.masking.placeholder_prefix,
                )
            prepared.append(normalized)
    return prepared


def prepare_preference_records(recipe: ResearchRecipe) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for source in recipe.dataset_sources:
        rows = _load_source_rows(source)
        for row in rows:
            if source.format == "toolpreference":
                prepared.append(
                    normalize_toolpreference_record(row, source_name=source.name)
                )
                continue
            if source.format == "preference_jsonl":
                prompt = _serialize_prompt_part(row.get("prompt"))
                chosen = _serialize_prompt_part(row.get("chosen"))
                rejected = _serialize_prompt_part(row.get("rejected"))
                if not prompt or not chosen or not rejected:
                    raise ValueError(
                        f"preference_jsonl rows require prompt/chosen/rejected fields: {source.name}"
                    )
                prepared.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "source": source.name,
                        "metadata": dict(row.get("metadata", {})),
                    }
                )
                continue
            raise ValueError(
                f"unsupported DPO dataset format '{source.format}' for source {source.name}"
            )
    return prepared


def run_sft_training(recipe: ResearchRecipe, *, dataset_path: str | Path) -> Path:
    datasets_module = _require_dependency(
        "datasets",
        "Install research dependencies with: uv sync --extra research",
    )
    peft_module = _require_dependency(
        "peft",
        "Install research dependencies with: uv sync --extra research",
    )
    torch_module = _require_dependency(
        "torch",
        "Install research dependencies with: uv sync --extra research",
    )
    transformers_module = _require_dependency(
        "transformers",
        "Install research dependencies with: uv sync --extra research",
    )
    trl_module = _require_dependency(
        "trl",
        "Install research dependencies with: uv sync --extra research",
    )

    train_dataset = datasets_module.load_dataset(
        "json", data_files=str(dataset_path), split="train"
    )
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        recipe.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = _training_dtype(recipe, torch_module=torch_module)
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if recipe.load_in_4bit:
        model_kwargs["quantization_config"] = transformers_module.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    if torch_module.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    model = transformers_module.AutoModelForCausalLM.from_pretrained(
        recipe.base_model,
        **model_kwargs,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = _prepare_model_for_quantized_training(
        recipe,
        model=model,
        peft_module=peft_module,
    )

    lora_config = peft_module.LoraConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        bias="none",
        task_type=peft_module.TaskType.CAUSAL_LM,
        target_modules=recipe.target_modules,
    )
    model, lora_config = _attach_trainable_adapter_if_present(
        recipe,
        model=model,
        peft_module=peft_module,
        lora_config=lora_config,
    )

    training_args = _build_sft_training_args(
        transformers_module=transformers_module,
        trl_module=trl_module,
        recipe=recipe,
    )

    trainer_kwargs = _build_sft_trainer_kwargs(
        trl_module=trl_module,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        recipe=recipe,
        lora_config=lora_config,
    )
    trainer = trl_module.SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(recipe.output_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(recipe.output_dir)
    return Path(recipe.output_dir)


def run_dpo_training(recipe: ResearchRecipe, *, dataset_path: str | Path) -> Path:
    datasets_module = _require_dependency(
        "datasets",
        "Install research dependencies with: uv sync --extra research",
    )
    peft_module = _require_dependency(
        "peft",
        "Install research dependencies with: uv sync --extra research",
    )
    torch_module = _require_dependency(
        "torch",
        "Install research dependencies with: uv sync --extra research",
    )
    transformers_module = _require_dependency(
        "transformers",
        "Install research dependencies with: uv sync --extra research",
    )
    trl_module = _require_dependency(
        "trl",
        "Install research dependencies with: uv sync --extra research",
    )

    train_dataset = datasets_module.load_dataset(
        "json", data_files=str(dataset_path), split="train"
    )
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        recipe.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = _training_dtype(recipe, torch_module=torch_module)
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if recipe.load_in_4bit:
        model_kwargs["quantization_config"] = transformers_module.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    if torch_module.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    model = transformers_module.AutoModelForCausalLM.from_pretrained(
        recipe.base_model,
        **model_kwargs,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = _prepare_model_for_quantized_training(
        recipe,
        model=model,
        peft_module=peft_module,
    )

    lora_config = peft_module.LoraConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        bias="none",
        task_type=peft_module.TaskType.CAUSAL_LM,
        target_modules=recipe.target_modules,
    )
    model, lora_config = _attach_trainable_adapter_if_present(
        recipe,
        model=model,
        peft_module=peft_module,
        lora_config=lora_config,
    )
    training_args = _build_dpo_training_args(
        transformers_module=transformers_module,
        trl_module=trl_module,
        recipe=recipe,
    )

    trainer_kwargs = _build_dpo_trainer_kwargs(
        trl_module=trl_module,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        recipe=recipe,
        lora_config=lora_config,
    )
    trainer = trl_module.DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(recipe.output_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(recipe.output_dir)
    return Path(recipe.output_dir)


def _extract_tool_names(tools: list[Any]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            name = str(tool["function"].get("name", "")).strip()
        else:
            name = str(tool.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _deep_replace(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _deep_replace(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_replace(item, replacements) for item in value]
    if isinstance(value, str):
        updated = value
        for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            updated = updated.replace(source, target)
        return updated
    return value


def _serialize_prompt_part(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _pick_first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _coerce_json_payload(payload: Any) -> Any:
    if isinstance(payload, (dict, list)):
        return payload
    if payload is None:
        return None
    if not isinstance(payload, str):
        return payload

    stripped = payload.strip()
    if not stripped:
        return None
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1).strip()
    for candidate in (stripped, _extract_json_candidate(stripped)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return stripped


def _extract_json_candidate(text: str) -> str | None:
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [index for index in (object_start, array_start) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    if text[start] == "{":
        end = text.rfind("}")
    else:
        end = text.rfind("]")
    if end <= start:
        return None
    return text[start : end + 1]


def _extract_tool_calls(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        calls: list[dict[str, Any]] = []
        for item in payload:
            calls.extend(_extract_tool_calls(item))
        return calls
    if isinstance(payload, str):
        return _extract_tool_calls(_coerce_json_payload(payload))
    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("tool_calls"), list):
        calls: list[dict[str, Any]] = []
        for item in payload["tool_calls"]:
            calls.extend(_extract_tool_calls(item))
        return calls

    if isinstance(payload.get("function_call"), dict):
        return _extract_tool_calls(payload["function_call"])

    function_payload = payload.get("function")
    if isinstance(function_payload, dict):
        name = str(function_payload.get("name", "")).strip()
        arguments = function_payload.get("arguments", {})
        return [{"name": name, "arguments": _normalize_arguments(arguments)}]

    if "name" in payload and ("arguments" in payload or "args" in payload or "parameters" in payload):
        arguments = payload.get("arguments", payload.get("args", payload.get("parameters", {})))
        return [
            {
                "name": str(payload.get("name", "")).strip(),
                "arguments": _normalize_arguments(arguments),
            }
        ]

    if payload.get("role") == "assistant" and payload.get("content") is not None:
        return _extract_tool_calls(payload.get("content"))
    if payload.get("role") == "tool" and payload.get("content") is not None:
        return _extract_tool_calls(payload.get("content"))

    return []


def _normalize_arguments(arguments: Any) -> dict[str, Any]:
    value = _coerce_json_payload(arguments)
    if isinstance(value, dict):
        return value
    return {}


def _tool_sequence_to_payload(tool_names: list[str]) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "name": str(tool_name).strip(),
                "arguments": {},
            }
            for tool_name in tool_names
            if str(tool_name).strip()
        ]
    }


def _load_source_rows(source: ResearchDatasetSource) -> list[dict[str, Any]]:
    if source.path:
        rows = load_records_from_path(source.path)
    else:
        if not source.dataset_id:
            raise ValueError(
                f"research source '{source.name}' requires either path or dataset_id"
            )
        datasets_module = _require_dependency(
            "datasets",
            "Install research dependencies with: uv sync --extra research",
        )
        if source.config_name:
            dataset = datasets_module.load_dataset(
                source.dataset_id,
                source.config_name,
                split=source.split,
            )
        else:
            dataset = datasets_module.load_dataset(source.dataset_id, split=source.split)
        row_count = len(dataset)
        limit = row_count if source.limit is None else min(row_count, source.limit)
        rows = [dataset[index] for index in range(limit)]
    if source.limit is not None:
        return list(rows[: source.limit])
    return list(rows)


def _build_sft_trainer_kwargs(
    *,
    trl_module: Any,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    tokenizer: Any,
    recipe: ResearchRecipe,
    lora_config: Any,
) -> dict[str, Any]:
    params = inspect.signature(trl_module.SFTTrainer.__init__).parameters
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": lora_config,
    }
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = recipe.max_seq_length
    if "packing" in params:
        kwargs["packing"] = recipe.packing
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    return kwargs


def _build_sft_training_args(
    *,
    transformers_module: Any,
    trl_module: Any,
    recipe: ResearchRecipe,
) -> Any:
    base_kwargs = {
        "output_dir": recipe.output_dir,
        "learning_rate": recipe.learning_rate,
        "num_train_epochs": recipe.num_train_epochs,
        "per_device_train_batch_size": recipe.per_device_train_batch_size,
        "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
        "warmup_ratio": recipe.warmup_ratio,
        **_training_precision_kwargs(recipe),
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
    }
    sft_config_cls = getattr(trl_module, "SFTConfig", None)
    if sft_config_cls is None:
        return transformers_module.TrainingArguments(**base_kwargs)

    params = inspect.signature(sft_config_cls.__init__).parameters
    sft_kwargs = {key: value for key, value in base_kwargs.items() if key in params}
    if "max_seq_length" in params:
        sft_kwargs["max_seq_length"] = recipe.max_seq_length
    if "packing" in params:
        sft_kwargs["packing"] = recipe.packing
    return sft_config_cls(**sft_kwargs)


def _build_dpo_training_args(
    *,
    transformers_module: Any,
    trl_module: Any,
    recipe: ResearchRecipe,
) -> Any:
    base_kwargs = {
        "output_dir": recipe.output_dir,
        "learning_rate": recipe.learning_rate,
        "num_train_epochs": recipe.num_train_epochs,
        "per_device_train_batch_size": recipe.per_device_train_batch_size,
        "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
        "warmup_ratio": recipe.warmup_ratio,
        **_training_precision_kwargs(recipe),
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
    }
    dpo_config_cls = getattr(trl_module, "DPOConfig", None)
    if dpo_config_cls is None:
        training_args = transformers_module.TrainingArguments(**base_kwargs)
        for field_name in ("model_init_kwargs", "ref_model_init_kwargs"):
            if not hasattr(training_args, field_name):
                setattr(training_args, field_name, None)
        return training_args

    params = inspect.signature(dpo_config_cls.__init__).parameters
    dpo_kwargs = dict(base_kwargs)
    if "max_length" in params:
        dpo_kwargs["max_length"] = recipe.max_seq_length
    if "max_prompt_length" in params:
        dpo_kwargs["max_prompt_length"] = max(256, recipe.max_seq_length // 2)
    return dpo_config_cls(**dpo_kwargs)


def _build_dpo_trainer_kwargs(
    *,
    trl_module: Any,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    tokenizer: Any,
    recipe: ResearchRecipe,
    lora_config: Any,
) -> dict[str, Any]:
    params = inspect.signature(trl_module.DPOTrainer.__init__).parameters
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    if "ref_model" in params:
        kwargs["ref_model"] = None
    if "beta" in params:
        kwargs["beta"] = recipe.beta
    if lora_config is not None and "peft_config" in params:
        kwargs["peft_config"] = lora_config
    if "max_length" in params:
        kwargs["max_length"] = recipe.max_seq_length
    if "max_prompt_length" in params:
        kwargs["max_prompt_length"] = max(256, recipe.max_seq_length // 2)
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    return kwargs


def _require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Missing optional dependency '{module_name}'. {install_hint}") from exc

import json

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Action
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.run import get_tasks
from tau2.utils import DATA_DIR
from tau2.utils.utils import get_dict_hash, show_dict_diff


@pytest.fixture
def base_task_name() -> str:
    return "create_task_1"


@pytest.fixture
def task_with_initial_state_message_history_name() -> str:
    return "update_task_with_message_history"


@pytest.fixture
def task_with_initial_state_initialization_data_name():
    return "update_task_with_initialization_data"


@pytest.fixture
def tasks_dict():
    task_file = DATA_DIR / "tau2" / "domains" / "mock" / "tasks.json"
    with open(task_file, "r") as f:
        task_dicts = json.load(f)
    tasks_dict = {v["id"]: v for v in task_dicts}
    return tasks_dict


def test_get_task_base(base_task_name: str, tasks_dict: dict):
    task_dict = tasks_dict[base_task_name]
    task_instance = get_tasks("mock", task_ids=[base_task_name])[0]
    task_instance_dict = task_instance.model_dump(
        exclude_unset=True, exclude_defaults=True, exclude_none=True
    )
    print(show_dict_diff(task_dict, task_instance_dict))
    assert get_dict_hash(task_dict) == get_dict_hash(task_instance_dict)


def test_get_task_with_initial_state_message_history(
    task_with_initial_state_message_history_name: str, tasks_dict: dict
):
    task_dict = tasks_dict[task_with_initial_state_message_history_name]
    task_instance = get_tasks(
        "mock", task_ids=[task_with_initial_state_message_history_name]
    )[0]
    task_instance_dict = task_instance.model_dump(
        exclude_unset=True, exclude_defaults=True, exclude_none=True
    )
    # FIXME: `ticket: null` is not removed in task_dict, but excluded in task_instance_dict
    print(json.dumps(task_dict, indent=2))
    print(json.dumps(task_instance_dict, indent=2))
    print(show_dict_diff(task_dict, task_instance_dict))
    assert get_dict_hash(task_dict) == get_dict_hash(task_instance_dict)


def test_get_task_with_initial_state_initialization_data(
    task_with_initial_state_initialization_data_name: str, tasks_dict: dict
):
    task_dict = tasks_dict[task_with_initial_state_initialization_data_name]
    task_instance = get_tasks(
        "mock", task_ids=[task_with_initial_state_initialization_data_name]
    )[0]
    task_instance_dict = task_instance.model_dump(
        exclude_unset=True, exclude_defaults=True, exclude_none=True
    )
    print(json.dumps(task_dict, indent=2))
    print(json.dumps(task_instance_dict, indent=2))
    print(show_dict_diff(task_dict, task_instance_dict))
    assert get_dict_hash(task_dict) == get_dict_hash(task_instance_dict)


# ---------------------------------------------------------------------------
# Negative (forbid) action check tests
# ---------------------------------------------------------------------------


def _make_trajectory(*tool_names: str) -> list[AssistantMessage]:
    """Build a minimal trajectory containing one AssistantMessage per tool call."""
    return [
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(name=name, arguments={}, id=f"tc_{i}")],
        )
        for i, name in enumerate(tool_names)
    ]


def _make_trajectory_with_args(tool_name: str, arguments: dict) -> list[AssistantMessage]:
    return [
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(name=tool_name, arguments=arguments, id="tc_0")],
        )
    ]


def test_forbid_action_passes_when_not_performed():
    """A forbidden action check must PASS when the action is absent from the trajectory."""
    action = Action(
        action_id="forbidden_1",
        name="delete_account",
        arguments={},
        forbid=True,
    )
    trajectory = _make_trajectory("get_user_details", "update_account")
    checks = ActionEvaluator.evaluate_actions(trajectory, [action])
    assert len(checks) == 1
    assert checks[0].action_match is True
    assert checks[0].action_reward == 1.0


def test_forbid_action_fails_when_performed():
    """A forbidden action check must FAIL when the action IS present in the trajectory."""
    action = Action(
        action_id="forbidden_2",
        name="delete_account",
        arguments={},
        forbid=True,
    )
    trajectory = _make_trajectory("get_user_details", "delete_account")
    checks = ActionEvaluator.evaluate_actions(trajectory, [action])
    assert len(checks) == 1
    assert checks[0].action_match is False
    assert checks[0].action_reward == 0.0


def test_normal_action_check_unaffected_by_forbid_default():
    """A normal (non-forbidden) action check must still pass when the action IS performed."""
    action = Action(
        action_id="normal_1",
        name="create_task",
        arguments={"title": "Meeting"},
        compare_args=["title"],
    )
    trajectory = _make_trajectory_with_args("create_task", {"title": "Meeting", "priority": "high"})
    checks = ActionEvaluator.evaluate_actions(trajectory, [action])
    assert checks[0].action_match is True


def test_mixed_actions_all_must_be_satisfied():
    """A mix of normal and forbidden actions: all conditions must be individually met."""
    required = Action(action_id="req_1", name="send_email", arguments={})
    forbidden = Action(action_id="forb_1", name="delete_user", arguments={}, forbid=True)

    # Both satisfied: send_email performed, delete_user NOT performed
    trajectory = _make_trajectory("send_email")
    checks = ActionEvaluator.evaluate_actions(trajectory, [required, forbidden])
    assert all(c.action_match for c in checks)

    # Only partially satisfied: delete_user also performed
    trajectory_bad = _make_trajectory("send_email", "delete_user")
    checks_bad = ActionEvaluator.evaluate_actions(trajectory_bad, [required, forbidden])
    assert checks_bad[0].action_match is True   # required: found ✓
    assert checks_bad[1].action_match is False  # forbidden: found ✗

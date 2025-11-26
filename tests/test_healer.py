import json
from unittest.mock import MagicMock
from app.agents.healer import HealerService
from app.agents.models import ExecutionResult
from app.core.chat_service import ChatService
from app.core.models import ChatChunk


def test_healer_generates_and_executes_plan():
    # Mock dependencies
    mock_chat_service = MagicMock(spec=ChatService)
    mock_executor = MagicMock()

    # Setup the HealerService with a mock executor
    healer = HealerService(chat_service=mock_chat_service)
    healer.executor = mock_executor

    # Mock the LLM response
    json_content = json.dumps(
        {
            "files": [{"path": "test.py", "content": "print('hello')"}],
            "command": "python test.py",
        }
    )
    llm_response = f"```json\n{json_content}\n```"

    # Make get_response yield chunks
    def mock_get_response(*args, **kwargs):
        yield ChatChunk(content=llm_response)

    mock_chat_service.get_response.side_effect = mock_get_response

    # Mock execution result
    mock_executor.execute_project.return_value = ExecutionResult(
        success=True, stdout="hello", stderr="", exit_code=0, execution_time_seconds=0.1
    )

    # Run the healer
    generator = healer.heal_code("Write a python script")

    # Consume the generator
    chunks = list(generator)

    # Verify interactions
    assert len(chunks) > 0

    # Check if execute_project was called with correct args
    mock_executor.execute_project.assert_called_once()
    call_args = mock_executor.execute_project.call_args
    assert call_args[0][0] == [{"path": "test.py", "content": "print('hello')"}]
    assert call_args[0][1] == "python test.py"

    # Verify success metric
    success_chunk = chunks[-1]
    assert success_chunk.metrics is not None
    assert success_chunk.metrics.successful is True


def test_healer_retries_on_failure():
    mock_chat_service = MagicMock(spec=ChatService)
    healer = HealerService(chat_service=mock_chat_service, max_attempts=2)
    healer.executor = MagicMock()

    # First response (valid JSON but fails execution)
    json_1 = json.dumps(
        {
            "files": [{"path": "bad.py", "content": "syntax error"}],
            "command": "python bad.py",
        }
    )
    response_1 = f"```json\n{json_1}\n```"

    # Second response (fix)
    json_2 = json.dumps(
        {
            "files": [{"path": "good.py", "content": "print('fixed')"}],
            "command": "python good.py",
        }
    )
    response_2 = f"```json\n{json_2}\n```"

    # Mock LLM responses
    def mock_get_response_sequence(*args, **kwargs):
        # Inspect arguments to see which prompt is being sent
        call_args = kwargs
        message = call_args.get("message", "")
        if not message and args:
            message = args[0]

        # If previous failure is mentioned, return success
        if (
            "previous plan failed" in str(call_args.get("system_message", ""))
            or "previous plan failed" in message
        ):
            yield ChatChunk(content=response_2)
        else:
            yield ChatChunk(content=response_1)

    mock_chat_service.get_response.side_effect = mock_get_response_sequence

    # Mock execution results
    # First attempt fails
    fail_result = ExecutionResult(
        success=False,
        stdout="",
        stderr="SyntaxError",
        exit_code=1,
        execution_time_seconds=0.1,
    )
    # Second attempt succeeds
    success_result = ExecutionResult(
        success=True, stdout="fixed", stderr="", exit_code=0, execution_time_seconds=0.1
    )

    healer.executor.execute_project.side_effect = [fail_result, success_result]

    chunks = list(healer.heal_code("Fix me"))

    # Verify retry logic
    assert healer.executor.execute_project.call_count == 2
    assert chunks[-1].metrics is not None
    assert chunks[-1].metrics.successful is True
    assert chunks[-1].metrics.total_attempts == 2


def test_healer_handles_json_error():
    mock_chat_service = MagicMock(spec=ChatService)
    healer = HealerService(chat_service=mock_chat_service, max_attempts=1)

    # Return invalid JSON
    mock_chat_service.get_response.return_value = iter([ChatChunk(content="Not JSON")])

    chunks = list(healer.heal_code("Make bad json"))

    # Should handle gracefully and return failed metrics
    assert any("Failed to parse plan" in (c.content or "") for c in chunks)
    assert chunks[-1].metrics is not None
    assert chunks[-1].metrics.successful is False

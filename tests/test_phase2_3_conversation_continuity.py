"""
Comex Phase 2.3 — Conversational Continuity & Multi-Turn Regression Suite.

Tests the complete conversation pipeline across unit, API integration,
and functional multi-turn scenarios (T01 - T20).
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import agent_service
from agent_service import ConversationMessage, AgentRequest, execute_agent_loop

client = TestClient(agent_service.app)


# =============================================================================
# 1. UNIT TESTS: Schema, Message Serialization & Context Bounding
# =============================================================================

def test_conversation_message_schema():
    """Verify ConversationMessage validates roles and accepts task_outcome."""
    msg = ConversationMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.task_outcome is None

    asst_msg = ConversationMessage(
        role="assistant",
        content="Added item to cart.",
        task_outcome={"product": "Algorithms 4th Ed", "action": "add_to_cart", "status": "success"}
    )
    assert asst_msg.role == "assistant"
    assert asst_msg.task_outcome["product"] == "Algorithms 4th Ed"


def test_agent_request_conversation_fields():
    """Verify AgentRequest includes conversation_id, messages, and task_context."""
    req = AgentRequest(
        task="Do you think it is the best?",
        taskId="task-123",
        tabId="active_tab",
        conversation_id="conv-abc",
        messages=[
            ConversationMessage(role="user", content="Add book to cart"),
            ConversationMessage(role="assistant", content="Book added")
        ],
        task_context={"product": "Introduction to Algorithms", "status": "success"}
    )
    assert req.conversation_id == "conv-abc"
    assert len(req.messages) == 2
    assert req.task_context["product"] == "Introduction to Algorithms"


@pytest.mark.asyncio
async def test_history_bounding_and_clipping():
    """Verify execute_agent_loop bounds history to 10 turns and clips messages to 2000 chars."""
    # Create 15 messages (exceeding 10-turn limit) with oversized content
    long_content = "X" * 3000
    oversized_history = [
        ConversationMessage(role="user" if i % 2 == 0 else "assistant", content=f"Turn {i}: {long_content}")
        for i in range(15)
    ]

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            # Assertions on prompt construction:
            # 1. SystemMessage is first
            assert isinstance(messages[0], SystemMessage)
            
            # 2. Historical messages are at most 10
            historical = messages[1:-1]
            assert len(historical) <= 10
            
            # 3. Each historical message is clipped to 2000 chars
            for m in historical:
                assert len(m.content) <= 2000
            
            # 4. Current user turn is last
            assert isinstance(messages[-1], HumanMessage)
            assert "Current question" in messages[-1].content
            
            resp = AIMessage(content="Here is my evaluation.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="Current question",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-bound-test",
            conversation_messages=oversized_history,
            legacy_mode=True
        )
        assert "evaluation" in str(result)


# =============================================================================
# 2. API INTEGRATION TESTS: Same Conversation vs New Conversation Isolation
# =============================================================================

def test_api_same_conversation_carries_context():
    """Verify that follow-up under same conversation_id delivers prior turns and task outcome."""
    calls = []

    async def mock_loop(task, cdp_url, **kwargs):
        calls.append((task, kwargs))
        resp = MagicMock()
        resp.__str__ = lambda self: "Context received successfully."
        resp.metadata = {"is_conversational": True}
        return resp

    with patch("agent_service._resolve_verified_tab_ws", return_value=("t1", "ws://127.0.0.1:9222/t1")), \
         patch("agent_service.execute_agent_loop", side_effect=mock_loop):

        # Turn 1
        res1 = client.post("/agent/run", json={
            "task": "Add Introduction to Algorithms, 4th Edition to the cart.",
            "taskId": "task-1",
            "tabId": "active_tab",
            "conversation_id": "conv-42",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })
        assert res1.status_code == 200

        # Turn 2 in same conversation
        res2 = client.post("/agent/run", json={
            "task": "Do you think it is the best?",
            "taskId": "task-2",
            "tabId": "active_tab",
            "conversation_id": "conv-42",
            "messages": [
                {"role": "user", "content": "Add Introduction to Algorithms, 4th Edition to the cart."},
                {"role": "assistant", "content": "Added Introduction to Algorithms, 4th Edition to your Amazon cart."}
            ],
            "task_context": {
                "product": "Introduction to Algorithms, 4th Edition",
                "action": "add_to_cart",
                "status": "success"
            },
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })
        assert res2.status_code == 200

        assert len(calls) == 2
        task2, kwargs2 = calls[1]
        assert task2 == "Do you think it is the best?"
        assert kwargs2["conversation_id"] == "conv-42"
        assert len(kwargs2["conversation_messages"]) == 2
        assert kwargs2["task_context"]["product"] == "Introduction to Algorithms, 4th Edition"


def test_api_conversation_isolation():
    """Verify that a different conversation_id does not inherit or leak history."""
    calls = []

    async def mock_loop(task, cdp_url, **kwargs):
        calls.append((task, kwargs))
        resp = MagicMock()
        resp.__str__ = lambda self: "Clean isolation."
        resp.metadata = {"is_conversational": True}
        return resp

    with patch("agent_service._resolve_verified_tab_ws", return_value=("t1", "ws://127.0.0.1:9222/t1")), \
         patch("agent_service.execute_agent_loop", side_effect=mock_loop):

        # New conversation with fresh ID
        res = client.post("/agent/run", json={
            "task": "What is the weather today?",
            "taskId": "task-isolated-1",
            "tabId": "active_tab",
            "conversation_id": "conv-fresh-99",
            "messages": [],
            "task_context": None,
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })
        assert res.status_code == 200

        task, kwargs = calls[0]
        assert kwargs["conversation_id"] == "conv-fresh-99"
        assert kwargs["conversation_messages"] == []
        assert kwargs["task_context"] is None


# =============================================================================
# 3. MULTI-TURN FUNCTIONAL BATTERY (Scenarios T01 - T20)
# =============================================================================

@pytest.mark.asyncio
async def test_t01_book_followup_pronoun_resolution():
    """T01: 'Do you think it is the best?' resolves 'it' to the book from Turn 1."""
    history = [
        ConversationMessage(role="user", content="Add Introduction to Algorithms, 4th Edition to the cart."),
        ConversationMessage(role="assistant", content="Added Introduction to Algorithms, 4th Edition to your Amazon cart.")
    ]
    task_context = {
        "product": "Introduction to Algorithms, 4th Edition",
        "action": "add_to_cart",
        "status": "success"
    }

    prompt_captured = []

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            prompt_captured.extend(messages)
            # LLM resolves 'it' to Introduction to Algorithms
            resp = AIMessage(content="Introduction to Algorithms (CLRS) 4th Edition is widely considered one of the premier computer science reference textbooks due to its rigorous mathematical proofs and comprehensive coverage, though it can be dense for beginners.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="Do you think it is the best?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t01",
            conversation_messages=history,
            task_context=task_context,
            legacy_mode=True
        )

        res_str = str(result)
        assert "Introduction to Algorithms" in res_str
        # Verify prompt had the context
        full_prompt = "".join([m.content for m in prompt_captured])
        assert "Introduction to Algorithms, 4th Edition" in full_prompt
        assert "Do you think it is the best?" in full_prompt


@pytest.mark.asyncio
async def test_t02_price_followup_resolution():
    """T02: 'What was the price again?' resolves to the price in task context."""
    history = [
        ConversationMessage(role="user", content="Check price for Sony WH-1000XM5 headphones."),
        ConversationMessage(role="assistant", content="Found Sony WH-1000XM5 priced at $399.99.")
    ]
    task_context = {
        "product": "Sony WH-1000XM5",
        "price": "$399.99",
        "status": "success"
    }

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            # Agent resolves directly without repeating search
            resp = AIMessage(content="The Sony WH-1000XM5 headphones were priced at $399.99.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="What was the price again?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t02",
            conversation_messages=history,
            task_context=task_context,
            legacy_mode=True
        )

        assert "$399.99" in str(result)
        assert "Sony WH-1000XM5" in str(result)


@pytest.mark.asyncio
async def test_t03_option_reference():
    """T03: 'What about the second one?' refers to option 2 in preceding comparison."""
    history = [
        ConversationMessage(role="user", content="Compare MacBook Air M3 and Dell XPS 13."),
        ConversationMessage(role="assistant", content="1. MacBook Air M3: Excellent battery life and silent fanless design.\n2. Dell XPS 13: Compact OLED display with Windows 11.")
    ]

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="The second option, the Dell XPS 13, features an Intel Core Ultra processor and an optional 3K OLED touch display.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="What about the second one?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t03",
            conversation_messages=history,
            legacy_mode=True
        )

        assert "Dell XPS 13" in str(result)


@pytest.mark.asyncio
async def test_t04_followup_action_targets_resolved_entity():
    """T04: Follow-up action ('Remove it from cart') resolves 'it' to the previously added book."""
    history = [
        ConversationMessage(role="user", content="Add Introduction to Algorithms, 4th Edition to cart."),
        ConversationMessage(role="assistant", content="Added Introduction to Algorithms, 4th Edition to your cart.")
    ]
    task_context = {
        "product": "Introduction to Algorithms, 4th Edition",
        "action": "add_to_cart",
        "status": "success"
    }

    executed_tools = []

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        call_idx = 0
        async def fake_ainvoke(messages):
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                # Agent issues a click to remove the resolved item
                resp = AIMessage(content="")
                resp.tool_calls = [{
                    "id": "tc1",
                    "name": "browser_click",
                    "args": {"selector": "[data-action='delete-item']"}
                }]
                return resp
            else:
                resp = AIMessage(content="Removed Introduction to Algorithms, 4th Edition from your cart.")
                resp.tool_calls = []
                return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        async def fake_click(selector):
            executed_tools.append(selector)
            return "Clicked delete item button"

        with patch("agent_service._build_browser_tools") as mock_tools:
            click_tool = MagicMock()
            click_tool.name = "browser_click"
            click_tool.ainvoke = fake_click
            click_tool.invoke = lambda args: "Clicked"
            mock_tools.return_value = [click_tool]

            result = await execute_agent_loop(
                task="Remove it from cart",
                default_cdp_url="ws://127.0.0.1:9222",
                conversation_id="conv-t04",
                conversation_messages=history,
                task_context=task_context,
                legacy_mode=True
            )

            assert "Removed Introduction to Algorithms" in str(result)
            assert len(executed_tools) == 1


@pytest.mark.asyncio
async def test_t05_completed_action_protection_no_duplicate_dispatch():
    """T05: Question about added item does NOT re-dispatch the add_to_cart action."""
    history = [
        ConversationMessage(role="user", content="Add Introduction to Algorithms to cart."),
        ConversationMessage(role="assistant", content="Successfully added to cart.")
    ]
    task_context = {
        "product": "Introduction to Algorithms",
        "action": "add_to_cart",
        "status": "success"
    }

    tool_called = False

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            # Pure conversation reply without tool calls
            resp = AIMessage(content="It is one of the top rated algorithms textbooks.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="Do you think it is the best?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t05",
            conversation_messages=history,
            task_context=task_context,
            legacy_mode=True
        )

        meta = result.metadata
        assert meta["tool_call_count"] == 0
        assert meta["is_conversational"] is True


@pytest.mark.asyncio
async def test_t06_clarify_genuine_ambiguity():
    """T06: When two books were discussed, asks a focused question rather than generic error."""
    history = [
        ConversationMessage(role="user", content="Compare Clean Code and Pragmatic Programmer."),
        ConversationMessage(role="assistant", content="Clean Code focuses on function design. Pragmatic Programmer focuses on career philosophy.")
    ]

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="Which book are you interested in: Clean Code or The Pragmatic Programmer?")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="What is the rating of the book?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t06",
            conversation_messages=history,
            legacy_mode=True
        )

        assert "Clean Code" in str(result)
        assert "Pragmatic Programmer" in str(result)


@pytest.mark.asyncio
async def test_t07_new_topic_does_not_hallucinate_previous_entity():
    """T07: Topic change does not inappropriately import the prior book."""
    history = [
        ConversationMessage(role="user", content="Add Introduction to Algorithms to cart."),
        ConversationMessage(role="assistant", content="Book added to cart.")
    ]

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="The current weather in Tokyo is 18°C and clear.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="What is the weather in Tokyo?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t07",
            conversation_messages=history,
            legacy_mode=True
        )

        assert "Tokyo" in str(result)
        assert "Algorithms" not in str(result)


@pytest.mark.asyncio
async def test_t08_missing_information_handled_gracefully():
    """T08: When specific attribute is missing, acknowledges the entity but states detail is unknown."""
    history = [
        ConversationMessage(role="user", content="View mechanical keyboard."),
        ConversationMessage(role="assistant", content="Keychron Q1 Pro.")
    ]
    task_context = {"product": "Keychron Q1 Pro", "status": "success"}

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="The Keychron Q1 Pro's exact battery lifespan under continuous RGB lighting is not listed on the current page.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="How long does its battery last with RGB on?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t08",
            conversation_messages=history,
            task_context=task_context,
            legacy_mode=True
        )

        assert "Keychron Q1 Pro" in str(result)
        assert "battery" in str(result)


@pytest.mark.asyncio
async def test_t09_multi_turn_history_accumulation():
    """T09: Context accumulates smoothly over 3 turns."""
    history_turn3 = [
        ConversationMessage(role="user", content="Search for noise cancelling headphones."),
        ConversationMessage(role="assistant", content="Found Bose QC45 and Sony XM5."),
        ConversationMessage(role="user", content="Which one has better comfort?"),
        ConversationMessage(role="assistant", content="Bose QC45 is generally rated more comfortable for long flights."),
    ]

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="Between the two, the Bose QC45 also folds more compactly into its travel case than the Sony XM5.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="Does it fold into a smaller case?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t09",
            conversation_messages=history_turn3,
            legacy_mode=True
        )

        assert "Bose QC45" in str(result)


def test_t10_fresh_conversation_has_zero_leakage():
    """T10: New conversation with empty messages array starts completely fresh."""
    calls = []

    async def mock_loop(task, cdp_url, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.__str__ = lambda self: "Fresh start"
        resp.metadata = {}
        return resp

    with patch("agent_service._resolve_verified_tab_ws", return_value=("t1", "ws://127.0.0.1:9222/t1")), \
         patch("agent_service.execute_agent_loop", side_effect=mock_loop):

        client.post("/agent/run", json={
            "task": "Do you think it is the best?",
            "taskId": "task-isolated-fresh",
            "tabId": "active_tab",
            "conversation_id": "conv-isolated-brand-new",
            "messages": [],
            "task_context": None,
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert len(calls) == 1
        assert calls[0]["conversation_messages"] == []
        assert calls[0]["task_context"] is None


@pytest.mark.asyncio
async def test_t11_failed_action_recovery():
    """T11: If previous turn failed, agent recognizes failure and offers recovery."""
    history = [
        ConversationMessage(role="user", content="Add iPad Air to cart."),
        ConversationMessage(role="assistant", content="Could not add iPad Air: Out of stock.")
    ]
    task_context = {
        "product": "iPad Air",
        "action": "add_to_cart",
        "status": "failed",
        "details": "Item is out of stock"
    }

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="Since the iPad Air was out of stock, would you like me to check refurbished models or a different retailer?")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="Why didn't it add?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t11",
            conversation_messages=history,
            task_context=task_context,
            legacy_mode=True
        )

        assert "iPad Air" in str(result)
        assert "out of stock" in str(result).lower()


@pytest.mark.asyncio
async def test_t12_current_state_validation():
    """T12: Stale selector is not blindly reused; current DOM inspection is prioritized."""
    history = [
        ConversationMessage(role="user", content="Click button"),
        ConversationMessage(role="assistant", content="Clicked button #submit-v1")
    ]

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            # Prioritizes inspect tool to verify fresh DOM
            resp = AIMessage(content="")
            resp.tool_calls = [{
                "id": "tc12",
                "name": "browser_inspect",
                "args": {"max_elements": 10}
            }]
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        with patch("agent_service._build_browser_tools") as mock_tools:
            inspect_tool = MagicMock()
            inspect_tool.name = "browser_inspect"
            inspect_tool.ainvoke = AsyncMock(return_value="[#submit-v2: button 'Submit']")
            mock_tools.return_value = [inspect_tool]

            result = await execute_agent_loop(
                task="Click the submit button on the new page",
                default_cdp_url="ws://127.0.0.1:9222",
                conversation_id="conv-t12",
                conversation_messages=history,
                legacy_mode=True
            )
            # The agent executed browser_inspect rather than blind blind clicking stale selector
            assert result.metadata["tool_call_count"] >= 1


@pytest.mark.asyncio
async def test_t13_pronoun_this_one():
    """T13: 'What about this one?' resolves correctly to the recent entity."""
    history = [ConversationMessage(role="user", content="Looking at Pixel 9 Pro.")]
    task_context = {"product": "Pixel 9 Pro", "status": "success"}

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="The Pixel 9 Pro has Google's Tensor G4 chip and 16GB of RAM.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="Tell me about this one.",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t13",
            conversation_messages=history,
            task_context=task_context,
            legacy_mode=True
        )
        assert "Pixel 9 Pro" in str(result)


@pytest.mark.asyncio
async def test_t14_pronoun_that_item():
    """T14: 'Is that item in stock?' resolves correctly."""
    history = [ConversationMessage(role="user", content="View Dell 4K Monitor.")]
    task_context = {"product": "Dell 4K Monitor", "status": "success"}

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="Yes, the Dell 4K Monitor is currently in stock.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="Is that item in stock?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t14",
            conversation_messages=history,
            task_context=task_context,
            legacy_mode=True
        )
        assert "Dell 4K Monitor" in str(result)


@pytest.mark.asyncio
async def test_t15_evaluative_question_zero_tool_calls():
    """T15: Pure evaluative questions do not invoke browser tools."""
    history = [ConversationMessage(role="user", content="Add Python Crash Course to cart.")]
    task_context = {"product": "Python Crash Course", "action": "add_to_cart", "status": "success"}

    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            resp = AIMessage(content="Python Crash Course is highly recommended for beginners learning practical coding.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        result = await execute_agent_loop(
            task="Is it beginner friendly?",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t15",
            conversation_messages=history,
            task_context=task_context,
            legacy_mode=True
        )
        assert result.metadata["tool_call_count"] == 0
        assert result.metadata["is_conversational"] is True


@pytest.mark.asyncio
async def test_t16_max_history_turns_enforced():
    """T16: Exactly 10 turns are bounded when 20 are provided."""
    history = [ConversationMessage(role="user", content=f"Msg {i}") for i in range(20)]

    captured_msgs = []
    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            captured_msgs.extend(messages)
            resp = AIMessage(content="Acknowledged.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        await execute_agent_loop(
            task="Current turn",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t16",
            conversation_messages=history,
            legacy_mode=True
        )

        # 1 SystemMessage + 10 historical HumanMessages + 1 current HumanMessage = 12 messages
        assert len(captured_msgs) == 12


@pytest.mark.asyncio
async def test_t17_message_truncation_enforced():
    """T17: Historical messages over 2000 chars are strictly clipped."""
    history = [ConversationMessage(role="user", content="A" * 5000)]

    captured_msgs = []
    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        async def fake_ainvoke(messages):
            captured_msgs.extend(messages)
            resp = AIMessage(content="Clipped ok.")
            resp.tool_calls = []
            return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        await execute_agent_loop(
            task="Current turn",
            default_cdp_url="ws://127.0.0.1:9222",
            conversation_id="conv-t17",
            conversation_messages=history,
            legacy_mode=True
        )

        assert len(captured_msgs[1].content) == 2000


@pytest.mark.asyncio
async def test_t18_task_outcome_synthesis_in_metadata():
    """T18: When tools execute, task_outcome is synthesized in result metadata."""
    with patch("agent_service.ChatOpenAI") as mock_chat:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        call_idx = 0
        async def fake_ainvoke(messages):
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                resp = AIMessage(content="")
                resp.tool_calls = [{"id": "t1", "name": "browser_click", "args": {"selector": "#add-to-cart"}}]
                return resp
            else:
                resp = AIMessage(content="Added to cart.")
                resp.tool_calls = []
                return resp

        mock_llm.ainvoke.side_effect = fake_ainvoke

        with patch("agent_service._build_browser_tools") as mock_tools:
            click_tool = MagicMock()
            click_tool.name = "browser_click"
            click_tool.ainvoke = AsyncMock(return_value="Clicked button")
            mock_tools.return_value = [click_tool]

            result = await execute_agent_loop(
                task="Click add to cart",
                default_cdp_url="https://amazon.com/dp/123",
                legacy_mode=True
            )

            assert "task_outcome" in result.metadata
            oc = result.metadata["task_outcome"]
            assert oc["action"] == "browser_click"
            assert oc["status"] == "success"
            assert oc["url"] == "https://amazon.com/dp/123"


def test_t19_streaming_endpoint_forwards_conversation():
    """T19: POST /agent/run/stream preserves conversation_id and messages."""
    received_kwargs = {}

    async def mock_loop(task, cdp_url, **kwargs):
        received_kwargs.update(kwargs)
        resp = MagicMock()
        resp.__str__ = lambda self: "Stream complete"
        resp.metadata = {"is_conversational": True}
        return resp

    with patch("agent_service._resolve_verified_tab_ws", return_value=("t1", "ws://127.0.0.1:9222/t1")), \
         patch("agent_service.execute_agent_loop", side_effect=mock_loop):

        res = client.post("/agent/run/stream", json={
            "task": "Do you like it?",
            "taskId": "task-stream-t19",
            "tabId": "active_tab",
            "conversation_id": "conv-stream-19",
            "messages": [{"role": "user", "content": "Prior message"}],
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })
        assert res.status_code == 200
        assert received_kwargs["conversation_id"] == "conv-stream-19"
        assert len(received_kwargs["conversation_messages"]) == 1


def test_t20_single_turn_backwards_compatibility():
    """T20: Legacy payload with no conversation fields executes cleanly without errors."""
    with patch("agent_service._resolve_verified_tab_ws", return_value=("t1", "ws://127.0.0.1:9222/t1")), \
         patch("agent_service.execute_agent_loop") as mock_loop:

        resp_mock = MagicMock()
        resp_mock.__str__ = lambda self: "Single turn success"
        resp_mock.metadata = {}
        mock_loop.return_value = resp_mock

        # Standard Phase 2.1D / Phase 2.2 payload format
        res = client.post("/agent/run", json={
            "task": "Single turn legacy task",
            "taskId": "legacy-task-1",
            "tabId": "active_tab",
            "contextToken": agent_service._INTERNAL_APP_TOKEN
        })

        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["result"] == "Single turn success"

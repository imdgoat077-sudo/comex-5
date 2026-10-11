"""
Reproduction Test for Comex Phase 2.3 — Conversational Continuity Failure.

Stage A: Reproduce and trace the multi-turn failure where Comex treats Turn 2
as an isolated request without context from Turn 1.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import agent_service


def test_stage_a_reproduce_multi_turn_context_loss():
    """
    Demonstrates that:
    1. Turn 1 executes a browser task (e.g., adding book).
    2. Turn 2 arrives ("Do you think it is the best?").
    3. The agent receives only Turn 2's text with ZERO context from Turn 1.
    4. The prompt completely lacks the book name, prior assistant response, and verified outcome.
    """
    client = TestClient(agent_service.app)
    
    received_messages_turn1 = []
    received_messages_turn2 = []

    class Result1:
        def __str__(self):
            return "Added Introduction to Algorithms, 4th Edition to your Amazon cart."
        metadata = {
            "tool_call_count": 2,
            "is_conversational": False,
            "verified_result": {
                "product": "Introduction to Algorithms, 4th Edition",
                "action": "add_to_cart",
                "status": "success",
            }
        }

    class Result2:
        def __str__(self):
            return "I'm not sure which book or item you are referring to. Please specify the context."
        metadata = {
            "tool_call_count": 0,
            "is_conversational": True,
        }

    with patch("agent_service._resolve_verified_tab_ws", return_value=("target-1", "ws://127.0.0.1:9222/devtools/page/target-1")), \
         patch("agent_service.execute_agent_loop", new_callable=AsyncMock) as mock_loop:

        mock_loop.side_effect = [Result1(), Result2()]

        # --- TURN 1 ---
        # Payload sent by current frontend (ChatDrawer.jsx lines 395-401)
        payload_turn1 = {
            "task": "Add Introduction to Algorithms, 4th Edition to the cart.",
            "taskId": "task-1001",
            "tabId": "active_tab",
            "url": "https://www.amazon.com",
            "contextToken": agent_service._INTERNAL_APP_TOKEN,
        }
        res1 = client.post("/agent/run", json=payload_turn1)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["success"] is True
        assert "Introduction to Algorithms, 4th Edition" in data1["result"]

        # Inspect what execute_agent_loop was called with in Turn 1
        args1, kwargs1 = mock_loop.call_args_list[0]
        called_task_turn1 = args1[0]
        assert "Introduction to Algorithms" in called_task_turn1

        # --- TURN 2 ---
        # User in the same chat says: "Do you think it is the best?"
        # In current frontend implementation, this is what is sent:
        payload_turn2 = {
            "task": "Do you think it is the best?",
            "taskId": "task-1002",
            "tabId": "active_tab",
            "url": "https://www.amazon.com",
            "contextToken": agent_service._INTERNAL_APP_TOKEN,
        }
        res2 = client.post("/agent/run", json=payload_turn2)
        assert res2.status_code == 200
        data2 = res2.json()

        # Inspect what execute_agent_loop was called with in Turn 2
        args2, kwargs2 = mock_loop.call_args_list[1]
        called_task_turn2 = args2[0]

        # VERIFY THE DEFECT:
        # In Turn 2, execute_agent_loop receives ONLY "Do you think it is the best?"
        # It has NO reference to the book, NO messages array, NO conversation_id!
        print("\n[DIAGNOSTIC] Turn 1 Task received:", called_task_turn1)
        print("[DIAGNOSTIC] Turn 2 Task received:", called_task_turn2)
        print("[DIAGNOSTIC] Turn 2 Kwargs received:", kwargs2)

        # Defect assertions:
        assert "Introduction to Algorithms" not in called_task_turn2
        assert kwargs2.get("conversation_id") is None or kwargs2.get("conversation_messages") is None


def test_stage_c_verified_multi_turn_continuity_resolution():
    """
    Demonstrates that with Phase 2.3 conversational continuity:
    1. Turn 1 executes under conversation_id 'conv-book-test'.
    2. Turn 2 arrives with conversation_id, messages history, and task_context.
    3. execute_agent_loop receives the full conversation history and verified outcome.
    4. The prompt resolves 'it' to 'Introduction to Algorithms, 4th Edition'.
    """
    client = TestClient(agent_service.app)

    captured_prompt_messages = []

    async def mock_loop(task, cdp_url, **kwargs):
        # Emulate execute_agent_loop message assembly
        conv_msgs = kwargs.get("conversation_messages") or []
        task_ctx = kwargs.get("task_context")
        
        prompt_msgs = [SystemMessage(content=agent_service.SYSTEM_PROMPT)]
        for m in conv_msgs[-10:]:
            r = m.role if hasattr(m, "role") else m.get("role", "")
            c = m.content if hasattr(m, "content") else m.get("content", "")
            if r == "user":
                prompt_msgs.append(HumanMessage(content=c))
            elif r == "assistant":
                prompt_msgs.append(AIMessage(content=c))
        
        user_prompt_parts = []
        if task_ctx:
            user_prompt_parts.append(
                f"[RECENT TASK OUTCOME (VERIFIED)]\n"
                f"- Product / Target: {task_ctx.get('product')}\n"
                f"- Action: {task_ctx.get('action')}\n"
                f"- Status: {task_ctx.get('status')}\n"
                f"[/RECENT TASK OUTCOME]"
            )
        user_prompt_parts.append(f"User Task: {task}")
        prompt_msgs.append(HumanMessage(content="\n\n".join(user_prompt_parts)))
        
        captured_prompt_messages.append(prompt_msgs)
        
        resp = MagicMock()
        resp.__str__ = lambda self: "Introduction to Algorithms, 4th Edition (CLRS) is widely considered one of the most comprehensive algorithm textbooks."
        resp.metadata = {"is_conversational": True, "tool_call_count": 0}
        return resp

    with patch("agent_service._resolve_verified_tab_ws", return_value=("target-1", "ws://127.0.0.1:9222/devtools/page/target-1")), \
         patch("agent_service.execute_agent_loop", side_effect=mock_loop):

        # --- TURN 1 ---
        res1 = client.post("/agent/run", json={
            "task": "Add Introduction to Algorithms, 4th Edition to the cart.",
            "taskId": "task-1001",
            "tabId": "active_tab",
            "conversation_id": "conv-book-test",
            "messages": [],
            "contextToken": agent_service._INTERNAL_APP_TOKEN,
        })
        assert res1.status_code == 200

        # --- TURN 2 (in same conversation) ---
        res2 = client.post("/agent/run", json={
            "task": "Do you think it is the best?",
            "taskId": "task-1002",
            "tabId": "active_tab",
            "conversation_id": "conv-book-test",
            "messages": [
                {"role": "user", "content": "Add Introduction to Algorithms, 4th Edition to the cart."},
                {"role": "assistant", "content": "Added Introduction to Algorithms, 4th Edition to your Amazon cart."}
            ],
            "task_context": {
                "product": "Introduction to Algorithms, 4th Edition",
                "action": "add_to_cart",
                "status": "success",
                "price": "$89.99"
            },
            "contextToken": agent_service._INTERNAL_APP_TOKEN,
        })
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["success"] is True

        # Assert Turn 2 received full context
        assert len(captured_prompt_messages) == 2
        turn2_prompts = captured_prompt_messages[1]

        # Verify prompt contains conversation history
        contents = [m.content for m in turn2_prompts]
        assert any("Introduction to Algorithms, 4th Edition" in c for c in contents)
        assert any("Do you think it is the best?" in c for c in contents)
        assert any("[RECENT TASK OUTCOME (VERIFIED)]" in c for c in contents)

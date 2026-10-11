#!/usr/bin/env python3
"""
Standalone test for browser-use CDP connection.
Tests that browser-use can attach to the existing Electron browser
and execute a simple task on https://example.com.
"""
import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, Browser, ChatOpenAI

load_dotenv()

CDP_URL = os.getenv('CDP_URL', 'http://127.0.0.1:9222')
ROUTER_API_KEY = os.getenv('ROUTER_API_KEY', 'sk-test-key')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'http://localhost:20128/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'ag/gemini-3.8-flash')

async def test_browser_use():
    print(f"Connecting to CDP: {CDP_URL}")
    print(f"LLM: {MODEL_NAME} @ {OPENAI_BASE_URL}")
    
    # Initialize LLM using browser-use's ChatOpenAI
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=OPENAI_BASE_URL,
        api_key=ROUTER_API_KEY,
    )
    
    # Connect to existing Electron browser via CDP
    browser = Browser(cdp_url=CDP_URL)
    
    # Create agent with test task
    task = "Go to https://example.com and return the page title and the text of the main heading."
    
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
    )
    
    print(f"\nRunning task: {task}")
    result = await agent.run()
    
    print(f"\n--- RESULT ---")
    print(result)
    print(f"\nTest completed successfully!")

if __name__ == '__main__':
    asyncio.run(test_browser_use())

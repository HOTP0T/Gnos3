---
description: "Create optimized Open WebUI plugins (tools, functions, pipes, filters, actions)"
---

# Open WebUI Plugin Development Expert

You are an expert at creating plugins for Open WebUI. When the user asks you to create a plugin, follow this guide to write optimized, functional code.

## Plugin Types Overview

Open WebUI has **4 main plugin types**:

1. **Tools** - Extend LLM abilities (weather, APIs, calculations, web scraping)
2. **Pipe Functions** - Create custom models/agents or proxy external APIs
3. **Filter Functions** - Modify input (inlet) and output (outlet) of messages
4. **Action Functions** - Add custom buttons to chat messages

## Required Frontmatter (Top-Level Docstring)

Every plugin MUST start with metadata:

```python
"""
title: Plugin Name
author: Your Name
author_url: https://your-website.com
git_url: https://github.com/username/repo.git
description: Brief description of what this plugin does
required_open_webui_version: 0.4.0
requirements: requests, beautifulsoup4, aiohttp
version: 1.0.0
licence: MIT
"""
```

---

## 1. TOOLS

Tools extend LLM capabilities. Use class `Tools`.

### Basic Tool Structure

```python
"""
title: My Tool
author: username
version: 1.0.0
description: Description of tool functionality
required_open_webui_version: 0.4.0
requirements: requests
"""

from pydantic import BaseModel, Field
from typing import Optional
import asyncio

class Tools:
    class Valves(BaseModel):
        """Admin-configurable settings"""
        api_key: str = Field(default="", description="API key for the service")
        base_url: str = Field(default="https://api.example.com", description="API base URL")

    class UserValves(BaseModel):
        """User-configurable settings (optional)"""
        preference: str = Field(default="default", description="User preference")

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False  # Set True to enable auto-citations, False for manual

    def my_tool_function(
        self,
        query: str,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> str:
        """
        Brief description of what this tool does.
        :param query: The search query to process
        :return: The result string
        """
        # Access valve settings
        api_key = self.valves.api_key

        # Access user info
        if __user__:
            user_name = __user__.get("name", "Unknown")
            user_valves = __user__.get("valves", None)

        return f"Result for: {query}"
```

### Async Tool with Event Emitter

```python
async def search_web(
    self,
    query: str,
    __event_emitter__=None,
) -> str:
    """
    Search the web for information.
    :param query: Search query
    """
    if __event_emitter__:
        await __event_emitter__({
            "type": "status",
            "data": {"description": "Searching...", "done": False}
        })

    # Do the work...
    result = "Search results here"

    if __event_emitter__:
        await __event_emitter__({
            "type": "status",
            "data": {"description": "Search complete!", "done": True}
        })

    return result
```

### Tool Optional Parameters (Dependency Injection)

These special parameters are automatically injected when present in the function signature:

- `__event_emitter__` - Emit real-time events to UI
- `__event_call__` - Handle user interactions (input dialogs)
- `__user__` - User information dict (id, name, role, email, valves)
- `__metadata__` - Chat metadata (chat_id, message_id, session_id)
- `__messages__` - Previous message history
- `__files__` - Attached file objects
- `__model__` - Model configuration
- `__oauth_token__` - User's OAuth credentials
- `__request__` - FastAPI request object

### Event Types for Tools

```python
# Status (works in both Default and Native modes)
await __event_emitter__({
    "type": "status",
    "data": {"description": "Processing...", "done": False, "hidden": False}
})

# Message (Default mode only - breaks in Native mode!)
await __event_emitter__({
    "type": "message",
    "data": {"content": "Appended content"}
})

# Citation (both modes, requires self.citation = False)
await __event_emitter__({
    "type": "citation",
    "data": {
        "document": ["Content being cited"],
        "metadata": [{"source": "Title", "url": "https://..."}],
        "source": {"name": "Source Name", "url": "https://..."}
    }
})

# Notification
await __event_emitter__({
    "type": "notification",
    "data": {"type": "info", "content": "Task completed!"}  # type: info, warning, error
})

# Follow-up suggestions
await __event_emitter__({
    "type": "chat:message:follow_ups",
    "data": {"follow_ups": ["What about X?", "Tell me more"]}
})

# Error
await __event_emitter__({
    "type": "chat:message:error",
    "data": {"content": "Error message here"}
})
```

---

## 2. PIPE FUNCTIONS

Pipes create custom models/agents that appear in the model selector. Use class `Pipe`.

### Basic Pipe

```python
"""
title: My Custom Model
author: username
version: 1.0.0
"""

from pydantic import BaseModel, Field

class Pipe:
    class Valves(BaseModel):
        model_name: str = Field(default="custom-model")

    def __init__(self):
        self.valves = self.Valves()

    def pipe(self, body: dict) -> str:
        """Process the chat completion request"""
        messages = body.get("messages", [])
        last_message = messages[-1]["content"] if messages else ""
        return f"Response to: {last_message}"
```

### Manifold Pipe (Multiple Models)

```python
class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="")
        OPENAI_API_BASE_URL: str = Field(default="https://api.openai.com/v1")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        """Return list of available models (manifold)"""
        return [
            {"id": "gpt-4", "name": "GPT-4"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
        ]

    def pipe(self, body: dict) -> str:
        model = body.get("model", "")  # Will be "pipe_id.gpt-4" format
        # Extract actual model ID after the dot
        model_id = model.split(".")[-1] if "." in model else model
        return f"Using model: {model_id}"
```

### Async Pipe with Streaming

```python
import requests
from typing import Generator

class Pipe:
    class Valves(BaseModel):
        API_KEY: str = Field(default="")
        API_URL: str = Field(default="https://api.example.com/v1")

    def __init__(self):
        self.valves = self.Valves()

    def pipe(self, body: dict, __user__: dict = None) -> Generator:
        """Stream response from external API"""
        headers = {
            "Authorization": f"Bearer {self.valves.API_KEY}",
            "Content-Type": "application/json"
        }

        if body.get("stream", False):
            response = requests.post(
                f"{self.valves.API_URL}/chat/completions",
                json=body,
                headers=headers,
                stream=True
            )
            return response.iter_lines()
        else:
            response = requests.post(
                f"{self.valves.API_URL}/chat/completions",
                json=body,
                headers=headers
            )
            return response.json()
```

### Pipe Using Internal Open WebUI Functions

```python
from fastapi import Request
from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion

class Pipe:
    def __init__(self):
        pass

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
    ) -> str:
        user = Users.get_user_by_id(__user__["id"])
        body["model"] = "llama3.2:latest"
        return await generate_chat_completion(__request__, body, user)
```

---

## 3. FILTER FUNCTIONS

Filters modify messages before (inlet) and after (outlet) LLM processing. Use class `Filter`.

**IMPORTANT**: Filters are different from Pipes! inlet/outlet only work in `class Filter`, NOT in `class Pipe`.

### Basic Filter

```python
"""
title: Message Filter
author: username
version: 1.0.0
"""

from pydantic import BaseModel, Field
from typing import Optional

class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Filter priority (lower = first)")
        enabled: bool = Field(default=True)

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Modify request BEFORE sending to LLM.
        :param body: The request body containing messages
        :return: Modified body
        """
        # Add system message
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant."
        }
        body.setdefault("messages", []).insert(0, system_msg)
        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Modify response AFTER receiving from LLM.
        :param body: The response body
        :return: Modified body
        """
        # Clean up or transform output
        for msg in body.get("messages", []):
            if msg.get("role") == "assistant":
                msg["content"] = msg["content"].replace("AI:", "Assistant:")
        return body
```

### Stream Filter (v0.5.17+)

```python
class Filter:
    def __init__(self):
        self.valves = self.Valves()

    def stream(self, event: dict) -> dict:
        """
        Intercept individual response chunks during streaming.
        :param event: Stream event with delta content
        :return: Modified event
        """
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta:
                # Remove emojis from streamed content
                delta["content"] = delta["content"].replace("😊", "")
        return event
```

### Toggle Filter with Icon (v0.6.10+)

```python
class Filter:
    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True  # Creates UI toggle switch
        self.icon = """data:image/svg+xml;base64,PHN2Zy..."""  # SVG icon

    async def inlet(self, body: dict, __event_emitter__, __user__=None) -> dict:
        await __event_emitter__({
            "type": "status",
            "data": {"description": "Filter activated!", "done": True}
        })
        return body
```

### Dropdown Valves Configuration

```python
from pydantic import BaseModel, Field

class Filter:
    class Valves(BaseModel):
        mode: str = Field(
            default="standard",
            description="Processing mode",
            json_schema_extra={"enum": ["standard", "verbose", "minimal"]}
        )
        language: str = Field(
            default="en",
            description="Output language",
            json_schema_extra={"enum": ["en", "es", "fr", "de"]}
        )
```

---

## 4. ACTION FUNCTIONS

Actions add custom buttons to chat messages. Use class `Action`.

### Basic Action

```python
"""
title: Summarize Action
author: username
version: 1.0.0
icon_url: data:image/svg+xml;base64,PHN2Zy...
"""

from pydantic import BaseModel
from typing import Optional

class Action:
    class Valves(BaseModel):
        max_length: int = 100

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        """
        Process the action when button is clicked.
        :param body: Message data and context
        """
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "Summarizing...", "done": False}
            })

        content = body.get("content", "")
        summary = content[:self.valves.max_length] + "..."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "Done!", "done": True}
            })

        return {"content": f"Summary: {summary}"}
```

### Action with User Input

```python
async def action(
    self,
    body: dict,
    __event_call__=None,
    __event_emitter__=None,
) -> Optional[dict]:
    # Ask for confirmation
    confirmed = await __event_call__({
        "type": "confirmation",
        "data": {
            "title": "Confirm Action",
            "message": "Are you sure you want to proceed?"
        }
    })

    if not confirmed:
        return {"content": "Action cancelled"}

    # Get user input
    user_input = await __event_call__({
        "type": "input",
        "data": {
            "title": "Enter Value",
            "message": "Please provide additional information:",
            "placeholder": "Type here..."
        }
    })

    return {"content": f"You entered: {user_input}"}
```

### Multi-Action Function

```python
class Action:
    # Define multiple action buttons
    actions = [
        {
            "id": "summarize",
            "name": "Summarize",
            "icon_url": "data:image/svg+xml;base64,..."
        },
        {
            "id": "translate",
            "name": "Translate",
            "icon_url": "data:image/svg+xml;base64,..."
        }
    ]

    async def action(
        self,
        body: dict,
        __id__=None,  # Receives the action ID
        **kwargs
    ) -> Optional[dict]:
        if __id__ == "summarize":
            return {"content": "Summary: ..."}
        elif __id__ == "translate":
            return {"content": "Translation: ..."}
```

---

## Common Patterns & Best Practices

### 1. Error Handling

```python
async def my_tool(self, query: str, __event_emitter__=None) -> str:
    try:
        result = await some_operation(query)
        return result
    except Exception as e:
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Error: {str(e)}", "done": True}
            })
        return f"Error occurred: {str(e)}"
```

### 2. HTTP Requests with aiohttp

```python
import aiohttp

async def fetch_data(self, url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
            else:
                raise Exception(f"HTTP {response.status}")
```

### 3. Mode-Adaptive Tool (Default vs Native)

```python
async def adaptive_tool(
    self,
    query: str,
    __event_emitter__=None,
    __metadata__=None
) -> str:
    mode = "default"
    if __metadata__:
        mode = __metadata__.get("params", {}).get("function_calling", "default")

    is_native = (mode == "native")

    # Status events work in both modes
    if __event_emitter__:
        await __event_emitter__({
            "type": "status",
            "data": {"description": "Working...", "done": False}
        })

    # Message events only work in Default mode
    if not is_native and __event_emitter__:
        await __event_emitter__({
            "type": "message",
            "data": {"content": "Processing your request...\n"}
        })

    return "Result"
```

### 4. File Handling

```python
def process_files(self, __files__: list = None) -> str:
    if not __files__:
        return "No files provided"

    results = []
    for file in __files__:
        file_type = file.get("type", "")
        file_name = file.get("name", "")
        file_url = file.get("url", "")
        results.append(f"Processed: {file_name}")

    return "\n".join(results)
```

### 5. Rich HTML Response

```python
from fastapi.responses import HTMLResponse

def create_chart(self, data: str) -> HTMLResponse:
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <div id="chart">Chart visualization here</div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, headers={"Content-Disposition": "inline"})
```

---

## Key Documentation Sources

When creating plugins, reference these resources:
- https://docs.openwebui.com/features/plugin/tools/
- https://docs.openwebui.com/features/plugin/functions/
- https://docs.openwebui.com/features/plugin/functions/pipe/
- https://docs.openwebui.com/features/plugin/functions/filter/
- https://docs.openwebui.com/features/plugin/functions/action/
- https://openwebui.com/functions/ (community examples)
- https://openwebui.com/tools/ (community examples)

## Critical Reminders

1. **Type hints are required** for all tool method parameters
2. **Docstrings are required** - they become the tool description
3. **Use async** for methods that use `__event_emitter__`
4. **Message events break in Native mode** - only use status events for compatibility
5. **inlet/outlet only work in Filter class**, not Pipe class
6. **Set self.citation = False** before emitting manual citations
7. **Valves are optional** but highly encouraged for configurability
8. **Test in both Default and Native modes** if using event emitters

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .index import TranscriptIndexManager
from .render import (
    render_conversation,
    render_index_status,
    render_list_conversations,
    render_search_conversations,
    render_search_sessions,
    render_turn,
)

manager = TranscriptIndexManager()
server = FastMCP("transcript-search")


@server.tool()
def search_conversations(
    query: str,
    limit: int = 10,
    session_id: str = "",
    source: str = "",
    project: str = "",
    start_date: str = "",
    end_date: str = "",
    mode: str = "hybrid",
    context_window: int = 0,
) -> str:
    result = manager.search_conversations(
        query=query,
        limit=limit,
        session_id=session_id,
        source=source,
        project=project,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        context_window=context_window,
    )
    return render_search_conversations(result)


@server.tool()
def search_sessions(
    query: str,
    limit: int = 5,
    source: str = "",
    project: str = "",
    start_date: str = "",
    end_date: str = "",
    mode: str = "hybrid",
) -> str:
    result = manager.search_sessions(
        query=query,
        limit=limit,
        source=source,
        project=project,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
    )
    return render_search_sessions(result)


@server.tool()
def list_conversations(
    source: str = "",
    project: str = "",
    start_date: str = "",
    end_date: str = "",
    limit: int = 20,
    offset: int = 0,
) -> str:
    page = manager.list_conversations(
        source=source,
        project=project,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    return render_list_conversations(page)


@server.tool()
def read_turn(
    session_id: str,
    turn_number: int,
    include_records: bool = False,
    include_content: bool = True,
) -> str:
    try:
        turn = manager.get_turn(session_id, turn_number)
    except KeyError:
        return f"error: session not found: {session_id}"
    except IndexError:
        return f"error: turn out of range: {turn_number}"
    return render_turn(turn, include_records=include_records, include_content=include_content)


@server.tool()
def read_conversation(
    session_id: str,
    offset: int = 0,
    limit: int = 20,
    include_records: bool = False,
    include_content: bool = True,
    compact: bool = False,
) -> str:
    try:
        convo = manager.get_conversation(session_id)
    except KeyError:
        return f"error: session not found: {session_id}"
    return render_conversation(
        convo,
        offset=offset,
        limit=limit,
        include_records=include_records,
        include_content=include_content,
        compact=compact,
    )


@server.tool()
def index_status() -> str:
    return render_index_status(manager.get_index_status())

"""
Milestone 5 (v2): Textual TUI.

Same pipeline as main.py (transcript -> chunk -> embed -> retrieve/answer),
wrapped in a proper terminal UI instead of raw input()/print() loops.

Run with: uv run textual run app.py
      or: uv run python app.py
"""

from __future__ import annotations

import asyncio

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Markdown, Static

from chunker import chunk_transcript
from embedder import (
    DEFAULT_COLLECTION_NAME,
    EmbeddingError,
    embed_and_store_chunks,
    is_video_indexed,
)
from retriever import RetrievalError, ask_question
from transcript import TranscriptExtractionError, extract_transcript_from_url



class URLScreen(Screen):
    """First screen: paste a link, watch it get processed."""

    CSS = """
    #url-container { align: center middle; height: 100%; }
    #url-box { width: 70; border: round $accent; padding: 1 2; }
    #status-label { margin-top: 1; color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="url-container"):
            with Vertical(id="url-box"):
                yield Static("Paste a YouTube URL to get started")
                yield Input(placeholder="https://youtu.be/...", id="url-input")
                yield Static("", id="status-label")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "url-input" and event.value.strip():
            self.process_url(event.value.strip())

    @work(exclusive=True)
    async def process_url(self, url: str) -> None:
        status = self.query_one("#status-label", Static)
        input_box = self.query_one("#url-input", Input)
        input_box.disabled = True

        try:
            status.update("Fetching transcript...")
            transcript = await asyncio.to_thread(extract_transcript_from_url, url)

            already_indexed = await asyncio.to_thread(
                is_video_indexed, transcript.video_id, DEFAULT_COLLECTION_NAME
            )
            if already_indexed:
                status.update(f"Video {transcript.video_id} already indexed. Loading chat...")
            else:
                status.update("Chunking transcript...")
                chunks = await asyncio.to_thread(chunk_transcript, transcript)

                status.update(f"Embedding {len(chunks)} chunks (calling Gemini API)...")
                await asyncio.to_thread(embed_and_store_chunks, chunks)

                status.update("Done indexing. Loading chat...")

            await self.app.push_screen(ChatScreen(transcript.video_id))

        except (ValueError, TranscriptExtractionError, EmbeddingError) as e:
            status.update(f"Error: {e}")
            input_box.disabled = False
        except Exception as e:  # noqa: BLE001
            status.update(f"Unexpected error: {e}")
            input_box.disabled = False


class ChatScreen(Screen):
    """Second screen: scrollable chat log + question input for one video."""

    CSS = """
    #chat-log { height: 1fr; padding: 0 2; }
    #question-input { margin: 1 2; }
    .user-msg { color: $accent; text-style: bold; margin-top: 1; }
    .status-msg { color: $text-muted; }
    """

    BINDINGS = [("ctrl+n", "new_video", "New video")]

    def __init__(self, video_id: str) -> None:
        super().__init__()
        self.video_id = video_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat-log")
        yield Input(placeholder="Ask a question about this video...", id="question-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#question-input", Input).focus()
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(
            Static(
                f"Loaded video: {self.video_id}  "
                "(Ctrl+N for a new video, Ctrl+C to quit)",
                classes="status-msg",
            )
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "question-input" and event.value.strip():
            question = event.value.strip()
            event.input.value = ""
            self.ask(question)

    @work(exclusive=False)
    async def ask(self, question: str) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        input_box = self.query_one("#question-input", Input)
        input_box.disabled = True

        await log.mount(Static(f"Q: {question}", classes="user-msg"))
        thinking = Static("Thinking...", classes="status-msg")
        await log.mount(thinking)
        log.scroll_end(animate=False)

        try:
            result = await asyncio.to_thread(ask_question, question, video_id=self.video_id)

            answer_md = f"**A:** {result.answer}"
            if result.sources:
                src_lines = []
                for c in result.sources:
                    m1, s1 = divmod(int(c.start), 60)
                    m2, s2 = divmod(int(c.end), 60)
                    preview = c.text[:80].replace("\n", " ")
                    src_lines.append(f"- `[{m1:02d}:{s1:02d}-{m2:02d}:{s2:02d}]` {preview}...")
                answer_md += "\n\n*Sources:*\n" + "\n".join(src_lines)

            await thinking.remove()
            await log.mount(Markdown(answer_md))

        except RetrievalError as e:
            thinking.update(f"Error: {e}")
        finally:
            input_box.disabled = False
            input_box.focus()
            log.scroll_end(animate=False)

    def action_new_video(self) -> None:
        self.app.pop_screen()


class YTQnaApp(App):
    """A terminal chat app for asking questions about a YouTube video."""

    TITLE = "YouTube Q&A"
    BINDINGS = [("ctrl+c", "quit", "Quit"), ("ctrl+q", "quit", "Quit")]

    def on_mount(self) -> None:
        self.push_screen(URLScreen())


if __name__ == "__main__":
    YTQnaApp().run()
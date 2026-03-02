"""LLM-based spatial reasoning using Claude."""

from __future__ import annotations

import logging

import anthropic

from scene.models import SceneGraph

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing 3D scene graphs from images.

COORDINATE FRAME:
- X-axis: points RIGHT (positive = right side of image)
- Y-axis: points DOWN (positive = lower in image) — THIS IS IMPORTANT
- Z-axis: points AWAY from camera (positive = further from camera)
- All measurements are in METERS

When answering:
- Always reference specific metric measurements
- "above" means SMALLER Y value (Y points down)
- "below" means LARGER Y value
- "left" means SMALLER X value
- "right" means LARGER X value
- "closer" means SMALLER Z value
- "further" means LARGER Z value
- Give distances between objects when relevant
- Be precise but conversational
"""

MAX_CONVERSATION_TURNS = 50


class SpatialReasoner:
    """Send scene graph to Claude for spatial Q&A with conversation history."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        self.model = model
        self.conversation_history: list[dict[str, str]] = []
        self._scene_graph: SceneGraph | None = None

    def set_scene(self, scene_graph: SceneGraph) -> None:
        """Set the scene graph and reset conversation history.

        Parameters
        ----------
        scene_graph:
            The scene graph to reason about.
        """
        self._scene_graph = scene_graph
        self.conversation_history = []
        logger.info(
            "Scene set: %d objects, %d relations.",
            len(scene_graph.objects),
            len(scene_graph.relations),
        )

    def ask(self, question: str) -> str:
        """Ask a spatial question about the current scene.

        Maintains conversation history so follow-up questions can reference
        previous context.

        Parameters
        ----------
        question:
            Natural-language spatial question about the scene.

        Returns
        -------
        str
            The assistant's answer.

        Raises
        ------
        RuntimeError
            If no scene graph has been set via :meth:`set_scene`.
        """
        if self._scene_graph is None:
            raise RuntimeError("No scene graph set. Call set_scene() first.")

        # Build messages list
        messages: list[dict[str, str]] = []

        # First message always includes the scene graph context
        scene_text = self._scene_graph.to_prompt_text()
        scene_context = (
            "Here is the 3D scene graph for the image you are analyzing:\n\n"
            f"{scene_text}\n\n"
            "Use this scene graph to answer spatial questions. "
            "Always cite specific measurements from the data."
        )

        if not self.conversation_history:
            # First turn: combine scene context with the question
            user_content = f"{scene_context}\n\nQuestion: {question}"
            messages.append({"role": "user", "content": user_content})
        else:
            # Subsequent turns: scene context was in the first message
            messages.append({
                "role": "user",
                "content": scene_context + "\n\nQuestion: " + self.conversation_history[0]["q"],
            })
            messages.append({
                "role": "assistant",
                "content": self.conversation_history[0]["a"],
            })
            for turn in self.conversation_history[1:]:
                messages.append({"role": "user", "content": turn["q"]})
                messages.append({"role": "assistant", "content": turn["a"]})
            messages.append({"role": "user", "content": question})

        # Trim conversation history to prevent context overflow
        if len(self.conversation_history) > MAX_CONVERSATION_TURNS:
            self.conversation_history = self.conversation_history[-MAX_CONVERSATION_TURNS:]

        logger.debug(
            "Sending %d messages to %s (question: %.80s...)",
            len(messages),
            self.model,
            question,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except anthropic.APIError as exc:
            logger.error("Claude API error: %s", exc)
            raise

        answer = response.content[0].text

        # Store in conversation history
        self.conversation_history.append({"q": question, "a": answer})

        logger.info(
            "Got answer (%d chars, %d input tokens, %d output tokens).",
            len(answer),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return answer

    def reset(self) -> None:
        """Clear conversation history (keeps scene graph)."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")

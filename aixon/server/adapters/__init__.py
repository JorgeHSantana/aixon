"""Concrete ProtocolAdapters (OpenAI, Anthropic). Each translates a wire
dialect to and from the neutral Message/Chunk types and declares the routes it
serves. New wire styles are new modules here — never edits to the Server."""

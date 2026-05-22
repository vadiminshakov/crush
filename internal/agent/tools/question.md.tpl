Ask the user a clarifying question and wait for their response before continuing.

<when_to_use>
Use when you need information from the user that you cannot determine on your own, and proceeding without it would risk doing the wrong thing or producing an incorrect result:
- The task is ambiguous and multiple valid interpretations exist
- A key decision depends on user preference (e.g. naming, style, approach)
- You are about to make a hard-to-reverse change and need confirmation of intent

Do not use for questions you can answer yourself by reading the codebase, documentation, or applying reasonable defaults.
</when_to_use>

<usage_notes>
- question: The question to ask the user. Keep it concise and specific.
- options: An optional list of 1–4 suggested answers. The user may select one or type a custom answer instead.
</usage_notes>

<behavior>
The tool blocks until the user responds. Their answer is returned as plain text.
</behavior>

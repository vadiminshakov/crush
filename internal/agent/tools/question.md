Ask the user a clarifying question and wait for their response before continuing.

<when_to_use>
When ending your turn of the conversation with a question,
pose it with this tool rather than as plain text.
</when_to_use>

<usage_notes>
- question: The question to ask the user. Keep it concise and specific.
- options: An optional list of 1–4 suggested answers. The user may select one or type a custom answer instead.
- allow_multiple: Set to true when the user may select multiple provided options. This requires options and disables custom free-text answers.
</usage_notes>

<behavior>
The tool blocks until the user responds. Single-answer questions return the
answer as plain text. Multiple-answer questions return a JSON array of selected
option strings.
</behavior>

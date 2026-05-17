-- +goose Up
-- +goose StatementBegin
CREATE TABLE IF NOT EXISTS goals (
    scope_id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    version INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (scope_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE TRIGGER IF NOT EXISTS update_goals_updated_at
AFTER UPDATE ON goals
BEGIN
    UPDATE goals SET updated_at = strftime('%s', 'now'), version = version + 1
    WHERE scope_id = new.scope_id;
END;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP TRIGGER IF EXISTS update_goals_updated_at;
DROP TABLE IF EXISTS goals;
-- +goose StatementEnd

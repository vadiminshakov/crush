-- name: GetGoalByScopeID :one
SELECT *
FROM goals
WHERE scope_id = ? LIMIT 1;

-- name: CreateOrReplaceGoal :one
INSERT INTO goals (
    scope_id,
    goal_id,
    objective,
    status,
    version,
    created_at,
    updated_at
) VALUES (
    ?,
    ?,
    ?,
    ?,
    1,
    strftime('%s', 'now'),
    strftime('%s', 'now')
)
ON CONFLICT(scope_id) DO UPDATE SET
    goal_id = excluded.goal_id,
    objective = excluded.objective,
    status = excluded.status,
    version = goals.version + 1,
    updated_at = strftime('%s', 'now')
RETURNING *;

-- name: UpdateGoalStatus :one
UPDATE goals
SET
    status = ?,
    updated_at = strftime('%s', 'now')
WHERE scope_id = ? AND goal_id = ?
RETURNING *;

-- name: DeleteGoal :exec
DELETE FROM goals
WHERE scope_id = ?;

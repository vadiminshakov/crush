-- name: GetGoalBySessionID :one
SELECT *
FROM goals
WHERE session_id = ? LIMIT 1;

-- name: CreateGoal :one
INSERT INTO goals (
    session_id,
    goal_id,
    objective,
    status,
    created_at,
    updated_at,
    active_seconds
) VALUES (
    ?,
    ?,
    ?,
    ?,
    strftime('%s', 'now'),
    strftime('%s', 'now'),
    0
)
RETURNING *;

-- name: UpdateGoalStatus :one
UPDATE goals
SET
    status = @status,
    updated_at = strftime('%s', 'now')
WHERE session_id = @session_id AND goal_id = @goal_id AND status = @from_status
RETURNING *;

-- name: PauseActiveGoals :execrows
UPDATE goals
SET
    status = 'paused',
    updated_at = strftime('%s', 'now')
WHERE status = 'active';

-- name: AccumulateActiveTime :exec
UPDATE goals
SET
    active_seconds = active_seconds + (strftime('%s', 'now') - updated_at),
    updated_at = strftime('%s', 'now')
WHERE session_id = ? AND status = 'active';

-- name: DeleteGoal :execrows
DELETE FROM goals
WHERE session_id = ? AND goal_id = ?;

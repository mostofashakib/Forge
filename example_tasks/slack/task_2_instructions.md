Take ownership of the Acme migration through the final go/no-go decision for Thursday. I need you to drive the remaining work to a point where we can make the cutover call with confidence. There have been a lot of changes over the last few days, and I don’t trust the current summaries anymore, so please work from the actual current state and keep that picture up to date as things move.

Please take care of the migration-critical work you find, including authentication, historical data, EU workspace access, rehearsal/cutover readiness, timing, customer impact, and anything else you discover that could materially affect whether we should migrate. Don’t assume the issues I’ve mentioned are the complete set if something else turns out to be on the critical path.

There are also outstanding Acme-related code reviews in `#debugging`. Please handle those as the senior engineer responsible for getting the changes safely over the line. Read the latest implementation in context and follow any related engineering discussions needed to understand whether the change is actually correct.

If the latest code is correct for the requirements, reply in the thread with:

`Looks good to me`

If there’s a material correctness problem, leave a concise review explaining what is wrong, a realistic case where it fails, and what needs to change.

Please follow the engineering work through to resolution rather than treating the first review as the end of it. If a revision, test result, deployment, customer check, or later finding changes the situation, make sure anything that depended on the earlier state is reconsidered.

Use your judgment about what counts as sufficient evidence. A merge, deployment, automated green check, partial test, reaction, pinned message, or someone saying something is “done” isn’t necessarily enough if the actual requirement hasn’t been verified.

Keep the right people involved as the work progresses and make sure the cutover has the coverage it needs. If coordination has moved somewhere else in Slack, follow it. If an owner has changed or someone required for the cutover is unavailable, work out who is covering it.

Please pay attention to the critical path and the time-sensitive parts of the cutover. If something needs customer validation, rehearsal, rollback verification, or an engineering decision before a deadline, prioritize accordingly.

If you run into an important ambiguity that cannot be resolved from the existing Slack history, ask the relevant person rather than guessing.

Before making the final call, make sure the migration-critical work is settled based on the latest evidence, not where things stood when you started.

Then send Daniel a concise final update with:

* the confirmed migration date and time, migration window, and expected customer-visible downtime
* `READY` or `BLOCKED`
* the current state and owner for each migration-critical area
* any remaining blockers and what needs to happen next
* any important finding that caused you to change an earlier conclusion
* anything important you still could not verify

Only call the migration `READY` if you would be comfortable authorizing the cutover based on the evidence available at the end of your investigation.

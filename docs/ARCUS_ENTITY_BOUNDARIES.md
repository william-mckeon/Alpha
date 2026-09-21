# Arcus and environment boundaries

Arcus is a persistent entity. The play area is an independent environment. A PlaySession coordinates their interaction; neither renderer owns simulation state.

| Owner | State and responsibilities |
| --- | --- |
| Embodiment | Stable entity_id, name, facing, radius, posture target, transition height, derived 12 joint coordinates. No room coordinates. |
| EmbodimentStore | Versioned body.json with atomic replace, fsync before replace, and one process owner per state directory. Corruption fails closed rather than silently creating a new identity. |
| Playroom | environment_id, dimensions, four walls, reset generation, placements keyed by entity_id, human marker. No body construction or posture state. |
| PlaySession | One 0.1-second clock, pause, action validation/routing, cues and results. Joins body and environment snapshots. |
| HTTP application | Serialized commands and clock updates, copy-before-commit, persistence before publication, request IDs and bounded interaction log. |
| Renderers | Environment draws the area. Arcus renderer takes only body state and pixel placement. The page composes them. |

The API snapshot version is arcus-playroom-v2. arcus contains entity_id and body fields; environment contains environment_id and placements. Old x/y body fields and room/balls/tv fields are removed. Clients must reload the updated page. Removed roll and tv commands return 400 without mutation.

Reset keeps the environment ID, increments its generation, recenters placements and the human marker, and clears location-dependent cues. It preserves Arcus's entire body record and the session's pause setting. Replacing an environment through PlaySession.replace_environment detaches the old placement and attaches the same body to a new environment ID. No duplicate body is created. There is no public environment-replacement button yet.

Restarting with the same state root restores Arcus's identity, facing and exact transition state, then creates a fresh environment and interaction log. Transition targets continue on the session clock. This is persistence of body state, not model memory, learning, or weights. Room positions and human cues are deliberately not persisted as body properties.

The CLI uses runs/arcus_playroom/entity-state by default; --state-root selects another directory. A different directory creates an independent body. The viewer role does not open this store. The local simulation and viewer retain their HTTP boundary; no GUI dependencies or Docker stack changes are required.

Bodies use abstract kinematic poses, not articulated physics. The existing model is not connected. Calls and encouragement remain recorded human cues. Desktop dragging is outside this implementation.

Model-facing tools now live in body_tools.py: observe_body and body_action. The optional separately authenticated endpoint exposes only these tools. Reset, pause, participant placement and feedback remain human controls. See ARCUS_BODY_TOOLS.md for the wire contract, activation and executor boundary.

Persistence errors on commands leave the published world and command log unchanged. Clock persistence errors stop updates and make readiness fail. Restart after fixing storage. Atomic replacement protects the prior file from partial writes; full disk/controller power-loss guarantees depend on the filesystem.
## Body-v2 extension

Persistent body state now includes twelve normalized leg joints, four head/eye direction values and one shared eyelid value. Placement and viewing permission remain separate. The conversation store is independent of motor/visual state; awake closed eyes do not prevent message availability. Human messages are not motor commands. See [contracts](../specs/0037-embodied-body-senses-and-gaze.md).

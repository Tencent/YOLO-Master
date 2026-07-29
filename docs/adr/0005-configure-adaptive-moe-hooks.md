# ADR-0005: Configure Adaptive MoE Variants with Ordered Hooks

## Status

Accepted for the compatibility-first migration slice.

## Context

The AdaptiveGateMoE family accumulated a deep inheritance chain while most
later variants only add one or more routing/fusion stages: detail conditioning,
pyramid context, and residual feature refinement. YAML files and checkpoints
still refer to the historical class names, so an immediate class replacement
would break model construction and serialized state-dict paths.

## Decision

Keep `AdaptiveGateMoE` as the shared execution owner and add an ordered
`RouterHook` policy list with a small registry. Hook policy objects are not
`nn.Module` instances; learnable components remain attached under the existing
`detail_gate`, `context_mixer`, `feature_refiner`, `feature_gate`, and
`refine_scale` attributes. Historical visual classes become compatibility
wrappers that select the corresponding hook list, while retaining their class
names, constructor arguments, and state-dict keys.

The first supported stages are:

- `pre_route`: transform the dynamic branch before router logits are computed;
- `post_fusion`: transform the concatenated static and dynamic expert output.

Hooks execute in declaration order. Duplicate names and unknown registry names
are rejected during construction. Existing classes continue to use their
historical forward entry points through the shared visual helper.

## Consequences

This reduces new-variant code to a hook declaration and leaves one common
forward contract to maintain. Ablations can disable a hook without creating a
new subclass, and custom integrations can register a hook factory. The policy
list itself is runtime metadata and is intentionally not serialized; component
weights retain historical names and remain checkpoint-compatible.

The migration does not remove old classes or alter existing YAML files. Export
and pruning behavior remain unchanged until their separate contracts are
designed and tested.

## Risks and Mitigations

- Hook ordering can change behavior; declaration-order tests and explicit
  stage contracts make the order auditable.
- A hook may reference a missing component; built-in hook resolution creates
  the required component before execution.
- External code may depend on a concrete subclass; compatibility wrappers and
  unchanged public names preserve that surface during the retirement window.

# Limitations

This policy is blind. It should not be expected to solve terrain that requires
anticipation, such as tall stairs or abrupt obstacles higher than the learned
foot-clearance strategy.

Known boundaries:

- no vision
- no height scan at runtime
- no online adaptation module
- no explicit terrain classifier
- no guarantee under Wi-Fi DDS unless multicast and peer traffic are validated

The intended terrain class is rough terrain that can be handled through contact,
history, and recovery behavior rather than exteroceptive planning.

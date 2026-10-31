# Palm sensing digital twin

The supplied [request](../../REQUEST.md) is the specification. Implement a local
Streamlit/Plotly application backed by pure Python functions and a single JSON-safe
configuration. No existing application or Senior Design PDF was supplied.

Use explicit seeded orchard coordinates, independent RF reachability and scheduling,
and separate Architecture A/B network loads. Support analytical area quadrature and
Monte Carlo shadowing. Use exact LoRa airtime and documented sensitivity tables;
label propagation, inference, memory and battery predictions as unverified models.

Three implementation boundaries: radio/network; CNN/energy/orchestration/research;
dashboard. Exports include the full configuration, numerical results, provenance and
limitations. Measured data can calibrate numerical assumptions but cannot establish
whole-device feasibility. Never synthesize a benchmark or claim hardware validation.

Streamlit is selected because the user prefers Python scientific engines. A custom
frontend would add maintenance without improving these scientific calculations.

The user explicitly requests immediate implementation without preliminary questions,
so this design records implementation decisions rather than adding an approval gate.

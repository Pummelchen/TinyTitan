// Which engine serves a model, as far as the generation path needs to know.
//
// Moved out of `ModelCatalog.Backend` (2026-10-02, phase A1 of
// `docs/plan-embedded-library.md`) so this target can name the backend without
// depending on the server's catalog. The only question the orchestrator asks of
// it is whether a prompt cache applies at all -- a CPU snapshot keeps none --
// and that is all this type carries. The catalog itself stays in
// `TinyTitanServerCore`, which keeps the old spelling as
// `ModelCatalog.Backend`.
package enum ModelBackend: String, Sendable {
    case gpu, cpu
}

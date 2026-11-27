/// The engine-factory surface shared by the CLI, the server and an embedder.
///
/// It is a namespace, not an engine: the selection table, the variant
/// factories and the plugin registry live in extensions over it, and every
/// engine it hands out comes from a plugin (TPP/1). There is no fallback
/// engine — an install no plugin serves is refused by name (TT-142).
public enum EngineFactories {}

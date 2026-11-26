/** Compare raw plugin configs using schema metadata without executing config hooks or validators. */
import type { Plugin } from '@deepseek-ai/cordis';
/**
 * Compare two raw configs, treating schema-declared volatile fields at fixed object paths as equal and absent objects as their schema default.
 * Schema backedges, expressions, unknown fields and opaque values keep strict raw equality; an absent or non-Schemastery schema compares everything raw.
 * @param previous - previous raw config.
 * @param next - next raw config.
 * @param schema - the plugin's config schema.
 * @returns Whether the configs differ at most in volatile fields, without evaluating expressions, validating config or modifying inputs.
 * @internal
 */
export declare function equalExceptVolatile(previous: unknown, next: unknown, schema: Plugin.Runtime['Config']): boolean;
//# sourceMappingURL=diff.d.ts.map
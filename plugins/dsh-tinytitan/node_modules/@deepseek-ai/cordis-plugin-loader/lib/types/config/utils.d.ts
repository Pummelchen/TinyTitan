/** Evaluate a JavaScript expression against a loader context scope. */
export declare const evaluate: ((ctx: object, expr: string) => any);
/** Recursively replace YAML `!!js` expression nodes with evaluated values. */
export declare function interpolate(ctx: object, value: any): any;
/** Return true when a value is a serialized loader JavaScript expression. */
export declare function isJsExpr(value: any): value is JsExpr;
/** Serialized JavaScript expression produced by the include YAML tag. */
export interface JsExpr {
    __jsExpr: string;
}
/** Source position of a config value: 1-based, with columns in UTF-16 code units. */
export interface SourceLocation {
    /** Path of the config file that writes the value. */
    filename: string;
    line: number;
    column: number;
}
/**
 * Format a source position as `file:line:column`.
 * @param location - Position to format.
 * @returns The formatted position.
 */
export declare function formatLocation(location: SourceLocation): string;
/**
 * Format one validation issue as an indented list item.
 * @param message - Issue description.
 * @param location - Source position of the value, when known.
 * @param path - Dotted option path of the value, when not the root.
 * @returns The list item, with its path and position in parentheses.
 */
export declare function formatIssue(message: string, location?: SourceLocation, path?: string): string;
/** One rejected entry option. */
export interface Issue {
    message: string;
    /** Path from the entry options to the rejected value; empty for the entry itself. */
    path: PropertyKey[];
}
/** Map an option path to the source position that writes it. */
export type Locate = (path: PropertyKey[]) => SourceLocation | undefined;
/**
 * Format a titled list of validation issues.
 * @param title - First line of the message.
 * @param issues - Issues to list in order.
 * @param locate - Resolves each issue path to a source position.
 * @returns The multi-line message.
 */
export declare function formatIssues(title: string, issues: readonly Issue[], locate?: Locate): string;
/**
 * Replace an error's message and stack frames, so the error points at the config source instead of the loader internals.
 * @param error - Error to rewrite in place.
 * @param message - New message.
 * @param frames - Stack frames listed after the message line.
 */
export declare function relocate(error: Error, message: string, frames: string[]): void;
/** Entry options rejected by `validateOptions()`; `issues` lists every rejected option. */
export declare class EntryValidationError extends TypeError {
    issues: Issue[];
    name: string;
    constructor(issues: Issue[]);
}
/**
 * Check entry options before an entry is created or updated. Explicit ids may contain `sep`.
 * @param options - Candidate entry options.
 * @param sep - Tree id separator; kept for the upstream signature.
 * @param deep - Also check the entries of a group's `config`.
 * @returns Every issue found; empty when the options are valid.
 */
export declare function validateOptions(options: unknown, sep: string, deep?: boolean): Issue[];
//# sourceMappingURL=utils.d.ts.map
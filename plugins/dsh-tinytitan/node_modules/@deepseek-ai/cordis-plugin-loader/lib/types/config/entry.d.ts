import { Context, Fiber, Inject, ValidationError } from '@deepseek-ai/cordis';
import { Loader } from '../index.ts';
import { EntryGroup } from './group.ts';
import { EntryTree } from './tree.ts';
/** Serialized plugin entry options stored in loader config files. */
export interface EntryOptions {
    /** Stable id inside the containing entry tree. */
    id: string;
    /** Module specifier imported by the entry tree. */
    name: string;
    /** Config passed to the plugin. */
    config?: any;
    /** Marks this entry as a nested group. */
    group?: boolean | null;
    /** Prevents this entry and descendants from running. */
    disabled?: boolean | null;
    /** Required services or service intercept config for this entry. */
    inject?: Inject | null;
}
/** One configured plugin node inside an `EntryTree`. */
export declare class Entry {
    loader: Loader;
    static readonly key: unique symbol;
    ctx: Context;
    fiber?: Fiber;
    /** Raw import result before export normalization; HMR updates it after a successful reload. */
    moduleNamespace: unknown;
    parent: EntryGroup;
    options: EntryOptions;
    subgroup?: EntryGroup;
    subtree?: EntryTree;
    _initTask?: Promise<void>;
    constructor(loader: Loader);
    /**
     * Merge partial options into `target`; a nullish value deletes its key.
     * @param target - Options to modify in place.
     * @param options - Changes to apply.
     * @returns `target`.
     */
    static merge(target: Partial<EntryOptions>, options: Partial<EntryOptions>): Partial<EntryOptions>;
    get context(): Context;
    get id(): string;
    /** True when this entry or any owning parent entry is disabled. */
    get disabled(): boolean;
    /**
     * Effective disabled state: a `!!js` expression evaluates against the loader
     * context. The raw node stays in the options, so write-back keeps the form.
     */
    private _resolveDisabled;
    evaluate(expr: string): any;
    private _patchContext;
    refresh(): Promise<void>;
    /** Merge new options, restart as needed, and persist through the parent tree. */
    update(options: Partial<EntryOptions>, create?: boolean, force?: boolean): Promise<void>;
    /**
     * Parse a volatile-only raw config change and commit its values into the running fiber's references.
     * An invalid candidate is logged and leaves the running references unchanged; the raw config stays retained for the next activation.
     * @returns `false` when an ordinary effective value changed, so the caller applies the ordinary update lifecycle.
     */
    private _commitVolatile;
    /**
     * Where a value of this entry's options is written.
     * @param path - Path inside the entry options; empty for the entry itself.
     * @returns The source position, or undefined when the tree has no source.
     */
    getLocation(path?: PropertyKey[]): import("./utils.ts").SourceLocation | undefined;
    /** Stack frames from this entry out through its owning entries, each with its source position when known. */
    getOuterStack: () => string[];
    /**
     * Rewrite a config validation error to list each issue with its source position, when this entry has a source.
     * @param error - Error to rewrite in place.
     */
    traceValidation(error: ValidationError): void;
    private _reportImport;
    /** Import and start the configured plugin if it is not already running. */
    init(): Promise<void>;
    private _init;
}
//# sourceMappingURL=entry.d.ts.map
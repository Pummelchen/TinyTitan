import type { EntryChange, EntryOptions } from '@deepseek-ai/cordis-plugin-loader';
import { type Dict } from '@deepseek-ai/cosmokit';
/** One tree-side mutation that has not reached the file yet, keyed by entry id. */
export type JournalRecord = JournalRecord.Remove | JournalRecord.Upsert;
export declare namespace JournalRecord {
    interface Remove {
        kind: 'remove';
    }
    interface Upsert {
        kind: 'upsert';
        /** The entry did not exist when the tree first reported it. */
        created: boolean;
        /**
         * The group the tree placed the entry in (`null` for the root), or
         * `undefined` when the tree did not relocate it.
         */
        parent: string | null | undefined;
        /** Index within `parent`; only meaningful alongside a `parent`. */
        position: number;
        /**
         * Per-key delta relative to the options the entry had when the change
         * was reported; `undefined` deletes a key. Collected at report time: the
         * live tree is reconciled asynchronously and may lag behind the file.
         */
        changes: Dict;
    }
}
export type Journal = Map<string, JournalRecord>;
export declare function diffOptions(legacy: EntryOptions, options: EntryOptions): Dict;
/** Fold a later record into an earlier one for the same entry. */
export declare function mergeRecords(older: JournalRecord | undefined, newer: JournalRecord): JournalRecord | undefined;
export declare function merge(journal: Journal, id: string, record: JournalRecord): void;
export declare function record(journal: Journal, change: EntryChange, parent: string | null): void;
export interface FlatEntry {
    parent: string | null;
    position: number;
    options: EntryOptions;
    list: EntryOptions[];
}
/** Index every entry (including nested group members) by id. */
export declare function flatten(data: EntryOptions[], parent?: string | null, result?: Map<string, FlatEntry>): Map<string, FlatEntry>;
export declare function applyChanges(options: EntryOptions, changes: Dict, filter?: (key: string) => boolean): void;
export declare function detach(data: EntryOptions[], id: string): EntryOptions | undefined;
/**
 * Insert `options` under `parent` (a group id, or `null` for the root) at
 * `position`, clamped to the group's length. Returns false when the parent
 * group does not exist.
 */
export declare function place(data: EntryOptions[], options: EntryOptions, parent: string | null, position: number): boolean;
/**
 * Overlay a journal onto `data` in place. `filter(id, key)` restricts which
 * changes apply; `key` is `null` when asking about the entry as a whole
 * (creation, removal, relocation).
 */
export declare function applyJournal(data: EntryOptions[], journal: Journal, filter?: (id: string, key: string | null) => boolean, warn?: (format: string, ...args: any[]) => void): EntryOptions[];
export interface Conflict {
    id: string;
    reason: string;
}
/**
 * Three-way reconciliation of a journal against a new file state. The file
 * wins every conflict: conflicting keys are dropped from the journal so that
 * the tree follows the file, and each drop is reported.
 *
 * `fileOwned(id, key)` scopes the check to what the file can actually express
 * (patch-owned keys and patch-inserted entries live elsewhere).
 */
export declare function reconcile(journal: Journal, base: Map<string, FlatEntry>, theirs: Map<string, FlatEntry>, fileOwned: (id: string, key: string | null) => boolean): Conflict[];
//# sourceMappingURL=journal.d.ts.map
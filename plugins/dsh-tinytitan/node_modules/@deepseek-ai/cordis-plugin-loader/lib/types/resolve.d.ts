/** Resolve a specifier to its URL in one configuration project. */
export type Resolve = (specifier: string) => string;
/**
 * Create or reuse a native ESM resolver anchored in the config project's package scope.
 * @param baseUrl Config file URL; absent, non-file or unscoped URLs cannot supply a resolver.
 * @returns The project resolver, or undefined when no package scope exists or its helper cannot be written.
 */
export declare function createResolve(baseUrl: string | undefined): Promise<Resolve | undefined>;
//# sourceMappingURL=resolve.d.ts.map
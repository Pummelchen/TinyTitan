//
//  main.swift
//  EmbeddedDemo
//
//  The smallest program that proves the engine is usable from a package that is
//  not this repository: it resolves the dependency, links the `TinyTitan`
//  module and touches its public surface. With `--model` it opens an install
//  for real; without one it only proves the link, which is all CI can do -- the
//  model store is gitignored and no install ships with the repository.
//
//  What this deliberately cannot do yet is render a prompt or generate: the
//  session orchestration lives in `TinyTitanServerCore`, which drags in NIO.
//  That boundary is the first work item of docs/plan-embedded-library.md, and
//  this file is where it will be visible when it moves.
//

import Foundation
import Metal
import TinyTitan

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("EmbeddedDemo: \(message)\n".utf8))
    exit(1)
}

var modelDirectory: String?
var arguments = CommandLine.arguments.dropFirst().makeIterator()
while let argument = arguments.next() {
    switch argument {
    case "--model":
        guard let value = arguments.next() else { fail("--model needs a directory") }
        modelDirectory = value
    case "--help", "-h":
        print("usage: EmbeddedDemo [--model <ssdai install directory>]")
        exit(0)
    default:
        fail("unknown argument \(argument)")
    }
}

// The link itself: naming a public engine type is what forces the library to be
// linked rather than merely planned.
print("EmbeddedDemo: TinyTitan linked (\(String(describing: Model.self)))")

guard let modelDirectory else {
    print("no --model given: resolution and link verified, nothing loaded")
    exit(0)
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fail("no Metal device on this machine, so an install cannot be opened")
}

do {
    let model = try Model.load(
        directoryURL: URL(fileURLWithPath: modelDirectory),
        device: device
    )
    print("loaded: \(String(describing: type(of: model)))")
} catch {
    fail("load failed: \(error)")
}

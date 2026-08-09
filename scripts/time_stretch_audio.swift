#!/usr/bin/env swift

import AVFoundation
import Foundation

guard CommandLine.arguments.count == 5,
      let tempo = Float(CommandLine.arguments[3]),
      let outputFrames = AVAudioFramePosition(CommandLine.arguments[4]),
      tempo > 0,
      outputFrames > 0 else {
    FileHandle.standardError.write(
        Data("usage: time_stretch_audio.swift INPUT.wav OUTPUT.wav TEMPO OUTPUT_FRAMES\n".utf8)
    )
    exit(2)
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let input = try AVAudioFile(forReading: inputURL)
let format = input.processingFormat

let engine = AVAudioEngine()
let player = AVAudioPlayerNode()
let timePitch = AVAudioUnitTimePitch()
timePitch.rate = tempo
timePitch.pitch = 0
timePitch.overlap = 32

engine.attach(player)
engine.attach(timePitch)
engine.connect(player, to: timePitch, format: format)
engine.connect(timePitch, to: engine.mainMixerNode, format: format)

let block: AVAudioFrameCount = 4096
try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: block)
let output = try AVAudioFile(forWriting: outputURL, settings: format.settings)
try engine.start()
player.scheduleFile(input, at: nil)
player.play()

while output.length < outputFrames {
    let remaining = AVAudioFrameCount(min(AVAudioFramePosition(block), outputFrames - output.length))
    guard let buffer = AVAudioPCMBuffer(
        pcmFormat: engine.manualRenderingFormat,
        frameCapacity: remaining
    ) else {
        throw NSError(domain: "H3DraftTimeStretch", code: 1)
    }
    let status = try engine.renderOffline(remaining, to: buffer)
    switch status {
    case .success:
        try output.write(from: buffer)
    case .cannotDoInCurrentContext:
        continue
    case .insufficientDataFromInputNode:
        // AVAudioUnitTimePitch has a processing tail. Rendering silence through that tail is the
        // documented offline-engine behaviour and gives the requested exact delivery length.
        try output.write(from: buffer)
    case .error:
        throw NSError(domain: "H3DraftTimeStretch", code: 2)
    @unknown default:
        throw NSError(domain: "H3DraftTimeStretch", code: 3)
    }
}

player.stop()
engine.stop()

//
//  ViewController.swift
//  VisionML
//
//  Created by Alexis Soto on 11.18.18.
//  Copyright © 2018 Alexis Soto. All rights reserved.
//

import UIKit
import AVFoundation
import Vision
import CoreAudioKit

class ViewController: UIViewController {
    
    @IBOutlet weak var previewView: PreviewView!
    @IBOutlet weak var objectTextView: UITextView!
    
    var ae: AVAudioEngine?
    var player: AVAudioPlayerNode?
    var mixer: AVAudioMixerNode?
    var buffer: AVAudioPCMBuffer?
    
    //VocalSynth
    let vsynth = AVSpeechSynthesizer()

    // Live Camera Properties
    let captureSession = AVCaptureSession()
    var captureDevice:AVCaptureDevice!
    var devicePosition: AVCaptureDevice.Position = .back
    
    var requests = [VNRequest]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupVision()
        
        ae = AVAudioEngine()
        player = AVAudioPlayerNode()
        mixer = ae?.mainMixerNode;
        
        //Buffer stuff
        buffer = AVAudioPCMBuffer(pcmFormat: (player?.outputFormat(forBus: 0))!, frameCapacity: 400)

        // setup audio engine
        ae?.attach(player!)
        ae?.connect(player!, to: mixer!, format: player?.outputFormat(forBus: 0))
        do{
            try ae?.start()
        } catch {}
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        prepareCamera()
    }
    
    func setupVision() {
        
        // Rectangle Request
        let rectangleDetectionRequest = VNDetectRectanglesRequest(completionHandler: handleRectangles)
        rectangleDetectionRequest.minimumSize = 0.1
        rectangleDetectionRequest.maximumObservations = 1
        
        // Object Classification
        guard let visionModel = try? VNCoreMLModel(for: Inceptionv3().model) else {fatalError("cant load Vision ML model")}
        
        let classificationRequest = VNCoreMLRequest(model: visionModel, completionHandler: handleClassification)
        classificationRequest.imageCropAndScaleOption = .centerCrop
        self.requests = [rectangleDetectionRequest, classificationRequest]
    }
    
    func handleRectangles (request:VNRequest, error:Error?) {
        DispatchQueue.main.async {
            self.drawVisionRequestResults(request.results as! [VNRectangleObservation])
        }
    }
    
    func drawVisionRequestResults (_ results:[VNRectangleObservation]) { //Includes sound processing depending on the size of the square.
        previewView.removeMask()
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -self.previewView.frame.height)
        let translate = CGAffineTransform.identity.scaledBy(x: self.previewView.frame.width, y: self.previewView.frame.height)
        for rectangle in results {
            let rectangleBounds = rectangle.boundingBox.applying(translate).applying(transform)
            previewView.drawLayer(in: rectangleBounds)
            buffer?.frameLength = UInt32(rectangleBounds.height)
            let sr: Float = Float((mixer?.outputFormat(forBus: 0).sampleRate)!)
            let n_channels = mixer?.outputFormat(forBus: 0).channelCount
            let number = Float(rectangleBounds.height * rectangleBounds.width) //Are of square created will define the harmonic content of pitch.
            
            if rectangleBounds.isEmpty == false {
                for i in stride(from:0, to: Int((buffer?.frameLength)!), by: Int(n_channels!)) {
                    let val = sin(number*Float(i)*2*Float(Double.pi)/sr)
                    buffer?.floatChannelData?.pointee[i] = val*0.7
                    self.player?.play()
                    self.player?.scheduleBuffer(self.buffer!, at: nil, options: .init(rawValue: 1000), completionHandler: nil)
                }
            }
        }
    }

    func handleClassification (request:VNRequest, error:Error?) {
        guard let observations = request.results else {print("no results:\(String(describing: error?.localizedDescription))"); return}
        
        let classifcations = observations[0...4]
        .compactMap({$0 as? VNClassificationObservation})
        .filter({$0.confidence > 0.25})
        .map({$0.identifier})
        
        for classification in classifcations {
            DispatchQueue.main.async {
                self.objectTextView.text = classification
                
                //Speak variables
                let utterance = AVSpeechUtterance(string: classification)
                utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
                //utterance.rate = 0.5
                self.vsynth.speak(utterance)
            }
        }
    }
}

//
//  PreviewView.swift
//  VisionDetection
//

import UIKit
import AVFoundation

class PreviewView: UIView {
    
    private var maskLayer = [CAShapeLayer]()
    
    // MARK: AV capture properties
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        return layer as! AVCaptureVideoPreviewLayer
    }
    
    var session: AVCaptureSession? {
        get {
            return videoPreviewLayer.session
        }
        
        set {
            videoPreviewLayer.session = newValue
        }
    }
    
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }
    
    func drawLayer(in rect: CGRect) {
        
        let mask = CAShapeLayer()
        mask.frame = rect
        
        mask.backgroundColor = UIColor.green.cgColor
        mask.cornerRadius = 9
        mask.opacity = 0.3
        mask.borderColor = UIColor.lightGray.cgColor
        mask.borderWidth = 4.0
        
        maskLayer.append(mask)
        layer.insertSublayer(mask, at: 1)
    }
    
    func removeMask() {
        for mask in maskLayer {
            mask.removeFromSuperlayer()
        }
        maskLayer.removeAll()
    }
}
